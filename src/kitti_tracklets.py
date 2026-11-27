"""Lecture et validation légère des annotations 3D KITTI Raw tracklets."""

from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _read_calibration_matrix(path, wanted_keys):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, data = line.split(":", 1)
        if key in wanted_keys:
            values[key] = np.fromstring(data.strip(), sep=" ")
    return values


def _compatible_labels(detection_label, ground_truth_label):
    groups = {
        "car": {"Car", "Van"},
        "truck": {"Truck"},
        "bus": {"Tram"},
        "person": {"Pedestrian", "Person_sitting"},
        "bicycle": {"Cyclist"},
        "cyclist": {"Cyclist"},
        "motorcycle": {"Cyclist"},
    }
    return ground_truth_label in groups.get(detection_label, set())


class TrackletValidator:
    """
    Associe une détection à un tracklet KITTI si son centre 3D projeté tombe
    dans la bounding box YOLO. Les résultats sont des mesures de validation,
    jamais des valeurs utilisées pour modifier l'estimation du modèle.
    """

    def __init__(self, tracklets_path, calib_velo_to_cam, calib_cam_to_cam):
        velo = _read_calibration_matrix(calib_velo_to_cam, {"R", "T"})
        cam = _read_calibration_matrix(calib_cam_to_cam, {"R_rect_00", "P_rect_02"})

        self.r_velo_to_cam = velo["R"].reshape(3, 3)
        self.t_velo_to_cam = velo["T"].reshape(3)
        self.r_rect = cam["R_rect_00"].reshape(3, 3)
        self.p_rect_02 = cam["P_rect_02"].reshape(3, 4)
        self.records_by_frame = defaultdict(list)
        self.last_tracklet_by_track = {}
        self._load(tracklets_path)

    def _load(self, tracklets_path):
        root = ET.parse(tracklets_path).getroot()
        tracklets = root.find("tracklets")
        if tracklets is None:
            raise ValueError("Balise <tracklets> introuvable dans le XML KITTI.")

        ground_truth_id = 0
        for item in tracklets.findall("item"):
            label = item.findtext("objectType", "Unknown")
            first_frame = int(item.findtext("first_frame", "0"))
            poses = item.find("poses")
            if poses is None:
                continue

            pose_index = 0
            for pose in poses.findall("item"):
                point_velo = np.array([
                    float(pose.findtext("tx", "nan")),
                    float(pose.findtext("ty", "nan")),
                    float(pose.findtext("tz", "nan")),
                ])
                projection = self._project_velodyne_point(point_velo)
                if projection is not None:
                    u, v, z_m = projection
                    self.records_by_frame[first_frame + pose_index].append({
                        "ground_truth_id": ground_truth_id,
                        "ground_truth_label": label,
                        "u": u,
                        "v": v,
                        "depth_m": z_m,
                    })
                pose_index += 1
            ground_truth_id += 1

        # Vitesse relative GT : variation de profondeur du même tracklet.
        previous_depth = {}
        for frame_index in sorted(self.records_by_frame):
            for record in self.records_by_frame[frame_index]:
                gt_id = record["ground_truth_id"]
                previous = previous_depth.get(gt_id)
                record["relative_mps"] = (
                    (record["depth_m"] - previous[1]) / ((frame_index - previous[0]) / 10.0)
                    if previous is not None and frame_index > previous[0]
                    else float("nan")
                )
                previous_depth[gt_id] = (frame_index, record["depth_m"])

    def _project_velodyne_point(self, point_velo):
        point_camera = self.r_velo_to_cam @ point_velo + self.t_velo_to_cam
        point_rectified = self.r_rect @ point_camera
        homogeneous = np.append(point_rectified, 1.0)
        pixel = self.p_rect_02 @ homogeneous

        if pixel[2] <= 0.0:
            return None
        return float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2]), float(point_rectified[2])

    def evaluate(self, frame_index, objects):
        """Ajoute à chaque objet la comparaison avec le tracklet correspondant."""
        candidates = self.records_by_frame.get(frame_index, [])
        used_ground_truth_ids = set()

        for obj in objects:
            x1, y1, x2, y2 = [float(value) for value in obj["box"]]
            width = max(x2 - x1, 1.0)
            height = max(y2 - y1, 1.0)
            margin_x = width * 0.15
            margin_y = height * 0.15

            matches = [
                record for record in candidates
                if record["ground_truth_id"] not in used_ground_truth_ids
                and _compatible_labels(obj["label"], record["ground_truth_label"])
                and x1 - margin_x <= record["u"] <= x2 + margin_x
                and y1 - margin_y <= record["v"] <= y2 + margin_y
            ]

            if not matches:
                obj["validation"] = None
                continue

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            track_id = obj.get("track_id")
            previous = self.last_tracklet_by_track.get(track_id)
            preferred_id = None

            if previous is not None:
                previous_id, previous_frame = previous
                if 0 < frame_index - previous_frame <= 5:
                    preferred_id = previous_id

            record = min(
                matches,
                key=lambda item: (
                    item["ground_truth_id"] != preferred_id,
                    (item["u"] - center_x) ** 2
                    + (item["v"] - center_y) ** 2,
                ),
            )

            if track_id is not None:
                self.last_tracklet_by_track[track_id] = (
                    record["ground_truth_id"],
                    frame_index,
                )
            used_ground_truth_ids.add(record["ground_truth_id"])
            estimated_depth = obj.get("depth_m", float("nan"))
            estimated_relative = obj.get("relative_speed_mps", float("nan"))
            gt_relative = record["relative_mps"]

            obj["validation"] = {
                **record,
                "depth_abs_error_m": (
                    abs(estimated_depth - record["depth_m"])
                    if np.isfinite(estimated_depth) else float("nan")
                ),
                "relative_speed_abs_error_kmh": (
                    abs(estimated_relative - gt_relative) * 3.6
                    if np.isfinite(estimated_relative) and np.isfinite(gt_relative)
                    else float("nan")
                ),
            }
