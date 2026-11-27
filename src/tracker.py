"""
Tracker hybride :
- même classe YOLO ;
- proximité des centres ;
- recouvrement des bounding boxes (IoU) ;
- cohérence de profondeur.
"""

import numpy as np


def box_iou(box_a, box_b):
    """Calcule le recouvrement entre deux boîtes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = (
        intersection_width * intersection_height
    )

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def box_center(box):
    """Retourne le centre (x, y) d'une boîte."""

    x1, y1, x2, y2 = box

    return np.array([
        (x1 + x2) / 2,
        (y1 + y2) / 2
    ])


class CentroidTracker:
    def __init__(
        self,
        max_distance=90,
        max_missed=12,
        min_iou=0.10,
        max_depth_change_m=5.0
    ):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.min_iou = min_iou
        self.max_depth_change_m = max_depth_change_m

        self.next_id = 1
        self.tracks = {}

    def update(self, detections):
        """
        detections doit contenir :
        box, label et depth_m.
        """

        candidates = []

        for track_id, track in self.tracks.items():
            for detection_index, detection in enumerate(detections):
                if track["label"] != detection["label"]:
                    continue

                detection_box = detection["box"]
                detection_center = box_center(detection_box)

                center_distance = np.linalg.norm(
                    track["center"] - detection_center
                )

                iou = box_iou(
                    track["box"],
                    detection_box
                )

                old_depth = track["depth_m"]
                new_depth = detection.get(
                    "depth_m",
                    float("nan")
                )

                depth_difference = abs(
                    new_depth - old_depth
                )

                depths_valid = (
                    old_depth == old_depth
                    and new_depth == new_depth
                )

                # Si les deux profondeurs sont fiables mais très différentes,
                # il ne s'agit probablement pas du même objet.
                if (
                    depths_valid
                    and depth_difference > self.max_depth_change_m
                ):
                    continue

                # L'objet doit être suffisamment proche OU se recouvrir.
                if (
                    center_distance > self.max_distance
                    and iou < self.min_iou
                ):
                    continue

                # Coût faible = meilleure association.
                distance_cost = min(
                    center_distance / self.max_distance,
                    1.0
                )

                iou_cost = 1.0 - iou

                depth_cost = 0.0

                if depths_valid:
                    depth_cost = min(
                        depth_difference
                        / self.max_depth_change_m,
                        1.0
                    )

                cost = (
                    0.45 * iou_cost
                    + 0.35 * distance_cost
                    + 0.20 * depth_cost
                )

                candidates.append(
                    (cost, track_id, detection_index)
                )

        used_tracks = set()
        used_detections = set()

        # Attribution des meilleures correspondances.
        for _, track_id, detection_index in sorted(candidates):
            if track_id in used_tracks:
                continue

            if detection_index in used_detections:
                continue

            detection = detections[detection_index]

            self.tracks[track_id].update({
                "box": detection["box"],
                "center": box_center(detection["box"]),
                "label": detection["label"],
                "depth_m": detection.get(
                    "depth_m",
                    float("nan")
                ),
                "missed": 0
            })

            detection["track_id"] = track_id

            used_tracks.add(track_id)
            used_detections.add(detection_index)

        # Les objets non associés reçoivent un nouvel ID.
        for detection_index, detection in enumerate(detections):
            if detection_index in used_detections:
                continue

            track_id = self.next_id
            self.next_id += 1

            self.tracks[track_id] = {
                "box": detection["box"],
                "center": box_center(detection["box"]),
                "label": detection["label"],
                "depth_m": detection.get(
                    "depth_m",
                    float("nan")
                ),
                "missed": 0
            }

            detection["track_id"] = track_id

        # Compte les frames où un ID n'a pas été vu.
        for track_id in list(self.tracks):
            if track_id not in used_tracks:
                if track_id not in [
                    detection["track_id"]
                    for detection in detections
                ]:
                    self.tracks[track_id]["missed"] += 1

            if (
                self.tracks[track_id]["missed"]
                > self.max_missed
            ):
                del self.tracks[track_id]

        return detections