import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.calibration import StereoCalibration
from src.config import load_config
from src.detector import YOLODetector
from src.oxts import KittiCameraPoses, read_ego_velocity_camera
from src.pipeline import StereoSpeedPipeline
from src.speed import SpeedEstimator
from src.stereo import load_disparity_map
from src.kitti_tracklets import TrackletValidator


def find_stereo_pairs(left_dir, right_dir):
    left_dir = Path(left_dir)
    right_dir = Path(right_dir)

    left_images = sorted(left_dir.glob("*.png"))
    pairs = []

    for left_path in left_images:
        right_path = right_dir / left_path.name

        if right_path.exists():
            pairs.append((left_path, right_path))

    return pairs


def number_or_empty(value, digits=2):
    """Retourne une cellule CSV vide si la valeur est NaN."""

    if value != value:
        return ""

    return round(value, digits)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline KITTI professionnel : "
            "FoundationStereo ou StereoSGBM-WLS + YOLO + BoT-SORT/ReID + OXTS."
        )
    )

    parser.add_argument(
        "--left-dir",
        required=True,
        help="Dossier image_02/data"
    )

    parser.add_argument(
        "--right-dir",
        required=True,
        help="Dossier image_03/data"
    )

    parser.add_argument(
        "--oxts-dir",
        required=True,
        help="Dossier oxts/data"
    )

    parser.add_argument(
        "--calib",
        required=True,
        help="Fichier calib_cam_to_cam.txt"
    )

    parser.add_argument(
        "--tracklets",
        default=None,
        help="tracklet_labels.xml : active la validation KITTI"
    )

    parser.add_argument(
        "--calib-velo-to-cam",
        default=None,
        help="calib_velo_to_cam.txt : OXTS 3D et validation tracklets"
    )

    parser.add_argument(
        "--calib-imu-to-velo",
        default=None,
        help="calib_imu_to_velo.txt : OXTS 3D complet"
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        help="Nombre de frames ; 0 = toute la séquence"
    )

    parser.add_argument(
        "--foundation-disparity-dir",
        default=None,
        help=(
            "Dossier contenant les disparités FoundationStereo .npy. "
            "Lorsqu'il est fourni, FoundationStereo remplace StereoSGBM/WLS."
        )
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dossier de sortie"
    )

    parser.add_argument(
        "--world-kalman",
        action="store_true",
        help="Estime la vitesse 3D dans un repère fixe avec les poses OXTS",
    )
    args = parser.parse_args()
    if args.world_kalman and not (args.calib_imu_to_velo and args.calib_velo_to_cam):
        parser.error("--world-kalman exige --calib-imu-to-velo et --calib-velo-to-cam.")

    foundation_disparity_dir = None
    if args.foundation_disparity_dir:
        foundation_disparity_dir = Path(args.foundation_disparity_dir)
        if not foundation_disparity_dir.is_dir():
            raise FileNotFoundError(
                f"Dossier FoundationStereo introuvable : {foundation_disparity_dir}"
            )
        print(f"Disparité FoundationStereo activée : {foundation_disparity_dir}")

    pairs = find_stereo_pairs(
        args.left_dir,
        args.right_dir
    )

    if not pairs:
        raise FileNotFoundError(
            "Aucune paire stéréo n'a été trouvée."
        )

    if args.max_frames > 0:
        pairs = pairs[:args.max_frames]

    camera_poses = None
    if args.world_kalman:
        camera_poses = KittiCameraPoses(
            Path(args.oxts_dir) / f"{pairs[0][0].stem}.txt",
            args.calib_imu_to_velo,
            args.calib_velo_to_cam,
            args.calib,
        )
        print("Vitesse 3D : Kalman dans le repère fixe OXTS.")

    cfg = load_config()

    project_root = Path(__file__).parents[1]

    tracking_cfg = cfg["tracking"]
    tracker_type = tracking_cfg.get("type", "botsort").lower()
    tracker_config = None
    # Seuls les trackers internes Ultralytics utilisent un YAML local.
    if tracker_type in {"botsort", "bytetrack"}:
        tracker_config = project_root / tracking_cfg["config"]
        if not tracker_config.exists():
            raise FileNotFoundError(
                f"Configuration tracker introuvable : {tracker_config}"
            )

    calibration = StereoCalibration.from_kitti(
        args.calib
    )

    # calib_velo_to_cam sert aussi à convertir la vitesse OXTS vers le repère
    # caméra ; il peut donc être fourni sans tracklets. En revanche, les
    # tracklets nécessitent cette calibration pour leur projection 3D.
    if args.tracklets and not args.calib_velo_to_cam:
        raise ValueError(
            "--tracklets nécessite --calib-velo-to-cam pour projeter "
            "les annotations 3D KITTI."
        )

    validator = None
    if args.tracklets:
        validator = TrackletValidator(
            tracklets_path=args.tracklets,
            calib_velo_to_cam=args.calib_velo_to_cam,
            calib_cam_to_cam=args.calib,
        )
        print("Validation KITTI tracklets activée.")

    detector = YOLODetector(
        model_path=cfg["detection"]["model"],
        confidence=cfg["detection"]["confidence"],
        allowed_classes=set(
            cfg["detection"]["allowed_classes"]
        ),
        tracker_config=str(tracker_config) if tracker_config else None,
        tracker_type=tracker_type,
        reid_device=tracking_cfg.get("reid_device", "cpu"),
        reid_weights=tracking_cfg.get("reid_weights"),
    )

    # Le Kalman adaptatif lit directement tous ses paramètres dans YAML.
    # Il n'existe plus de position_noise_m fixe : R dépend de la profondeur.
    speed_estimator = SpeedEstimator(**cfg["speed"])

    pipeline = StereoSpeedPipeline(
        cfg,
        detector,
        speed_estimator,
        calibration
    )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif foundation_disparity_dir:
        output_dir = Path("outputs/sequence_foundation")
    else:
        output_dir = Path("outputs/sequence_botsort")
    output_dir.mkdir(parents=True, exist_ok=True)

    measurements = []

    for frame_index, (left_path, right_path) in enumerate(pairs):
        left_image = cv2.imread(str(left_path))
        right_image = cv2.imread(str(right_path))

        if left_image is None or right_image is None:
            print(f"Image ignorée : {left_path.name}")
            continue

        oxts_path = Path(args.oxts_dir) / (
            f"{left_path.stem}.txt"
        )

        ego_velocity_camera_mps = read_ego_velocity_camera(
            oxts_path,
            calib_imu_to_velo=args.calib_imu_to_velo,
            calib_velo_to_cam=args.calib_velo_to_cam,
            calib_cam_to_cam=args.calib,
        )
        ego_forward_mps = float(ego_velocity_camera_mps[2])
        camera_pose = camera_poses.camera_to_world(oxts_path) if camera_poses is not None else None

        foundation_disparity = None
        if foundation_disparity_dir:
            foundation_path = foundation_disparity_dir / f"{left_path.stem}.npy"
            if not foundation_path.exists():
                raise FileNotFoundError(
                    f"Disparité FoundationStereo absente pour {left_path.name} : {foundation_path}"
                )
            foundation_disparity = load_disparity_map(
                foundation_path,
                expected_shape=left_image.shape[:2],
            )

        # KITTI Raw : environ 10 images/seconde.
        timestamp_s = frame_index / 10.0

        result = pipeline.process(
            left_image,
            right_image,
            timestamp_s,
            ego_velocity_camera_mps,
            disparity_map=foundation_disparity,
            camera_pose=camera_pose,
        )

        if validator is not None:
            validator.evaluate(frame_index, result["objects"])

        output_path = output_dir / left_path.name

        cv2.imwrite(
            str(output_path),
            result["annotated"]
        )

        print(
            f"Frame : {left_path.name} | "
            f"vitesse voiture : "
            f"{ego_forward_mps * 3.6:.2f} km/h"
        )

        for obj in result["objects"]:
            depth = obj["depth_m"]
            relative_speed = obj["relative_speed_mps"]
            world_speed = obj["world_speed_mps"]
            world_velocity = obj["world_velocity_3d_mps"]
            relative_velocity = obj["relative_velocity_3d_mps"]
            speed_magnitude = obj["speed_magnitude_mps"]
            validation = obj.get("validation") or {}

            measurements.append({
                "frame": left_path.stem,
                "depth_source": "FoundationStereo" if foundation_disparity_dir else "StereoSGBM-WLS",
                "time_s": round(timestamp_s, 2),
                "track_id": obj["track_id"],
                "object_class": obj["label"],
                "distance_m": number_or_empty(depth),
                "depth_spread_m": number_or_empty(obj["depth_spread_m"], digits=3),
                "depth_valid_ratio": number_or_empty(obj["depth_valid_ratio"], digits=3),
                "depth_reliability": obj["depth_reliability"],
                "position_sigma_x_m": number_or_empty(
                    obj["position_measurement_std_m"][0], digits=3
                ),
                "position_sigma_y_m": number_or_empty(
                    obj["position_measurement_std_m"][1], digits=3
                ),
                "position_sigma_z_m": number_or_empty(
                    obj["position_measurement_std_m"][2], digits=3
                ),
                "relative_speed_km_h": number_or_empty(
                    relative_speed * 3.6
                ),
                "world_speed_km_h": number_or_empty(
                    world_speed * 3.6
                ),
                "speed_3d_km_h": number_or_empty(speed_magnitude * 3.6),
                "speed_3d_raw_km_h": number_or_empty(obj["raw_world_speed_mps"] * 3.6),
                "position_x_m": number_or_empty(obj["position_3d_m"][0]),
                "position_y_m": number_or_empty(obj["position_3d_m"][1]),
                "position_z_m": number_or_empty(obj["position_3d_m"][2]),
                "relative_vx_mps": number_or_empty(relative_velocity[0]),
                "relative_vy_mps": number_or_empty(relative_velocity[1]),
                "relative_vz_mps": number_or_empty(relative_velocity[2]),
                "world_vx_mps": number_or_empty(world_velocity[0]),
                "world_vy_mps": number_or_empty(world_velocity[1]),
                "world_vz_mps": number_or_empty(world_velocity[2]),
                "ego_speed_km_h": number_or_empty(
                    ego_forward_mps * 3.6
                ),
                "speed_reliability": obj[
                    "speed_reliability"
                ],
                "speed_status": obj["speed_status"],
                "speed_reason": obj["speed_reason"],
                "motion_state": obj["motion_state"],
                "velocity_noise_margin_kmh": number_or_empty(
                    obj["velocity_noise_margin_kmh"], digits=2
                ),
                "normalized_innovation": number_or_empty(
                    obj["normalized_innovation"], digits=3
                ),
                "speed_r2": number_or_empty(
                    obj["speed_r2"],
                    digits=3
                ),
                "depth_residual_m": number_or_empty(
                    obj["depth_residual_m"],
                    digits=3
                ),
                "analysis_eligible": obj["analysis_eligible"],
                "analysis_reason": obj["analysis_reason"],
                "ground_truth_id": validation.get("ground_truth_id", ""),
                "ground_truth_class": validation.get("ground_truth_label", ""),
                "ground_truth_depth_m": number_or_empty(
                    validation.get("depth_m", float("nan"))
                ),
                "depth_abs_error_m": number_or_empty(
                    validation.get("depth_abs_error_m", float("nan"))
                ),
                "relative_speed_abs_error_km_h": number_or_empty(
                    validation.get("relative_speed_abs_error_kmh", float("nan"))
                ),
            })

            if not obj["analysis_eligible"]:
                print(
                    f"  Track ID {obj['track_id']} | {obj['label']} | "
                    f"ignoré pour analyse : {obj['analysis_reason']}"
                )
                continue

            if obj["speed_status"] == "accepted":
                print(
                    f"  Track ID {obj['track_id']} | "
                    f"{obj['label']} | "
                    f"distance : {depth:.2f} m | "
                    f"|V| : {speed_magnitude * 3.6:.2f} km/h | "
                    f"Vz : {world_velocity[2]:.2f} m/s | "
                    f"fiabilité : "
                    f"{obj['speed_reliability']}"
                )
            else:
                print(
                    f"  Track ID {obj['track_id']} | "
                    f"{obj['label']} | "
                    f"vitesse rejetée / attente | "
                    f"raison : {obj['speed_reason']}"
                )

    csv_path = output_dir / "measurements.csv"

    columns = [
        "frame",
        "depth_source",
        "time_s",
        "track_id",
        "object_class",
        "distance_m",
        "depth_spread_m",
        "depth_valid_ratio",
        "depth_reliability",
        "position_sigma_x_m",
        "position_sigma_y_m",
        "position_sigma_z_m",
        "relative_speed_km_h",
        "world_speed_km_h",
        "speed_3d_km_h",
        "speed_3d_raw_km_h",
        "position_x_m",
        "position_y_m",
        "position_z_m",
        "relative_vx_mps",
        "relative_vy_mps",
        "relative_vz_mps",
        "world_vx_mps",
        "world_vy_mps",
        "world_vz_mps",
        "ego_speed_km_h",
        "speed_reliability",
        "speed_status",
        "speed_reason",
        "motion_state",
        "velocity_noise_margin_kmh",
        "normalized_innovation",
        "speed_r2",
        "depth_residual_m",
        "analysis_eligible",
        "analysis_reason",
        "ground_truth_id",
        "ground_truth_class",
        "ground_truth_depth_m",
        "depth_abs_error_m",
        "relative_speed_abs_error_km_h",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            delimiter=";"
        )

        writer.writeheader()
        writer.writerows(measurements)

    print(f"CSV créé : {csv_path}")


if __name__ == "__main__":
    main()
