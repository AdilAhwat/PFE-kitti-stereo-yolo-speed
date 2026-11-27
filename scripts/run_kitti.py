"""Traite une paire KITTI unique avec l'architecture 3D actuelle."""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.calibration import StereoCalibration
from src.config import load_config
from src.detector import YOLODetector
from src.pipeline import StereoSpeedPipeline
from src.speed import SpeedEstimator


def main():
    parser = argparse.ArgumentParser(description="Traite une paire KITTI image_02/image_03.")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--output", default="outputs/kitti_result.jpg")
    parser.add_argument("--config", default=None)
    parser.add_argument("--time", type=float, default=0.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(__file__).parents[1]
    detector = YOLODetector(
        model_path=cfg["detection"]["model"],
        confidence=cfg["detection"]["confidence"],
        allowed_classes=set(cfg["detection"]["allowed_classes"]),
        tracker_config=str(project_root / cfg["tracking"]["config"]) if cfg["tracking"].get("type", "botsort") in {"botsort", "bytetrack"} else None,
        tracker_type=cfg["tracking"].get("type", "botsort"),
        reid_device=cfg["tracking"].get("reid_device", "cpu"),
        reid_weights=cfg["tracking"].get("reid_weights"),
    )
    estimator = SpeedEstimator(
        min_dt_s=cfg["speed"]["min_dt_s"],
        min_samples=cfg["speed"]["min_samples"],
        position_noise_m=cfg["speed"]["position_noise_m"],
        process_noise=cfg["speed"]["process_noise"],
        max_innovation_m=cfg["speed"]["max_innovation_m"],
        max_speed_kmh=cfg["speed"]["max_speed_kmh"],
    )
    pipeline = StereoSpeedPipeline(
        cfg, detector, estimator, StereoCalibration.from_kitti(args.calib)
    )
    result = pipeline.process(
        cv2.imread(args.left),
        cv2.imread(args.right),
        args.time,
        [0.0, 0.0, 0.0],
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, result["annotated"])
    print(f"Résultat : {args.output}")
    print("Une seule image donne une distance, pas encore une vitesse fiable.")


if __name__ == "__main__":
    main()
