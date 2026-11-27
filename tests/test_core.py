import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.depth import disparity_to_depth, robust_box_depth
from src.speed import SpeedEstimator


def test_depth():
    disparity = np.full((10, 10), 36.0, np.float32)
    depth = disparity_to_depth(disparity, 720.0, 0.54)
    assert abs(robust_box_depth(depth, (0, 0, 10, 10)) - 10.8) < 0.01


def test_kalman_3d_speed():
    estimator = SpeedEstimator(
        min_samples=4,
        position_noise_m=0.1,
        process_noise=0.1,
        max_innovation_m=5.0,
    )
    result = None
    # Objet immobile dans le monde ; la caméra avance à 10 m/s.
    for index in range(5):
        result = estimator.update(
            track_id=1,
            position_3d=np.array([0.0, 0.0, 20.0 - index]),
            timestamp_s=index * 0.1,
            ego_velocity_camera_mps=np.array([0.0, 0.0, 10.0]),
            label="car",
        )
    assert result["status"] == "accepted"
    assert abs(result["world_velocity_mps"][2]) < 2.0
