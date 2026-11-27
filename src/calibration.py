"""Lecture de calibration KITTI et rectification stereo."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class StereoCalibration:
    focal_length_px: float
    baseline_m: float
    cx: float = 0.0
    cy: float = 0.0

    @classmethod
    def from_kitti(cls, calib_file):
        values = {}

        for line in Path(calib_file).read_text().splitlines():
            if ":" not in line:
                continue

            key, data = line.split(":", 1)

            # KITTI Raw : P_rect_02 / P_rect_03
            # KITTI Object : P2 / P3
            if key in ("P2", "P3", "P_rect_02", "P_rect_03"):
                values[key] = np.fromstring(data.strip(), sep=" ")

        p_l_data = values.get("P2", values.get("P_rect_02"))
        p_r_data = values.get("P3", values.get("P_rect_03"))

        if p_l_data is None or p_r_data is None:
            raise ValueError(
                "Matrices de projection introuvables dans le fichier calibration."
            )

        p_l = p_l_data.reshape(3, 4)
        p_r = p_r_data.reshape(3, 4)

        focal_length = p_l[0, 0]

        baseline = abs(
            p_r[0, 3] / p_r[0, 0]
            - p_l[0, 3] / p_l[0, 0]
        )

        return cls(
            focal_length_px=float(focal_length),
            baseline_m=float(baseline),
            cx=float(p_l[0, 2]),
            cy=float(p_l[1, 2]),
        )


def rectify_pair(left, right, maps=None):
    """Les images KITTI synced+rectified sont déjà rectifiées."""
    if maps is None:
        return left, right

    l1, l2, r1, r2 = maps

    left_rectified = cv2.remap(left, l1, l2, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right, r1, r2, cv2.INTER_LINEAR)

    return left_rectified, right_rectified