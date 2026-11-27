"""Lecture OXTS KITTI et conversion de la vitesse véhicule vers la caméra."""

from pathlib import Path

import numpy as np


def _read_rigid_calibration(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, data = line.split(":", 1)
        if key in {"R", "T"}:
            values[key] = np.fromstring(data.strip(), sep=" ")
    if "R" not in values or "T" not in values:
        raise ValueError(f"Calibration rigide invalide : {path}")
    return values["R"].reshape(3, 3), values["T"].reshape(3)


def _read_rectification_rotation(cam_to_cam_path):
    for line in Path(cam_to_cam_path).read_text(encoding="utf-8").splitlines():
        if line.startswith("R_rect_00:"):
            return np.fromstring(line.split(":", 1)[1].strip(), sep=" ").reshape(3, 3)
    raise ValueError(f"R_rect_00 introuvable : {cam_to_cam_path}")


def read_ego_velocity_camera(
    oxts_file,
    calib_imu_to_velo=None,
    calib_velo_to_cam=None,
    calib_cam_to_cam=None,
):
    """Retourne la vitesse voiture dans le repère caméra [droite, bas, avant]."""
    values = Path(oxts_file).read_text(encoding="utf-8").split()
    if len(values) < 11:
        raise ValueError(f"Fichier OXTS invalide : {oxts_file}")

    velocity_imu = np.array([float(values[8]), float(values[9]), float(values[10])])
    if not all([calib_imu_to_velo, calib_velo_to_cam, calib_cam_to_cam]):
        return np.array([0.0, 0.0, velocity_imu[0]], dtype=float)

    r_imu_to_velo, _ = _read_rigid_calibration(calib_imu_to_velo)
    r_velo_to_cam, _ = _read_rigid_calibration(calib_velo_to_cam)
    r_rect = _read_rectification_rotation(calib_cam_to_cam)
    return r_rect @ r_velo_to_cam @ r_imu_to_velo @ velocity_imu


def read_ego_forward_speed(oxts_file):
    """Compatibilité avec les anciens scripts : retourne vf en m/s."""
    return float(read_ego_velocity_camera(oxts_file)[2])


class KittiCameraPoses:
    """Poses de la caméra couleur gauche rectifiée dans un repère local fixe."""

    def __init__(self, first_oxts_file, calib_imu_to_velo, calib_velo_to_cam, calib_cam_to_cam):
        first = self._read_pose_values(first_oxts_file)
        self.scale = np.cos(np.deg2rad(first[0]))
        self.origin = self._translation(first)

        imu_rotation, imu_translation = _read_rigid_calibration(calib_imu_to_velo)
        velo_rotation, velo_translation = _read_rigid_calibration(calib_velo_to_cam)
        rect = _read_rectification_rotation(calib_cam_to_cam)
        projection = next(
            np.fromstring(line.split(":", 1)[1], sep=" ").reshape(3, 4)
            for line in Path(calib_cam_to_cam).read_text(encoding="utf-8").splitlines()
            if line.startswith("P_rect_02:")
        )

        # P_rect_02 = K_02 [I | t_02] dans le repère rectifié de la caméra 0.
        rect_to_left = np.linalg.solve(projection[:, :3], projection[:, 3])
        imu_to_left = rect @ velo_rotation @ imu_rotation
        imu_origin_left = rect @ (velo_rotation @ imu_translation + velo_translation)
        imu_origin_left += rect_to_left
        self.left_to_imu = imu_to_left.T
        self.left_origin_imu = -self.left_to_imu @ imu_origin_left

    @staticmethod
    def _read_pose_values(oxts_file):
        fields = Path(oxts_file).read_text(encoding="utf-8").split()
        values = np.asarray(fields[:6], dtype=float)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Pose OXTS invalide : {oxts_file}")
        return values

    def _translation(self, values):
        latitude, longitude, altitude = values[:3]
        radius_m = 6_378_137.0
        return np.array([
            self.scale * radius_m * np.deg2rad(longitude),
            self.scale * radius_m * np.log(np.tan(np.pi / 4 + np.deg2rad(latitude) / 2)),
            altitude,
        ])

    def camera_to_world(self, oxts_file):
        values = self._read_pose_values(oxts_file)
        roll, pitch, yaw = values[3:]
        cr, cp, cy = np.cos([roll, pitch, yaw])
        sr, sp, sy = np.sin([roll, pitch, yaw])
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        imu_to_world = rz @ ry @ rx
        rotation = imu_to_world @ self.left_to_imu
        translation = self._translation(values) - self.origin + imu_to_world @ self.left_origin_imu
        return rotation, translation
