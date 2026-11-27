"""Filtre de Kalman 3D adaptatif et estimation de vitesse.

L'état d'une piste est ``[X, Y, Z, Vx, Vy, Vz]`` dans son repère de suivi. La covariance de mesure n'est pas constante : l'incertitude de Z
augmente avec la distance, comme l'impose la géométrie stéréo.
"""

from collections import deque

import numpy as np


class KalmanTrack3D:
    """Filtre à vitesse constante avec mesures 3D à bruit adaptatif."""

    def __init__(self, position, process_noise, temporal_window):
        self.state = np.zeros(6, dtype=float)
        self.state[:3] = position
        self.covariance = np.eye(6, dtype=float) * 10.0
        self.process_noise = float(process_noise)
        self.last_timestamp = None
        self.samples = 1
        self.position_history = deque(
            [np.asarray(position, dtype=float)],
            maxlen=max(1, int(temporal_window)),
        )
        self.static_frames = 0

    def temporal_position(self, position):
        """Médiane des dernières mesures 3D pour réduire le bruit stéréo."""
        self.position_history.append(np.asarray(position, dtype=float))
        return np.median(np.stack(self.position_history), axis=0)

    def predict(self, dt):
        transition = np.eye(6, dtype=float)
        transition[0, 3] = dt
        transition[1, 4] = dt
        transition[2, 5] = dt

        q = self.process_noise * max(dt, 0.01)
        process = np.diag([q * dt * dt, q * dt * dt, q * dt * dt, q, q, q])
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process

    def update(self, position, measurement_std_m, measurement_covariance_m2=None):
        """Mise à jour avec la covariance de mesure exprimée dans le repère du filtre."""
        measurement = np.asarray(position, dtype=float)
        std = np.maximum(np.asarray(measurement_std_m, dtype=float), 0.03)
        h = np.zeros((3, 6), dtype=float)
        h[0, 0] = h[1, 1] = h[2, 2] = 1.0
        r = (
            np.diag(std ** 2)
            if measurement_covariance_m2 is None
            else measurement_covariance_m2
        )
        innovation = measurement - h @ self.state
        s = h @ self.covariance @ h.T + r
        gain = np.linalg.solve(s.T, (self.covariance @ h.T).T).T
        self.state = self.state + gain @ innovation
        identity = np.eye(6)
        residual = identity - gain @ h
        self.covariance = residual @ self.covariance @ residual.T + gain @ r @ gain.T
        self.samples += 1
        normalized = float(np.sqrt(innovation.T @ np.linalg.solve(s, innovation)))
        return float(np.linalg.norm(innovation)), normalized


class SpeedEstimator:
    """Vitesse relative, compensation OXTS et Zero-Velocity Update (ZUPT)."""

    def __init__(
        self,
        min_dt_s=0.05,
        min_samples=8,
        process_noise=2.0,
        max_innovation_m=2.0,
        max_normalized_innovation=4.5,
        temporal_window=3,
        zero_velocity_base_kmh=2.5,
        zero_velocity_depth_gain_kmh_per_m2=0.025,
        zero_velocity_max_kmh=12.0,
        zero_velocity_min_frames=4,
        max_speed_kmh=None,
        **_ignored,
    ):
        self.min_dt_s = float(min_dt_s)
        self.min_samples = int(min_samples)
        self.process_noise = float(process_noise)
        self.max_innovation_m = float(max_innovation_m)
        self.max_normalized_innovation = float(max_normalized_innovation)
        self.temporal_window = int(temporal_window)
        self.zero_velocity_base_kmh = float(zero_velocity_base_kmh)
        self.zero_velocity_depth_gain = float(zero_velocity_depth_gain_kmh_per_m2)
        self.zero_velocity_max_kmh = float(zero_velocity_max_kmh)
        self.zero_velocity_min_frames = int(zero_velocity_min_frames)
        self.max_speed_kmh = max_speed_kmh or {}
        self.tracks = {}
        self.world_tracks = {}

    @staticmethod
    def _empty(reason="Kalman 3D en phase de stabilisation"):
        vector = np.full(3, np.nan, dtype=float)
        return {
            "relative_velocity_mps": vector,
            "world_velocity_mps": vector,
            "relative_mps": float("nan"),
            "world_mps": float("nan"),
            "speed_magnitude_mps": float("nan"),
            "raw_world_speed_mps": float("nan"),
            "reliability": "waiting",
            "status": "waiting",
            "motion_state": "unknown",
            "reason": reason,
            "r2": float("nan"),
            "residual_m": float("nan"),
            "normalized_innovation": float("nan"),
            "velocity_noise_margin_kmh": float("nan"),
        }

    def _noise_margin_kmh(self, depth_m):
        """Marge ZUPT croissante avec Z², plafonnée pour rester prudente."""
        margin = self.zero_velocity_base_kmh + self.zero_velocity_depth_gain * depth_m ** 2
        return float(min(margin, self.zero_velocity_max_kmh))

    def update(
        self,
        track_id,
        position_3d,
        timestamp_s,
        ego_velocity_camera_mps,
        label,
        measurement_std_m=None,
        camera_pose=None,
    ):
        position = np.asarray(position_3d, dtype=float)
        if track_id is None or not np.all(np.isfinite(position)) or position[2] <= 0:
            return self._empty("Position 3D stéréo indisponible")

        ego_velocity = np.asarray(ego_velocity_camera_mps, dtype=float)
        if not np.all(np.isfinite(ego_velocity)):
            return self._empty("Vitesse OXTS indisponible")
        if measurement_std_m is None:
            measurement_std_m = np.full(3, 0.45, dtype=float)

        track = self.tracks.get(track_id)
        if track is None:
            track = KalmanTrack3D(position, self.process_noise, self.temporal_window)
            track.last_timestamp = float(timestamp_s)
            self.tracks[track_id] = track
            if camera_pose is not None:
                rotation, translation = camera_pose
                world_track = KalmanTrack3D(
                    rotation @ position + translation, self.process_noise, self.temporal_window
                )
                world_track.last_timestamp = float(timestamp_s)
                self.world_tracks[track_id] = world_track
            return self._empty("Kalman 3D en phase de stabilisation")

        dt = float(timestamp_s) - track.last_timestamp
        if dt < self.min_dt_s:
            return self._empty("Intervalle temporel insuffisant")

        filtered_position = track.temporal_position(position)
        track.predict(dt)
        innovation_m, normalized_innovation = track.update(filtered_position, measurement_std_m)
        track.last_timestamp = float(timestamp_s)

        relative_velocity = track.state[3:].copy()
        if camera_pose is None:
            world_velocity = relative_velocity + ego_velocity
        else:
            rotation, translation = camera_pose
            world_position = rotation @ position + translation
            world_track = self.world_tracks.get(track_id)
            if world_track is None:
                world_track = KalmanTrack3D(world_position, self.process_noise, self.temporal_window)
                world_track.last_timestamp = float(timestamp_s)
                self.world_tracks[track_id] = world_track
                result = self._empty()
                result.update(
                    relative_velocity_mps=relative_velocity,
                    relative_mps=float(relative_velocity[2]),
                )
                return result
            world_dt = float(timestamp_s) - world_track.last_timestamp
            world_track.predict(world_dt)
            std = np.maximum(np.asarray(measurement_std_m, dtype=float), 0.03)
            world_covariance = (rotation * std**2) @ rotation.T
            innovation_m, normalized_innovation = world_track.update(
                world_track.temporal_position(world_position),
                measurement_std_m,
                measurement_covariance_m2=world_covariance,
            )
            world_track.last_timestamp = float(timestamp_s)
            track = world_track
            # Le CSV et le filtre routier utilisent les axes caméra.
            # Cette rotation de sortie conserve la norme de la vitesse.
            world_velocity = rotation.T @ world_track.state[3:]
        world_speed_mps = float(np.linalg.norm(world_velocity))
        world_speed_kmh = world_speed_mps * 3.6
        margin_kmh = self._noise_margin_kmh(float(filtered_position[2]))

        if world_speed_kmh <= margin_kmh:
            track.static_frames += 1
        else:
            track.static_frames = 0

        base = {
            "relative_velocity_mps": relative_velocity,
            "relative_mps": float(relative_velocity[2]),
            "raw_world_speed_mps": world_speed_mps,
            "residual_m": innovation_m,
            "normalized_innovation": normalized_innovation,
            "velocity_noise_margin_kmh": margin_kmh,
        }

        if innovation_m > self.max_innovation_m or normalized_innovation > self.max_normalized_innovation:
            result = self._empty("Innovation Kalman trop élevée")
            result.update({**base, "status": "rejected", "reliability": "rejected"})
            return result

        if track.samples < self.min_samples:
            result = self._empty("Kalman 3D en phase de stabilisation")
            result.update(base)
            return result

        if track.static_frames >= self.zero_velocity_min_frames:
            zeros = np.zeros(3, dtype=float)
            return {
                **base,
                "world_velocity_mps": zeros,
                "world_mps": 0.0,
                "speed_magnitude_mps": 0.0,
                "reliability": "high",
                "status": "static",
                "motion_state": "static",
                "reason": "ZUPT : objet immobile dans la marge d'incertitude",
                "r2": float("nan"),
            }

        if world_speed_kmh > float(self.max_speed_kmh.get(label, 180.0)):
            result = self._empty("Vitesse 3D supérieure à la limite de classe")
            result.update({**base, "status": "rejected", "reliability": "rejected"})
            return result

        reliability = "high" if normalized_innovation <= 2.0 else "medium"
        return {
            **base,
            "world_velocity_mps": world_velocity,
            "world_mps": float(world_velocity[2]),
            "speed_magnitude_mps": world_speed_mps,
            "reliability": reliability,
            "status": "accepted",
            "motion_state": "moving",
            "reason": "Vecteur vitesse 3D Kalman accepté",
            "r2": float("nan"),
        }
