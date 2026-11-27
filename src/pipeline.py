from src.depth import box_to_3d, disparity_to_depth, robust_box_depth_details
from src.analysis_filter import RoadMotionFilter
from src.label_stabilizer import TrackLabelStabilizer
from src.stereo import compute_disparity
from src.visualization import draw_objects


class StereoSpeedPipeline:
    """
    Pipeline complet du projet :

    images stéréo
        -> carte de disparité précalculée ou StereoSGBM + WLS
        -> profondeur
        -> YOLO + ByteTrack
        -> stabilisation des labels
        -> vitesse
        -> image annotée
    """

    def __init__(self, cfg, detector, speed_estimator, calibration):
        self.cfg = cfg
        self.detector = detector
        self.speed_estimator = speed_estimator
        self.calibration = calibration

        semantic_cfg = cfg.get("semantic_stabilization", {})

        self.label_stabilizer = TrackLabelStabilizer(
            history_size=semantic_cfg.get("history_size", 12),
            min_observations=semantic_cfg.get("min_observations", 3),
            min_majority_ratio=semantic_cfg.get(
                "min_majority_ratio",
                0.60
            ),
            max_inactive_frames=semantic_cfg.get(
                "max_inactive_frames",
                40
            ),
        )

        self.frame_index = 0
        self.road_motion_filter = RoadMotionFilter(
            cfg.get("analysis_filter", {})
        )

    def process(
        self,
        left_image,
        right_image,
        timestamp_s,
        ego_velocity_camera_mps=None,
        disparity_map=None,
        camera_pose=None,
    ):
        """
        Traite une paire d'images stéréo KITTI.

        Paramètres :
        - left_image : image couleur de la caméra gauche ;
        - right_image : image couleur de la caméra droite ;
        - timestamp_s : temps de la frame en secondes ;
        - ego_velocity_camera_mps : vitesse OXTS transformée dans le repère
          caméra rectifié, sous la forme [droite, bas, avant].
        - disparity_map : carte de disparité précalculée.
          Si elle est fournie, StereoSGBM/WLS n'est pas exécuté.

        Retour :
        - image annotée ;
        - carte de disparité ;
        - carte de profondeur ;
        - objets détectés avec distance et vitesse.
        """

        if left_image is None or right_image is None:
            raise ValueError(
                "Image gauche ou image droite invalide."
            )

        if ego_velocity_camera_mps is None:
            ego_velocity_camera_mps = [0.0, 0.0, 0.0]

        # 1. carte de disparité précalculée est prioritaire si sa carte pré-calculée est fournie.
        # Sinon, le projet garde StereoSGBM/WLS comme solution de secours.
        if disparity_map is None:
            disparity = compute_disparity(
                left_image,
                right_image,
                self.cfg,
            )
        else:
            disparity = disparity_map

        # 2. Conversion disparité -> profondeur métrique.
        depth_map = disparity_to_depth(
            disparity=disparity,
            focal_length_px=self.calibration.focal_length_px,
            baseline_m=self.calibration.baseline_m,
        )

        # 3. YOLO détecte les objets ; ByteTrack conserve leur ID.
        detections = self.detector.track(left_image)

        objects = []

        for detection in detections:
            track_id = detection.get("track_id")
            raw_label = detection["label"]
            confidence = detection["score"]
            box = detection["box"]

            # 4. Vote sur plusieurs frames pour stabiliser la classe.
            stable_label = self.label_stabilizer.update(
                track_id=track_id,
                raw_label=raw_label,
                confidence=confidence,
                frame_index=self.frame_index,
            )

            # 5. Profondeur robuste dans la zone centrale de la bounding box.
            depth_info = robust_box_depth_details(
                depth=depth_map,
                box=box,
                mask=detection.get("mask"),
            )
            depth_m = depth_info["depth_m"]

            # 6. Le centre 2D de la boîte devient un point 3D (X, Y, Z).
            # La vitesse utilise principalement Z, l'axe avant/arrière.
            position_3d = box_to_3d(
                box=box,
                z_m=depth_m,
                calib=self.calibration,
            )

            # Incertitude stéréo : sigma_Z = Z²/(fB) × sigma_d. La dispersion
            # réellement observée dans la boîte est aussi retenue si elle est
            # plus élevée. Le Kalman ne doit pas interpréter ce bruit comme un
            # déplacement réel de l'objet.
            speed_cfg = self.cfg.get("speed", {})
            if depth_m == depth_m:
                geometry_sigma_z = (
                    depth_m ** 2
                    / (self.calibration.focal_length_px * self.calibration.baseline_m)
                    * float(speed_cfg.get("disparity_noise_px", 1.0))
                )
                depth_sigma = max(
                    float(depth_info["spread_m"])
                    if depth_info["spread_m"] == depth_info["spread_m"] else 0.0,
                    geometry_sigma_z,
                    float(speed_cfg.get("min_depth_sigma_m", 0.15)),
                )
                x1, y1, x2, y2 = box
                u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                pixel_sigma = float(speed_cfg.get("box_center_noise_px", 2.0))
                sigma_x = (
                    (depth_m / self.calibration.focal_length_px * pixel_sigma) ** 2
                    + ((u - self.calibration.cx) / self.calibration.focal_length_px * depth_sigma) ** 2
                ) ** 0.5
                sigma_y = (
                    (depth_m / self.calibration.focal_length_px * pixel_sigma) ** 2
                    + ((v - self.calibration.cy) / self.calibration.focal_length_px * depth_sigma) ** 2
                ) ** 0.5
                measurement_std_m = [sigma_x, sigma_y, depth_sigma]
            else:
                measurement_std_m = [float("nan")] * 3

            # 7. Vitesse relative et vitesse monde compensée par OXTS.
            speed_info = self.speed_estimator.update(
                track_id=track_id,
                position_3d=position_3d,
                timestamp_s=timestamp_s,
                label=stable_label,
                ego_velocity_camera_mps=ego_velocity_camera_mps,
                measurement_std_m=measurement_std_m,
                camera_pose=camera_pose,
            )

            analysis_eligible, analysis_reason = self.road_motion_filter.evaluate(
                box=box,
                image_shape=left_image.shape,
                speed_info=speed_info,
            )

            objects.append({
                "track_id": track_id,
                "label": stable_label,
                "raw_label": raw_label,
                "score": confidence,
                "box": box,
                "depth_m": depth_m,
                "depth_spread_m": depth_info["spread_m"],
                "depth_valid_ratio": depth_info["valid_ratio"],
                "depth_reliability": depth_info["reliability"],
                "position_measurement_std_m": measurement_std_m,
                "relative_speed_mps": speed_info["relative_mps"],
                "world_speed_mps": speed_info["world_mps"],
                "position_3d_m": position_3d,
                "relative_velocity_3d_mps": speed_info["relative_velocity_mps"],
                "world_velocity_3d_mps": speed_info["world_velocity_mps"],
                "speed_magnitude_mps": speed_info["speed_magnitude_mps"],
                "raw_world_speed_mps": speed_info["raw_world_speed_mps"],
                "speed_mps": speed_info["world_mps"],
                "speed_reliability": speed_info["reliability"],
                "speed_status": speed_info["status"],
                "speed_reason": speed_info["reason"],
                "motion_state": speed_info["motion_state"],
                "velocity_noise_margin_kmh": speed_info["velocity_noise_margin_kmh"],
                "normalized_innovation": speed_info["normalized_innovation"],
                "speed_r2": speed_info["r2"],
                "depth_residual_m": speed_info["residual_m"],
                "analysis_eligible": analysis_eligible,
                "analysis_reason": analysis_reason,
            })

        # Supprime la mémoire des IDs absents depuis longtemps.
        self.label_stabilizer.cleanup(self.frame_index)

        # 7. Création de l'image finale avec boîtes, ID, distance et vitesse.
        annotated = draw_objects(
            left_image,
            objects,
        )

        self.frame_index += 1

        return {
            "annotated": annotated,
            "disparity": disparity,
            "depth_map": depth_map,
            "objects": objects,
        }
