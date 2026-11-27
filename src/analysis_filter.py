"""Filtre prudent des objets retenus pour l'analyse de vitesse."""

import math


class RoadMotionFilter:
    """
    Ne retire pas les détections YOLO de l'image.
    Il indique seulement si une mesure doit participer à l'analyse CSV.

    La zone route est une ROI géométrique configurable. Ce n'est pas une
    segmentation sémantique : un objet peut donc être marqué à vérifier.
    """

    def __init__(self, config):
        self.enabled = bool(config.get("enabled", True))
        self.x_min = float(config.get("road_x_min_ratio", 0.10))
        self.x_max = float(config.get("road_x_max_ratio", 0.90))
        self.y_min = float(config.get("road_y_min_ratio", 0.42))
        self.y_max = float(config.get("road_y_max_ratio", 1.00))
        self.require_forward_motion = bool(
            config.get("require_forward_motion", False)
        )
        self.min_forward_speed_kmh = float(
            config.get("min_forward_speed_kmh", 3.0)
        )

    def evaluate(self, box, image_shape, speed_info):
        if not self.enabled:
            return True, "Filtre désactivé"

        height, width = image_shape[:2]
        x1, _, x2, y2 = [float(value) for value in box]
        center_x_ratio = ((x1 + x2) / 2.0) / max(width, 1)
        bottom_y_ratio = y2 / max(height, 1)

        if not (
            self.x_min <= center_x_ratio <= self.x_max
            and self.y_min <= bottom_y_ratio <= self.y_max
        ):
            return False, "Objet hors de la zone routière analysée"

        world_mps = speed_info.get("world_mps", float("nan"))
        status = speed_info.get("status", "waiting")

        # Les objets en attente restent visibles et seront réévalués dès que
        # leur historique contiendra assez de frames.
        if status == "static":
            return False, "Objet immobile (ZUPT)"

        if status != "accepted" or not math.isfinite(world_mps):
            return True, "En attente de vitesse fiable"

        world_kmh = world_mps * 3.6

        if self.require_forward_motion and world_kmh < self.min_forward_speed_kmh:
            return False, "Objet immobile ou en sens opposé"

        if not self.require_forward_motion and abs(world_kmh) < self.min_forward_speed_kmh:
            return False, "Objet quasi immobile"

        return True, "Objet routier mobile"
