import cv2
import numpy as np


def draw_objects(image, objects):
    """Affiche seulement les vitesses retenues pour l'analyse routière."""
    out = image.copy()
    for obj in objects:
        x1, y1, x2, y2 = map(int, obj["box"])
        depth = obj.get("depth_m", float("nan"))
        speed = obj.get("speed_magnitude_mps", float("nan"))
        show_speed = (
            obj.get("speed_status") == "accepted"
            and np.isfinite(speed)
            and np.isfinite(depth)
            and 0 < depth <= 30.0
            and obj.get("depth_reliability") in {"high", "medium"}
            and x2 - x1 >= 20
            and y2 - y1 >= 20
        )
        if not obj.get("analysis_eligible", True):
            text = (
                f'#{obj.get("track_id", "?")} {obj["label"]} | '
                f'{depth:.1f}m | hors analyse'
            )
            color = (0, 165, 255)
        elif show_speed:
            text = f'#{obj.get("track_id", "?")} {obj["label"]} | {depth:.1f}m | |V| {speed * 3.6:.1f} km/h'
            color = (0, 255, 70)
        else:
            text = f'#{obj.get("track_id", "?")} {obj["label"]} | {depth:.1f}m'
            color = (0, 255, 70)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    return out
