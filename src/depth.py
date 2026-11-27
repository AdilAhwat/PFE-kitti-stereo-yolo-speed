"""Conversion stéréo et estimation robuste de profondeur par objet."""

import numpy as np


def disparity_to_depth(disparity, focal_length_px, baseline_m):
    """Transforme la disparité en profondeur métrique avec Z = f × B / d."""
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 0.1)
    depth[valid] = focal_length_px * baseline_m / disparity[valid]
    return depth


def robust_box_depth_details(
    depth,
    box,
    inner_width_ratio=0.55,
    inner_height_ratio=0.60,
    trim=0.15,
    min_valid_pixels=20,
    min_depth_m=0.2,
    max_depth_m=150.0,
    mask=None,
    
):
    """Estime Z dans une boîte avec médiane robuste et score de qualité.

    Les contours YOLO mélangent souvent objet, route et arrière-plan. La zone
    centrale est donc utilisée, les valeurs extrêmes sont retirées et la
    dispersion MAD indique si la mesure est cohérente.
    """
    x1, y1, x2, y2 = map(int, box)
    image_height, image_width = depth.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image_width, x2), min(image_height, y2)
    empty = {"depth_m": float("nan"), "spread_m": float("nan"), "valid_pixels": 0, "valid_ratio": 0.0, "reliability": "low"}
    if x2 <= x1 or y2 <= y1:
        return empty

    width, height = x2 - x1, y2 - y1
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_width = width * inner_width_ratio / 2.0
    half_height = height * inner_height_ratio / 2.0
    ix1, ix2 = max(0, int(center_x - half_width)), min(image_width, int(center_x + half_width))
    iy1, iy2 = max(0, int(center_y - half_height)), min(image_height, int(center_y + half_height))
    if mask is None:
        roi = depth[iy1:iy2, ix1:ix2]
    else:
        if mask.shape != depth.shape:
            raise ValueError(
                f"Masque de taille {mask.shape}, profondeur de taille {depth.shape}."
            )
        roi = depth[mask]
    values = roi[np.isfinite(roi) & (roi >= min_depth_m) & (roi <= max_depth_m)]
    valid_ratio = len(values) / max(roi.size, 1)
    if len(values) < min_valid_pixels:
        return {**empty, "valid_pixels": int(len(values)), "valid_ratio": valid_ratio}

    low, high = np.quantile(values, [trim, 1.0 - trim])
    inliers = values[(values >= low) & (values <= high)]
    if len(inliers) < min_valid_pixels:
        return {**empty, "valid_pixels": int(len(inliers)), "valid_ratio": valid_ratio}

    estimate = float(np.median(inliers))
    spread = float(1.4826 * np.median(np.abs(inliers - estimate)))
    if valid_ratio >= 0.50 and spread <= max(0.35, estimate * 0.06):
        reliability = "high"
    elif valid_ratio >= 0.25 and spread <= max(0.70, estimate * 0.12):
        reliability = "medium"
    else:
        reliability = "low"
    return {"depth_m": estimate, "spread_m": spread, "valid_pixels": int(len(inliers)), "valid_ratio": float(valid_ratio), "reliability": reliability}


def robust_box_depth(depth, box, **kwargs):
    """Compatibilité : retourne uniquement la profondeur robuste Z."""
    return robust_box_depth_details(depth, box, **kwargs)["depth_m"]


def box_to_3d(box, z_m, calib):
    """Projette le centre 2D de la boîte en coordonnées caméra (X, Y, Z)."""
    if not np.isfinite(z_m):
        return np.full(3, np.nan, dtype=float)
    x1, y1, x2, y2 = box
    u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return np.array([
        (u - calib.cx) * z_m / calib.focal_length_px,
        (v - calib.cy) * z_m / calib.focal_length_px,
        z_m,
    ], dtype=float)
