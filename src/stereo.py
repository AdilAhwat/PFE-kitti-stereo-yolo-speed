import cv2
import numpy as np


def load_disparity_map(disparity_path, expected_shape):
    """Charge et valide une carte de disparité  au format .npy."""
    disparity = np.load(str(disparity_path)).astype(np.float32)

    if disparity.ndim != 2:
        raise ValueError(
            f"La carte de disparité doit être une matrice 2D, pas {disparity.shape}."
        )
    if tuple(disparity.shape) != tuple(expected_shape):
        raise ValueError(
            "Taille carte de disparité incompatible avec l'image gauche : "
            f"disparité={disparity.shape}, image={expected_shape}."
        )

    # La profondeur Z = f * B / d n'existe pas pour d <= 0.
    disparity[~np.isfinite(disparity) | (disparity <= 0.1)] = np.nan
    return disparity

def to_gray(image):
    """Transforme une image BGR en niveaux de gris."""

    if image.ndim == 2:
        return image

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


def create_stereo_matcher(stereo_config, min_disparity=None):
    """
    Crée le matcher StereoSGBM pour l'image gauche.

    StereoSGBM recherche, pour chaque pixel gauche,
    le pixel correspondant dans l'image droite.
    """

    block_size = int(stereo_config["block_size"])

    num_disparities = int(
        stereo_config["num_disparities"]
    )

    # OpenCV exige un multiple de 16.
    num_disparities = max(
        16,
        (num_disparities // 16) * 16
    )

    if min_disparity is None:
        min_disparity = int(stereo_config["min_disparity"])

    return cv2.StereoSGBM_create(
        minDisparity=int(min_disparity),
        numDisparities=num_disparities,
        blockSize=block_size,

        # Paramètres de régularisation StereoSGBM.
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,

        disp12MaxDiff=1,
        uniquenessRatio=int(
            stereo_config["uniqueness_ratio"]
        ),
        speckleWindowSize=int(
            stereo_config["speckle_window_size"]
        ),
        speckleRange=int(
            stereo_config["speckle_range"]
        ),

        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )


def compute_disparity(left, right, cfg):
    """
    Calcule une disparité propre avec :

    1. StereoSGBM gauche.
    2. StereoSGBM droit.
    3. Filtre WLS.

    Le filtre WLS réduit les erreurs dans les zones
    uniformes et préserve davantage les contours
    des voitures, vélos et piétons.
    """

    gray_left = to_gray(left)
    gray_right = to_gray(right)

    stereo_config = cfg["stereo"]

    if not hasattr(cv2, "ximgproc"):
        raise ImportError(
            "Le module cv2.ximgproc est introuvable. "
            "Installez opencv-contrib-python et désinstallez "
            "opencv-python dans votre environnement virtuel."
        )

    left_matcher = create_stereo_matcher(
        stereo_config
    )

    # Matcher droit nécessaire au filtre WLS. Certaines versions OpenCV ne
    # fournissent plus createRightMatcher dans Python ; nous recréons alors
    # exactement le matcher droit avec la formule officielle OpenCV.
    if hasattr(cv2.ximgproc, "createRightMatcher"):
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    else:
        num_disparities = max(16, (int(stereo_config["num_disparities"]) // 16) * 16)
        right_min_disparity = -int(stereo_config["min_disparity"]) - num_disparities
        right_matcher = create_stereo_matcher(
            stereo_config,
            min_disparity=right_min_disparity,
        )

    disparity_left_raw = left_matcher.compute(
        gray_left,
        gray_right
    )

    disparity_right_raw = right_matcher.compute(
        gray_right,
        gray_left
    )

    # Filtre Weighted Least Squares.
    if not hasattr(cv2.ximgproc, "createDisparityWLSFilter"):
        raise ImportError(
            "Le filtre WLS est absent. Réinstallez opencv-contrib-python "
            "dans l'environnement virtuel."
        )

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(
        matcher_left=left_matcher
    )

    wls_filter.setLambda(
        float(stereo_config["wls_lambda"])
    )

    wls_filter.setSigmaColor(
        float(stereo_config["wls_sigma_color"])
    )

    disparity_filtered_raw = wls_filter.filter(
        disparity_left_raw,
        gray_left,
        None,
        disparity_right_raw
    )

    # StereoSGBM stocke les disparités avec un facteur 16.
    disparity = (
        disparity_filtered_raw.astype(np.float32)
        / 16.0
    )

    # Les disparités nulles ou négatives ne sont pas fiables.
    disparity[disparity <= 0.1] = np.nan

    return disparity


def colorize_disparity(disparity):
    """
    Transforme la carte de disparité en image colorée
    pour la visualisation.
    """

    valid = np.isfinite(disparity) & (disparity > 0)

    normalized = np.zeros(
        disparity.shape,
        dtype=np.uint8
    )

    if not np.any(valid):
        return cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_TURBO
        )

    valid_values = disparity[valid]

    low_value = np.percentile(
        valid_values,
        2
    )

    high_value = np.percentile(
        valid_values,
        98
    )

    if high_value <= low_value:
        high_value = low_value + 1.0

    normalized_values = (
        (valid_values - low_value)
        / (high_value - low_value)
        * 255.0
    )

    normalized[valid] = np.clip(
        normalized_values,
        0,
        255
    ).astype(np.uint8)

    return cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_TURBO
    )
