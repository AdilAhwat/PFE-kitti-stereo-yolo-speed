"""Interface web de visualisation des séquences KITTI."""
from html import escape
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.calibration import StereoCalibration
from src.depth import disparity_to_depth
from src.stereo import colorize_disparity, compute_disparity, load_disparity_map

st.set_page_config(
    page_title="AutoDrive Vision",
    page_icon=":material/directions_car:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

OUTPUTS = PROJECT_ROOT / "outputs"
KITTI_DATE = "2011_09_26"
KITTI_0005_RESULTS = {
    "sequence_foundation",
    "sequence_foundation_yolo11m_seg",
    "sequence_foundation_temps_kitti",
    "sequence_foundation_vitesse_fenetre1",
}
DEPTH_LIMIT_M = 80.0
DEPTH_SCALE_POWER = 0.55
CLASS_LABELS = {
    "car": "Voiture", "truck": "Camion", "bus": "Bus", "person": "Piéton",
    "bicycle": "Vélo", "cyclist": "Cycliste", "motorcycle": "Moto",
}
VIEWS = {
    "Détection": "center_focus_strong",
    "Profondeur": "layers",
    "Séquence": "movie",
    "Suivi": "route",
    "Analyse": "monitoring",
}

STYLE = """
<style>
.stMainBlockContainer {max-width: 1460px; padding: 3.35rem 2.4rem 3rem;}
div[data-testid="stHeader"] {background: #090e18;}
.st-key-vision_header h1 {font-size: clamp(1.8rem, 3vw, 2.55rem); line-height:1.18;
    letter-spacing: -.045em; padding: 0 0 .12rem;}
.st-key-vision_header [data-testid="stCaptionContainer"] p {color: #9cbbdd;}
.st-key-vision_navigation [role="radio"] {min-height: 2.65rem; padding-inline: 1rem;}
.st-key-vision_navigation [role="radio"][aria-checked="true"] {background:#193f82; color:#edf6ff; border-color:#3a83ee;}
.stMainBlockContainer [data-testid="stVerticalBlockBorderWrapper"] {border-color: #26354c;}
.stMainBlockContainer [data-testid="stImage"] img {border-radius: 8px; object-fit: contain;}
.stMainBlockContainer [data-testid="stMetric"] {background: #121c2e; padding: .7rem .95rem; border-radius: 9px;}
.stMainBlockContainer [data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
.av-meta {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.65rem;}
.av-meta div {display:flex; gap:.35rem .7rem; align-items:baseline; flex-wrap:wrap;
    border:1px solid #28364d; background:#121c2e; border-radius:9px; padding:.54rem .8rem;}
.av-meta span {font-size:.75rem; color:#a6bad4;}
.av-meta strong {font-size:.88rem; font-weight:650; color:#f1f5fd; overflow-wrap:anywhere;}
.av-scale {position:relative; padding-bottom:1.15rem; color:#a6bad4; font-size:.72rem;}
.av-scale i {display:block; height:6px; border-radius:4px;
    background:linear-gradient(90deg,#800000,#ff3000,#f5f23c,#1be9db,#1647ff,#000080);}
.av-scale span {position:absolute; bottom:0; transform:translateX(-50%);}
.av-scale span:first-of-type {left:0; transform:none;}
.av-scale span:last-of-type {right:0; transform:none;}
.av-road-note {font-size:.75rem; color:#a6bad4; min-height:1.53rem; margin:0;}
.av-track-table {border:1px solid #26354c; border-radius:9px; overflow:auto;}
.av-track-table table {width:100%; min-width:34rem; border-collapse:collapse; font-size:.88rem;
    font-variant-numeric:tabular-nums;}
.av-track-table th {background:#141e30; color:#a6bad4; font-weight:600; text-align:left;}
.av-track-table td,.av-track-table th {padding:.53rem .78rem; white-space:nowrap;}
.av-track-table td {border-top:1px solid #213047;}
.av-track-table tbody tr:nth-child(even) {background:#101a2a;}
.av-id,.av-state {display:inline-block; padding:.11rem .48rem; border-radius:5px; font-weight:600;}
.av-id {color:#99d3ff; background:#163655;}
.av-state {color:#c3cfe2; background:#28354b; font-size:.76rem;}
.av-state.approach {color:#ffd0c7; background:#512733;}
.av-state.away {color:#b9f2d4; background:#163b36;}
.av-state.waiting {color:#ffe1a0; background:#473722;}
@media(max-width:850px) {.stMainBlockContainer {padding-inline:1rem;} .av-meta {grid-template-columns:1fr;}}
</style>
"""


@st.cache_data(show_spinner=False)
def load_measurements(csv_path_text: str, modified_ns: int) -> pd.DataFrame:
    """Charge le CSV créé par scripts/run_sequence.py."""
    del modified_ns  # L'horodatage invalide le cache après une nouvelle exécution.
    csv_path = Path(csv_path_text)
    if not csv_path.exists():
        return pd.DataFrame()
    # La colonne frame doit rester du texte. Sinon pandas transforme
    # 0000000002 en entier 2 et le dashboard cherche à tort l'image 2.png.
    dataframe = pd.read_csv(csv_path, sep=";", dtype={"frame": "string"})
    if "frame" in dataframe:
        dataframe["frame"] = (
            dataframe["frame"]
            .astype("string")
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(10)
        )
    numeric_columns = [
        "time_s", "track_id", "distance_m", "relative_speed_km_h", "world_speed_km_h",
        "speed_3d_km_h", "position_x_m", "position_y_m", "position_z_m",
        "relative_vx_mps", "relative_vy_mps", "relative_vz_mps", "world_vx_mps",
        "world_vy_mps", "world_vz_mps", "ego_speed_km_h", "speed_r2",
        "depth_residual_m", "ground_truth_id", "ground_truth_depth_m", "depth_abs_error_m",
        "relative_speed_abs_error_km_h",
    ]
    for column in numeric_columns:
        if column in dataframe:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    if "analysis_eligible" in dataframe:
        dataframe["analysis_eligible"] = dataframe["analysis_eligible"].astype(str).str.lower().isin(["true", "1", "yes"])
    if "track_id" in dataframe:
        dataframe["track_id"] = dataframe["track_id"].astype("Int64")
    return dataframe


def annotated_frame_paths(output_dir: Path) -> list[Path]:
    """Retourne uniquement les images de frames, dans l'ordre temporel."""
    if not output_dir.exists():
        return []
    return sorted(
        (path for path in output_dir.glob("*.png") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )


@st.cache_data(show_spinner=False)
def build_annotated_video(
    output_dir_text: str,
    fps: int,
    frames_signature: tuple,
) -> tuple[str, int, str]:
    """Assemble les images annotées de la séquence en vidéo."""
    del frames_signature  # Sert à invalider le cache lorsque les PNG changent.
    output_dir = Path(output_dir_text)
    frame_paths = annotated_frame_paths(output_dir)
    if not frame_paths:
        return "", 0, "Aucune image annotée trouvée dans le dossier de sortie."

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        return "", 0, f"La première image est illisible : {frame_paths[0]}"

    height, width = first.shape[:2]
    writer = None
    selected_codec = ""

    for codec, extension in (("VP80", ".webm"), ("avc1", ".mp4"), ("mp4v", ".mp4")):
        video_path = output_dir / f"sequence_annotee_{fps}fps{extension}"
        candidate = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*codec),
            float(fps),
            (width, height),
        )
        if candidate.isOpened():
            writer = candidate
            selected_codec = codec
            break
        candidate.release()

    if writer is None:
        return "", 0, "OpenCV ne trouve aucun encodeur vidéo utilisable."

    written = 0
    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(image)
        written += 1
    writer.release()

    if written == 0 or not video_path.exists():
        return "", 0, "La vidéo n'a pas pu être créée."
    return str(video_path), written, selected_codec


def accepted_mask(dataframe: pd.DataFrame) -> pd.Series:
    status = dataframe.get("speed_status", pd.Series("accepted", index=dataframe.index))
    eligible = dataframe.get("analysis_eligible", pd.Series(True, index=dataframe.index))
    return status.astype(str).str.lower().eq("accepted") & eligible.astype(bool)


def stereo_demo() -> None:
    st.subheader("Test d'une paire stéréo")
    st.caption("Chargez deux images rectifiées correspondantes : caméra 2 à gauche et caméra 3 à droite.")
    left_column, right_column = st.columns(2)
    with left_column:
        left_upload = st.file_uploader("Image gauche", type=["png", "jpg", "jpeg"], key="left")
    with right_column:
        right_upload = st.file_uploader("Image droite", type=["png", "jpg", "jpeg"], key="right")
    if not (left_upload and right_upload):
        st.info("Les deux images sont nécessaires pour calculer la disparité.")
        return

    left = cv2.imdecode(np.frombuffer(left_upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    right = cv2.imdecode(np.frombuffer(right_upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if left is None or right is None or left.shape[:2] != right.shape[:2]:
        st.error("Les deux images doivent être lisibles et avoir la même taille.")
        return
    try:
        config = load_config()
        disparity = compute_disparity(left, right, config)
        camera = config["camera"]
        depth = disparity_to_depth(disparity, camera["focal_length_px"], camera["baseline_m"])
    except Exception as error:
        st.error(f"Calcul stéréo impossible : {error}")
        return

    first, second, third = st.columns(3)
    first.image(cv2.cvtColor(left, cv2.COLOR_BGR2RGB), caption="Image gauche")
    second.image(cv2.cvtColor(colorize_disparity(disparity), cv2.COLOR_BGR2RGB), caption="Carte de disparité")
    with third:
        st.image(depth_colors(depth), caption="Profondeur")
        depth_legend()


def format_number(value, unit, digits=1, signed=False):
    if pd.isna(value) or not np.isfinite(value):
        return "—"
    number = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return f"{number.replace('.', ',')} {unit}"


def depth_colors(depth, scale_power=DEPTH_SCALE_POWER):
    valid = np.isfinite(depth) & (depth > 0)
    intensity = np.zeros(depth.shape, dtype=np.uint8)
    scaled = np.clip(depth[valid] / DEPTH_LIMIT_M, 0, 1) ** scale_power
    intensity[valid] = np.round(255 * (1 - scaled)).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(intensity, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    rgb[~valid] = (8, 16, 31)
    return rgb


@st.cache_data(show_spinner=False, max_entries=24)
def load_depth_preview(
    path: str, modified_ns: int, shape: tuple, focal_length: float, baseline: float, scale_power: float,
):
    disparity = load_disparity_map(path, shape)
    depth = disparity_to_depth(disparity, focal_length, baseline)
    return depth_colors(depth, scale_power)


def depth_legend():
    st.html(
        '<div class="av-scale" aria-label="Échelle de profondeur en mètres">'
        '<i></i><span>0 m</span><span style="left:31.9%">10 m</span>'
        '<span style="left:58.3%">30 m</span><span>80 m et +</span></div>'
    )


def scene_rows(dataframe):
    if dataframe.empty:
        return dataframe.assign(visible_speed=pd.Series(dtype=float), ttc_s=pd.Series(dtype=float))
    scene = dataframe.copy()
    visible = accepted_mask(scene) & scene["distance_m"].gt(0) & scene["distance_m"].le(30)
    if "depth_reliability" in scene:
        visible &= scene["depth_reliability"].isin(["high", "medium"])
    scene["visible_speed"] = scene.get(
        "relative_speed_km_h", pd.Series(np.nan, index=scene.index)
    ).where(visible)
    approaching = scene["visible_speed"].where(scene["visible_speed"] < 0)
    scene["ttc_s"] = scene["distance_m"] * 3.6 / -approaching
    return scene


def tracking_table(scene):
    if scene.empty:
        st.info("Aucun objet suivi sur cette image.")
        return

    lines = []
    for row in scene.sort_values("distance_m", na_position="last").to_dict("records"):
        track_id = row.get("track_id")
        object_id = "—" if pd.isna(track_id) else str(int(track_id))
        label = CLASS_LABELS.get(row.get("object_class"), str(row.get("object_class", "—")))
        speed = row["visible_speed"]
        if not row.get("analysis_eligible", True):
            state, style = "Hors analyse", ""
        elif row.get("speed_status") == "waiting":
            state, style = "En attente", "waiting"
        elif row.get("speed_status") == "rejected":
            state, style = "Rejetée", "waiting"
        elif pd.isna(speed):
            state, style = "Vitesse masquée", ""
        elif speed < 0:
            state, style = "Approche", "approach"
        elif speed > 0:
            state, style = "S’éloigne", "away"
        else:
            state, style = "Stable", ""
        lines.append(
            f'<tr><td><span class="av-id">{object_id}</span></td>'
            f'<td>{escape(label)}</td>'
            f'<td>{format_number(row.get("distance_m", np.nan), "m")}</td>'
            f'<td>{format_number(speed, "km/h", signed=True)}</td>'
            f'<td>{format_number(row["ttc_s"], "s")}</td>'
            f'<td><span class="av-state {style}">{state}</span></td></tr>'
        )

    st.html(
        '<div class="av-track-table"><table aria-label="Objets suivis">'
        '<thead><tr><th scope="col">ID</th><th scope="col">Classe</th>'
        '<th scope="col">Distance</th><th scope="col">Vitesse relative</th>'
        '<th scope="col">TTC</th><th scope="col">État</th></tr></thead>'
        f'<tbody>{"".join(lines)}</tbody></table></div>'
    )
    st.caption("Le TTC estime le temps de rapprochement longitudinal à vitesse relative constante.")


def show_depth(disparity_dir, frame, image, calibration):
    if disparity_dir is None:
        st.info("Choisis le dossier des cartes .npy dans les paramètres.")
        return
    path = disparity_dir / f"{int(frame):010d}.npy"
    if not path.is_file():
        st.info(f"Carte de disparité indisponible pour l’image {int(frame)}.")
        return
    if image is None:
        st.info("L’image annotée est nécessaire pour vérifier l’alignement de la carte.")
        return
    try:
        depth = load_depth_preview(
            str(path), path.stat().st_mtime_ns, image.shape[:2],
            calibration.focal_length_px, calibration.baseline_m, DEPTH_SCALE_POWER,
        )
    except (OSError, ValueError) as error:
        st.warning(f"Carte illisible ou incompatible : {error}")
        return
    st.image(depth)
    depth_legend()


def detection_view(image, frame, disparity_dir, calibration, scene):
    road, depth = st.columns(2, gap="medium")
    with road, st.container(border=True):
        st.markdown("**:material/center_focus_strong: Vue annotée**")
        if image is None:
            st.info("Image annotée indisponible.")
        else:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            st.html('<p class="av-road-note">|V| indique la vitesse 3D estimée.</p>')
    with depth, st.container(border=True):
        st.markdown("**:material/layers: Carte de profondeur**")
        show_depth(disparity_dir, frame, image, calibration)

    count, distance, ttc = st.columns(3)
    count.metric("Objets suivis", len(scene))
    distance.metric("Distance minimale", format_number(scene.get("distance_m", pd.Series(dtype=float)).min(), "m"))
    ttc.metric("TTC minimal", format_number(scene["ttc_s"].min(), "s"))
    st.subheader("Objets suivis")
    tracking_table(scene)


def tracking_view(dataframe, scene, output_dir):
    ids = sorted(dataframe["track_id"].dropna().astype(int).unique())
    if ids:
        names = dataframe.dropna(subset=["track_id"]).drop_duplicates("track_id", keep="last")
        classes = names.set_index("track_id")["object_class"].to_dict()
        frame_ids = set(scene["track_id"].dropna()) if "track_id" in scene else set()
        default_id = next((track_id for track_id in ids if track_id in frame_ids), ids[0])
        selection, filters = st.columns([1, 2], vertical_alignment="bottom")
        with selection:
            selected_id = st.selectbox(
                "Objet suivi", ids, index=ids.index(default_id),
                format_func=lambda value: f"#{value} · {CLASS_LABELS.get(classes.get(value), classes.get(value))}",
                key=f"track-{output_dir}",
            )
        with filters:
            raw_speeds = st.checkbox("Inclure les vitesses brutes dans les courbes")
        history = dataframe[dataframe["track_id"] == selected_id].sort_values("time_s").set_index("time_s")
        time_label = "Temps (s)"
        history.index.name = time_label
        selected_rows = scene_rows(history)
        latest = history.iloc[-1]
        latest_speed = selected_rows.loc[:, "visible_speed"].iloc[-1]
        distance, speed, count = st.columns(3)
        distance.metric("Dernière profondeur", format_number(latest["distance_m"], "m", digits=2))
        speed.metric("Dernière vitesse relative", format_number(latest_speed, "km/h", signed=True))
        count.metric("Observations", len(history))

        distance_chart, speed_chart = st.columns(2)
        with distance_chart, st.container(border=True):
            st.markdown("**Profondeur (m)**")
            st.line_chart(history["distance_m"], height=255, color="#55b9ff")
        with speed_chart, st.container(border=True):
            st.markdown("**Vitesse (km/h)**")
            columns = {
                "relative_speed_km_h": "Relative",
                "world_speed_km_h": "Longitudinale compensée",
                "speed_3d_km_h": "Norme 3D",
            }
            if raw_speeds:
                chart = history[[name for name in columns if name in history]].rename(columns=columns)
            else:
                chart = selected_rows[["visible_speed"]].rename(columns={"visible_speed": "Relative"})
            if chart.dropna(how="all").empty:
                st.info("Aucune vitesse disponible avec ce filtre.")
            else:
                st.line_chart(chart, height=255)
    else:
        st.info("Aucun identifiant de suivi dans cette sortie.")


def sequence_view(output_dir, frame_paths):
    st.subheader("Séquence annotée complète")
    if not frame_paths:
        st.info("Aucune image annotée disponible dans ce dossier.")
        return

    rate, play = st.columns([1, 2], vertical_alignment="bottom")
    with rate:
        fps = st.slider("Images par seconde", 1, 20, 10, key="sequence_fps")
    selected = (str(output_dir.resolve()), fps)
    playback = st.session_state.get("sequence_playback")
    is_current = playback is not None and playback[:2] == selected

    def start_playback():
        previous = st.session_state.get("sequence_playback")
        st.session_state["sequence_playback"] = (*selected, previous[2] + 1 if previous else 1)

    with play:
        st.button(
            "Relancer depuis le début" if is_current else "Lancer la vidéo",
            icon=":material/play_arrow:",
            type="primary",
            key="play_sequence",
            on_click=start_playback,
        )

    if not is_current:
        st.caption(f"{len(frame_paths)} images · {len(frame_paths) / fps:.1f} s à {fps} images/s")
        return

    signature = tuple((path.name, path.stat().st_mtime_ns) for path in frame_paths)
    with st.spinner("Préparation de la vidéo complète…"):
        path, written, message = build_annotated_video(str(output_dir), fps, signature)
    if not path:
        st.warning(message)
        return
    if written < len(frame_paths):
        st.warning(f"{len(frame_paths) - written} images illisibles ont été ignorées.")

    video = Path(path)
    mime = "video/webm" if video.suffix == ".webm" else "video/mp4"
    with st.container(key=f"sequence_player_{playback[2]}", border=True):
        st.video(video, format=mime, autoplay=True, muted=True)
        st.caption(f"{written} images · {written / fps:.1f} s à {fps} images/s")
        st.download_button(
            "Télécharger la vidéo", video.read_bytes(), video.name, mime, on_click="ignore",
        )


def analysis_view(dataframe, csv_path):
    truth = dataframe.get("ground_truth_depth_m", pd.Series(np.nan, index=dataframe.index))
    has_truth = truth.notna()
    accepted = dataframe.get("speed_status", pd.Series("", index=dataframe.index)).eq("accepted")
    depth_error = dataframe.get("depth_abs_error_m", pd.Series(np.nan, index=dataframe.index)).where(has_truth).dropna()
    speed_error = dataframe.get("relative_speed_abs_error_km_h", pd.Series(np.nan, index=dataframe.index)).where(has_truth & accepted).dropna()
    st.subheader("Validation KITTI")
    if not has_truth.any():
        st.info("Cette sortie ne contient aucune annotation KITTI associée.")
    else:
        count, depth, speed = st.columns(3)
        count.metric("Associations KITTI", int(has_truth.sum()))
        depth.metric("Erreur moyenne de profondeur", format_number(depth_error.mean(), "m", digits=2))
        speed.metric("Erreur moyenne de vitesse relative", format_number(speed_error.mean(), "km/h", digits=2))
        st.caption(f"{len(depth_error)} profondeurs et {len(speed_error)} vitesses acceptées comparées.")
        comparison = dataframe.loc[has_truth, ["ground_truth_depth_m", "distance_m"]].dropna()
        if not comparison.empty:
            comparison = comparison.rename(columns={"ground_truth_depth_m": "Référence KITTI (m)", "distance_m": "Profondeur estimée (m)"})
            st.scatter_chart(comparison, x="Référence KITTI (m)", y="Profondeur estimée (m)", height=300)
    if "object_class" in dataframe:
        with st.expander("Observations par classe"):
            classes = dataframe["object_class"].replace(CLASS_LABELS)
            st.bar_chart(classes.value_counts().rename("Observations"), height=240)
    with st.expander("Mesures brutes"):
        st.dataframe(dataframe, hide_index=True, height=380)
    st.download_button("Télécharger le CSV", csv_path.read_bytes(), "measurements.csv", "text/csv", icon=":material/download:")


def stereo_comparison_view(output_dir):
    reports = {
        path.stem.removeprefix("kitti_"): path
        for path in sorted((OUTPUTS / "stereo_evaluation").glob("kitti_*.json"))
    }
    if not reports:
        st.info("Les résultats de précision stéréo sur les séquences complètes sont indisponibles.")
        return

    sequences = list(reports)
    selected_run = re.fullmatch(r"sequence_(?:foundation|sgbm)_(\d{4})_methode_b", output_dir.name)
    default_sequence = selected_run[1] if selected_run and selected_run[1] in reports else sequences[0]
    sequence = st.selectbox("Séquence KITTI", sequences, index=sequences.index(default_sequence))
    report = json.loads(reports[sequence].read_text(encoding="utf-8"))
    reference_count = report["reference_pixels"]
    frame_count = len(report["frames"])
    count_label = f"{reference_count:,}".replace(",", " ")
    st.caption(f"{frame_count} images évaluées sur la séquence complète · {count_label} pixels LiDAR de référence")

    stats = []
    for card, (model, scores) in zip(st.columns(len(report["models"])), report["models"].items()):
        accuracy = 100 * scores["correct"] / reference_count
        with card:
            st.metric(model, format_number(accuracy, "%", digits=2), help="Part des pixels LiDAR correctement estimés.")
        valid_count = scores["valid"]
        stats.append({
            "Modèle": model,
            "Accuracy": format_number(accuracy, "%", digits=2),
            "Couverture": format_number(100 * valid_count / reference_count, "%", digits=2),
            "Erreur moyenne sur les pixels valides": format_number(
                scores["absolute_error_sum_px"] / valid_count if valid_count else np.nan, "px", digits=2
            ),
        })

    st.dataframe(pd.DataFrame(stats), hide_index=True, key="stereo-comparison-table")
    st.caption(
        "Un pixel est correct si l’erreur est inférieure à 3 pixels ou à 5 % de la disparité KITTI. "
        "Les valeurs manquantes comptent dans le total de référence. "
        "La couverture indique la part de ce total pour laquelle le modèle fournit une mesure valide."
    )
    st.subheader("Accuracy par image sur la séquence")
    history = pd.DataFrame([
        {"Image KITTI": frame["frame"], **frame["accuracy"]}
        for frame in report["frames"]
    ])
    st.line_chart(history, x="Image KITTI", y=list(report["models"]), y_label="Accuracy (%)", height=280)


def main():
    st.html(STYLE)
    results = sorted(
        (csv.parent for csv in OUTPUTS.glob("*/measurements.csv")),
        key=lambda path: (path / "measurements.csv").stat().st_mtime_ns,
        reverse=True,
    )
    with st.sidebar:
        st.header("Paramètres")
        selection = st.selectbox(
            "Résultats", [path.name for path in results] + ["Autre dossier"],
        )
        if selection == "Autre dossier":
            output_dir = Path(st.text_input("Dossier de résultats", str(OUTPUTS / "sequence_foundation")))
        else:
            output_dir = OUTPUTS / selection

        known_sequence = output_dir.resolve().parent == OUTPUTS and output_dir.name in KITTI_0005_RESULTS
        default_maps = OUTPUTS / "foundationstereo_kitti_0005" / "disparities"
        maps_path = st.text_input(
            "Dossier des cartes de disparité",
            value=str(default_maps) if known_sequence and default_maps.is_dir() else "",
            key=f"maps-{output_dir.resolve()}",
        ).strip()
        disparity_dir = Path(maps_path) if maps_path else None

        default_calib = PROJECT_ROOT.parent / "KITTI" / KITTI_DATE / "calib_cam_to_cam.txt"
        calibration_path = st.text_input(
            "Calibration KITTI",
            value=str(default_calib) if known_sequence and default_calib.is_file() else "",
            key=f"calib-{output_dir.resolve()}",
        ).strip()
        if calibration_path:
            try:
                calibration = StereoCalibration.from_kitti(calibration_path)
            except (ValueError, OSError) as error:
                st.error(f"Calibration illisible : {error}")
                st.stop()
        else:
            calibration = StereoCalibration(**load_config()["camera"])
        if st.button("Actualiser", icon=":material/refresh:"):
            load_measurements.clear()
            load_depth_preview.clear()

    with st.container(key="vision_header", gap="xxsmall"):
        st.markdown("# :material/directions_car: AutoDrive Vision")
        st.caption("Perception stéréo · KITTI · YOLO")

    with st.container(key="vision_navigation", gap="small"):
        view = st.segmented_control(
            "Navigation", list(VIEWS), default="Détection", required=True,
            format_func=lambda label: f":material/{VIEWS[label]}: {label}",
            label_visibility="collapsed",
            key="view",
        )

    csv_path = output_dir / "measurements.csv"
    dataframe = load_measurements(str(csv_path), csv_path.stat().st_mtime_ns) if csv_path.is_file() else pd.DataFrame()
    frame_paths = annotated_frame_paths(output_dir)
    if view == "Séquence":
        sequence_view(output_dir, frame_paths)
        return
    image_paths = {path.stem.zfill(10): path for path in frame_paths}
    frames = list(image_paths)
    if not frames and "frame" in dataframe:
        frames = sorted(name for name in dataframe["frame"].dropna().unique() if name.isdigit())
    if not frames:
        st.warning("Choisis un dossier de résultats contenant des images annotées ou un fichier measurements.csv.")
        if view == "Profondeur":
            stereo_demo()
        return

    observed = dataframe.loc[accepted_mask(dataframe), "frame"].dropna().tolist() if "frame" in dataframe else []
    default_frame = next((name for name in observed if name in frames), frames[0])
    source_values = dataframe.get("depth_source", pd.Series(dtype=str)).dropna().unique()
    source = " / ".join(source_values) if source_values.size else "Indisponible"
    sequence = "KITTI · Séq. 0005" if known_sequence else output_dir.name
    navigation, metadata = st.columns([1, 1.1], vertical_alignment="center")
    with navigation:
        frame = st.select_slider(
            "Image KITTI", frames, value=default_frame,
            format_func=lambda name: str(int(name)), key=f"frame-{output_dir.resolve()}",
        )
    with metadata:
        st.html(
            '<div class="av-meta">'
            f'<div><span>Séquence</span><strong>{escape(sequence)}</strong></div>'
            f'<div><span>Position</span><strong>{frames.index(frame) + 1} / {len(frames)}</strong></div>'
            f'<div><span>Stéréo</span><strong>{escape(source)}</strong></div>'
            '</div>'
        )

    image_path = image_paths.get(frame)
    image = cv2.imread(str(image_path)) if image_path else None
    frame_data = dataframe[dataframe["frame"] == frame] if "frame" in dataframe else dataframe.iloc[0:0]
    scene = scene_rows(frame_data)

    if view == "Détection":
        detection_view(image, frame, disparity_dir, calibration, scene)
    elif view == "Profondeur":
        with st.container(border=True):
            st.subheader("Carte de profondeur")
            show_depth(disparity_dir, frame, image, calibration)
        with st.expander("Tester une paire stéréo"):
            stereo_demo()
    elif dataframe.empty:
        st.info("Les mesures ne sont pas disponibles pour cette séquence.")
    elif view == "Suivi":
        tracking_view(dataframe, scene, output_dir)
    else:
        validation, comparison = st.tabs(["Validation KITTI", "Comparaison des modèles"])
        with validation:
            analysis_view(dataframe, csv_path)
        with comparison:
            stereo_comparison_view(output_dir)


if __name__ == "__main__":
    main()
