import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def to_number(value):
    """Transforme une cellule CSV en nombre ou None."""

    if value is None:
        return None

    value = value.strip()

    if value == "":
        return None

    return float(value.replace(",", "."))


def read_measurements(csv_path):
    """Lit le CSV exporté par run_sequence.py."""

    tracks = defaultdict(list)

    with open(
        csv_path,
        "r",
        encoding="utf-8"
    ) as file:
        reader = csv.DictReader(
            file,
            delimiter=";"
        )

        for row in reader:
            try:
                time_s = to_number(row["time_s"])
                distance_m = to_number(row["distance_m"])
                speed_kmh = to_number(
                    row["world_speed_km_h"]
                )
            except (KeyError, ValueError):
                continue

            tracks[row["track_id"]].append({
                "time_s": time_s,
                "distance_m": distance_m,
                "speed_kmh": speed_kmh,
                "label": row["object_class"],
                "reliability": row["speed_reliability"],
                "status": row["speed_status"],
                "analysis_eligible": row.get("analysis_eligible", "True") == "True",
            })

    return tracks


def choose_best_track(tracks):
    """
    Choisit l'ID ayant le plus de vitesses acceptées
    de qualité high ou medium.
    """

    best_track_id = None
    best_count = -1

    for track_id, records in tracks.items():
        accepted_count = sum(
            1
            for record in records
            if record["status"] == "accepted"
            and record["analysis_eligible"]
            and record["reliability"] in ("high", "medium")
            and record["speed_kmh"] is not None
        )

        if accepted_count > best_count:
            best_count = accepted_count
            best_track_id = track_id

    return best_track_id


def plot_speed(records, output_path, track_id, label):
    """Crée la courbe vitesse / temps."""

    accepted = [
        record
        for record in records
        if record["status"] == "accepted"
        and record["analysis_eligible"]
        and record["reliability"] in ("high", "medium")
        and record["speed_kmh"] is not None
    ]

    accepted.sort(key=lambda record: record["time_s"])

    if len(accepted) < 2:
        print(
            "Pas assez de vitesses fiables "
            "pour créer le graphique de vitesse."
        )
        return

    times = [record["time_s"] for record in accepted]
    speeds = [record["speed_kmh"] for record in accepted]

    plt.figure(figsize=(10, 5))

    plt.plot(
        times,
        speeds,
        marker="o",
        linewidth=2,
        color="#1565C0",
        label="Vitesse estimée"
    )

    plt.title(
        f"Évolution de la vitesse — ID {track_id} ({label})"
    )

    plt.xlabel("Temps (s)")
    plt.ylabel("Vitesse longitudinale estimée (km/h)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=160)
    plt.close()

    print(f"Graphique vitesse créé : {output_path}")


def plot_distance(records, output_path, track_id, label):
    """Crée la courbe distance / temps."""

    valid = [
        record
        for record in records
        if record["distance_m"] is not None
    ]

    valid.sort(key=lambda record: record["time_s"])

    if len(valid) < 2:
        print(
            "Pas assez de distances "
            "pour créer le graphique de distance."
        )
        return

    times = [record["time_s"] for record in valid]
    distances = [
        record["distance_m"]
        for record in valid
    ]

    plt.figure(figsize=(10, 5))

    plt.plot(
        times,
        distances,
        marker="o",
        linewidth=2,
        color="#2E7D32",
        label="Distance estimée"
    )

    plt.title(
        f"Évolution de la distance — ID {track_id} ({label})"
    )

    plt.xlabel("Temps (s)")
    plt.ylabel("Distance à la caméra (m)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=160)
    plt.close()

    print(f"Graphique distance créé : {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Crée automatiquement les graphiques "
            "distance/temps et vitesse/temps."
        )
    )

    parser.add_argument(
        "--csv",
        default="outputs/sequence_botsort/measurements.csv",
        help="Chemin de measurements.csv"
    )

    parser.add_argument(
        "--track-id",
        default=None,
        help=(
            "ID à afficher ; si absent, "
            "le meilleur ID est choisi automatiquement"
        )
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV introuvable : {csv_path}"
        )

    tracks = read_measurements(csv_path)

    if not tracks:
        raise ValueError(
            "Aucune donnée valide trouvée dans le CSV."
        )

    track_id = args.track_id

    if track_id is None:
        track_id = choose_best_track(tracks)

    if track_id not in tracks:
        raise ValueError(
            f"ID introuvable : {track_id}"
        )

    records = tracks[track_id]

    label = records[0]["label"]

    output_dir = csv_path.parent

    speed_output = output_dir / (
        f"speed_track_{track_id}.png"
    )

    distance_output = output_dir / (
        f"distance_track_{track_id}.png"
    )

    print(
        f"Objet sélectionné : ID {track_id} ({label})"
    )

    plot_speed(
        records,
        speed_output,
        track_id,
        label
    )

    plot_distance(
        records,
        distance_output,
        track_id,
        label
    )


if __name__ == "__main__":
    main()
