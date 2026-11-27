"""Compatibilité : crée la courbe vitesse/temps du CSV ByteTrack actuel."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def to_number(value):
    if value is None or value.strip() == "":
        return None
    return float(value.replace(",", "."))


def main():
    parser = argparse.ArgumentParser(
        description="Crée une courbe de vitesse monde à partir du CSV ByteTrack."
    )
    parser.add_argument(
        "--csv",
        default="outputs/sequence_botsort/measurements.csv",
        help="Chemin du fichier measurements.csv",
    )
    parser.add_argument(
        "--track-id",
        default=None,
        help="ID à afficher ; meilleur ID si absent",
    )
    args = parser.parse_args()

    tracks = defaultdict(list)
    csv_path = Path(args.csv)

    with csv_path.open(encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            speed_kmh = to_number(row.get("world_speed_km_h"))
            time_s = to_number(row.get("time_s"))
            if (
                row.get("speed_status") == "accepted"
                and speed_kmh is not None
                and time_s is not None
            ):
                tracks[row["track_id"]].append(
                    (time_s, speed_kmh, row["object_class"])
                )

    if not tracks:
        raise ValueError("Aucune vitesse acceptée dans le CSV.")

    track_id = args.track_id or max(tracks, key=lambda key: len(tracks[key]))
    if track_id not in tracks or len(tracks[track_id]) < 2:
        raise ValueError(f"ID {track_id} absent ou insuffisant.")

    points = sorted(tracks[track_id])
    times = [point[0] for point in points]
    speeds = [point[1] for point in points]
    label = points[0][2]

    plt.figure(figsize=(10, 5))
    plt.plot(times, speeds, marker="o", linewidth=2, color="#1565C0")
    plt.title(f"Évolution de la vitesse — ID {track_id} ({label})")
    plt.xlabel("Temps (s)")
    plt.ylabel("Vitesse monde longitudinale estimée (km/h)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    output_path = csv_path.parent / f"speed_track_{track_id}.png"
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"Graphique créé : {output_path}")


if __name__ == "__main__":
    main()
