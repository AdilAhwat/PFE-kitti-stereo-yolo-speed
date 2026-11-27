"""Produit un rapport quantitatif à partir du CSV validé par les tracklets KITTI."""

import argparse
import csv
from pathlib import Path

import numpy as np


def to_number(value):
    if value is None or value.strip() == "":
        return float("nan")
    return float(value.replace(",", "."))


def main():
    parser = argparse.ArgumentParser(
        description="Calcule les erreurs profondeur/vitesse par rapport aux tracklets KITTI."
    )
    parser.add_argument(
        "--csv",
        default="outputs/sequence_botsort/measurements.csv",
        help="CSV produit avec --tracklets et --calib-velo-to-cam",
    )
    args = parser.parse_args()

    rows = []
    with Path(args.csv).open(encoding="utf-8") as file:
        for row in csv.DictReader(file, delimiter=";"):
            if row.get("analysis_eligible") != "True":
                continue
            if row.get("ground_truth_id", "") == "":
                continue
            rows.append(row)

    if not rows:
        raise ValueError(
            "Aucune association avec un tracklet. Lancez run_sequence.py avec --tracklets."
        )

    depth_errors = np.array([to_number(row["depth_abs_error_m"]) for row in rows])
    accepted_rows = [
        row for row in rows
        if row.get("speed_status") == "accepted"
    ]

    speed_errors = np.array([
        to_number(row["relative_speed_abs_error_km_h"])
        for row in accepted_rows
    ])

    valid_depth = depth_errors[np.isfinite(depth_errors)]
    valid_speed = speed_errors[np.isfinite(speed_errors)]

    print("=== Validation KITTI Raw Tracklets ===")
    print(f"Associations détection / vérité terrain : {len(rows)}")
    print(f"Vitesses retenues après stabilisation Kalman : {len(accepted_rows)}")
    print(
        f"Erreur absolue moyenne de profondeur : {np.mean(valid_depth):.2f} m"
        if valid_depth.size else "Erreur de profondeur : données insuffisantes"
    )
    print(
        f"RMSE profondeur : {np.sqrt(np.mean(valid_depth ** 2)):.2f} m"
        if valid_depth.size else "RMSE profondeur : données insuffisantes"
    )
    print(
        f"Erreur absolue moyenne de vitesse relative : {np.mean(valid_speed):.2f} km/h"
        if valid_speed.size else "Erreur de vitesse : données insuffisantes"
    )
    print(
        f"RMSE vitesse relative : {np.sqrt(np.mean(valid_speed ** 2)):.2f} km/h"
        if valid_speed.size else "RMSE vitesse : données insuffisantes"
    )
    print(
        "Note : les vitesses en attente ou rejetées ne sont pas incluses dans "
        "l'erreur de vitesse. La validation porte sur la vitesse relative, "
        "dans le repère véhicule."
    )


if __name__ == "__main__":
    main()
