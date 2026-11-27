"""Évalue la continuité des IDs BoT-SORT à partir du CSV validé KITTI."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Mesure ID switches, pertes et fragments de suivi.")
    parser.add_argument("--csv", required=True, help="measurements.csv créé avec --tracklets")
    parser.add_argument("--output", default=None, help="Rapport JSON de sortie")
    parser.add_argument("--tracker", default="tracker", help="Nom du tracker évalué, ex. StrongSORT")
    args = parser.parse_args()

    by_gt = defaultdict(list)
    with Path(args.csv).open(encoding="utf-8") as file:
        for row in csv.DictReader(file, delimiter=";"):
            if not row.get("ground_truth_id") or not row.get("track_id"):
                continue
            by_gt[row["ground_truth_id"]].append((float(row["time_s"]), int(row["track_id"])))

    switches = fragments = interruptions = observations = 0
    durations = []
    for rows in by_gt.values():
        rows.sort()
        observations += len(rows)
        durations.append(rows[-1][0] - rows[0][0])
        previous_time, previous_id = rows[0]
        local_fragments = 1
        for time_s, track_id in rows[1:]:
            if time_s - previous_time > 0.11:
                interruptions += 1
                local_fragments += 1
            if track_id != previous_id:
                switches += 1
            previous_time, previous_id = time_s, track_id
        fragments += local_fragments

    report = {
        "ground_truth_tracks_observed": len(by_gt),
        "matched_observations": observations,
        "id_switches": switches,
        "interrupted_trajectories": interruptions,
        "trajectory_fragments": fragments,
        "mean_track_duration_s": round(sum(durations) / len(durations), 3) if durations else 0.0,
    }
    output = Path(args.output) if args.output else Path(args.csv).parent / "tracking_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"=== Évaluation {args.tracker} ===")
    for key, value in report.items():
        print(f"{key} : {value}")
    print(f"Rapport créé : {output}")


if __name__ == "__main__":
    main()
