"""Évalue chacune des cartes stéréo d'une séquence KITTI Raw sur le LiDAR projeté."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


MODEL_NAMES = ("StereoSGBM + WLS", "FoundationStereo")


def calibration_values(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        if key in {"R_rect_00", "P_rect_02", "P_rect_03", "R", "T"}:
            values[key] = np.fromstring(value, sep=" ")
    return values


def projected_reference(scan_path, shape, velo_to_rect, project_left, project_right):
    """Retient le point LiDAR le plus proche pour chaque pixel gauche."""
    height, width = shape
    scan = np.fromfile(scan_path, dtype="<f4").reshape(-1, 4)[:, :3]
    scan = scan[np.isfinite(scan).all(axis=1) & (scan[:, 0] > 0)]
    rect = scan @ velo_to_rect[:, :3].T + velo_to_rect[:, 3]
    left = rect @ project_left[:, :3].T + project_left[:, 3]
    right = rect @ project_right[:, :3].T + project_right[:, 3]
    front = (left[:, 2] > 0.1) & (right[:, 2] > 0.1)
    left, right = left[front], right[front]
    left_pixel = left[:, :2] / left[:, 2:3]
    right_pixel = right[:, :2] / right[:, 2:3]
    disparity = left_pixel[:, 0] - right_pixel[:, 0]
    x, y = np.rint(left_pixel).astype(np.int32).T
    keep = (
        (0 <= x) & (x < width) & (0 <= y) & (y < height)
        & (0 <= right_pixel[:, 0]) & (right_pixel[:, 0] < width)
        & (0 <= right_pixel[:, 1]) & (right_pixel[:, 1] < height)
        & (np.abs(left_pixel[:, 1] - right_pixel[:, 1]) <= 2)
        & (disparity > 0.1)
    )
    x, y, disparity = x[keep], y[keep], disparity[keep]
    pixel = y * width + x
    nearest = np.lexsort((left[keep, 2], pixel))
    _, first = np.unique(pixel[nearest], return_index=True)
    best = nearest[first]
    return y[best], x[best], disparity[best]


def evaluate(sequence, calibration, maps_by_model):
    camera = calibration_values(calibration / "calib_cam_to_cam.txt")
    velo = calibration_values(calibration / "calib_velo_to_cam.txt")
    velo_to_rect = camera["R_rect_00"].reshape(3, 3) @ np.column_stack(
        (velo["R"].reshape(3, 3), velo["T"])
    )
    project_left = camera["P_rect_02"].reshape(3, 4)
    project_right = camera["P_rect_03"].reshape(3, 4)
    frames = sorted(p.stem for p in (sequence / "image_02" / "data").glob("*.png"))
    if not frames:
        raise ValueError(f"Aucune image gauche dans {sequence}.")

    totals = {name: {"correct": 0, "valid": 0, "absolute_error_sum_px": 0.0} for name in maps_by_model}
    all_frames = []
    reference_pixels = 0
    for index, frame in enumerate(frames, 1):
        maps = {
            name: np.load(directory / f"{frame}.npy", allow_pickle=False)
            for name, directory in maps_by_model.items()
        }
        shape = next(iter(maps.values())).shape
        if len(shape) != 2 or any(disparity.shape != shape for disparity in maps.values()):
            raise ValueError(f"Dimensions des cartes incompatibles à l’image {frame}.")
        scan = sequence / "velodyne_points" / "data" / f"{frame}.bin"
        y, x, reference = projected_reference(scan, shape, velo_to_rect, project_left, project_right)
        if reference.size == 0:
            raise ValueError(f"Aucun pixel LiDAR de référence sur l’image {frame}.")

        reference_pixels += int(reference.size)
        row = {"frame": int(frame), "reference_pixels": int(reference.size), "accuracy": {}}
        for name, disparity in maps.items():
            values = disparity[y, x]
            valid = np.isfinite(values) & (values > 0.1)
            error = np.abs(values[valid] - reference[valid])
            correct = int(np.count_nonzero((error < 3.0) | (error < 0.05 * reference[valid])))
            totals[name]["correct"] += correct
            totals[name]["valid"] += int(valid.sum())
            totals[name]["absolute_error_sum_px"] += float(error.sum())
            row["accuracy"][name] = 100 * correct / reference.size
        all_frames.append(row)
        if index % 50 == 0 or index == len(frames):
            print(f"{sequence.name} : {index}/{len(frames)} images", flush=True)

    return {
        "sequence": sequence.name,
        "reference_pixels": reference_pixels,
        "frames": all_frames,
        "models": totals,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--sgbm", type=Path, required=True)
    parser.add_argument("--foundation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.sequence, args.sequence.parent, dict(zip(MODEL_NAMES, (args.sgbm, args.foundation))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent, delete=False) as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    Path(file.name).replace(args.output)


if __name__ == "__main__":
    main()
