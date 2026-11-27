"""Évalue Precision, Recall et mAP d'un modèle YOLO sur un YAML KITTI préparé."""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Évaluation YOLO : precision, recall, mAP50 et mAP50-95.")
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--data", required=True, help="YAML Ultralytics décrivant train/val KITTI")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    metrics = YOLO(args.model).val(data=args.data, imgsz=args.imgsz, device=args.device)
    box = metrics.box
    print(f"Precision : {box.mp:.4f}")
    print(f"Recall : {box.mr:.4f}")
    print(f"mAP@50 : {box.map50:.4f}")
    print(f"mAP@50-95 : {box.map:.4f}")


if __name__ == "__main__":
    main()
