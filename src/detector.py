"""Détection YOLO et trackers interchangeables : BoT-SORT ou StrongSORT."""
import numpy as np

LABEL_ALIASES = {"van": "car", "tram": "bus", "pedestrian": "person", "person_sitting": "person", "cyclist": "cyclist"}


def box_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge_box(a, b):
    return np.array([min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])], dtype=np.float32)


def is_person_on_bicycle(person, bicycle):
    if box_iou(person, bicycle) >= 0.20:
        return True
    px1, _, px2, py2 = person
    bx1, by1, bx2, by2 = bicycle
    return abs((px1 + px2) / 2 - (bx1 + bx2) / 2) <= (bx2 - bx1) * .75 and by1 - (by2 - by1) * .8 <= py2 <= by2 + (by2 - by1) * .4


class YOLODetector:
    """YOLO produit les boîtes ; BoT-SORT ou StrongSORT leur attribue les IDs."""

    def __init__(self, model_path="yolo11n.pt", confidence=.35, allowed_classes=None,
                 tracker_config="botsort.yaml", tracker_type="botsort", reid_device="cpu", reid_weights=None):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.confidence, self.tracker_config = float(confidence), tracker_config
        self.tracker_type = str(tracker_type).lower()
        self.allowed_classes = allowed_classes or {"car", "truck", "bus", "person", "bicycle", "motorcycle"}
        names = self.model.names.items() if isinstance(self.model.names, dict) else enumerate(self.model.names)
        self.class_ids = [int(i) for i, label in names if self.normalize_label(label) in self.allowed_classes]
        self.strongsort = None
        if self.tracker_type == "strongsort":
            try:
                from boxmot.trackers.registry import create_tracker
            except ImportError as error:
                raise ImportError("StrongSORT exige BoxMOT. Exécutez : python -m pip install boxmot") from error
            # La fabrique BoxMOT lit la configuration StrongSORT et construit
            # aussi l'encodeur ReID. Le poids est téléchargé si absent.
            self.strongsort = create_tracker(
                "strongsort",
                reid_weights=reid_weights or "osnet_x0_25_msmt17.pt",
                device=reid_device,
                # Même seuil que YOLO : sinon StrongSORT supprimait les
                # détections entre 0.35 et 0.60 malgré leur présence YOLO.
                tracker_kwargs={
                    "min_conf": self.confidence,
                    "max_age": 40,
                    "n_init": 3,
                    "max_cos_dist": 0.35,
                },
            )
        elif self.tracker_type not in {"botsort", "bytetrack"}:
            raise ValueError("tracker_type doit être botsort, bytetrack ou strongsort.")

    @staticmethod
    def normalize_label(label):
        return LABEL_ALIASES.get(str(label).strip().lower().replace(" ", "_"), str(label).strip().lower().replace(" ", "_"))

    def _read_detections(self, result):
        if result.boxes is None:
            return []

        masks = (
            result.masks.data.cpu().numpy().astype(bool)
            if result.masks is not None
            else None
        )
        detections = []

        for index, box in enumerate(result.boxes):
            class_id = int(box.cls[0].item())
            label = self.normalize_label(result.names[class_id])
            if label not in self.allowed_classes:
                continue

            detection = {
                "box": box.xyxy[0].cpu().numpy(),
                "score": float(box.conf[0].item()),
                "class_id": class_id,
                "label": label,
                "mask": masks[index] if masks is not None else None,
            }
            if box.id is not None:
                detection["track_id"] = int(box.id[0].item())

            detections.append(detection)

        return detections

    def _detect_yolo(self, image):
        result = self.model.predict(
            image,
            conf=self.confidence,
            classes=self.class_ids,
            retina_masks=True,
            verbose=False,
        )[0]
        return self._read_detections(result)

    def _track_strongsort(self, image):
        detections = self._detect_yolo(image)
        rows = np.asarray([[*d["box"], d["score"], d["class_id"]] for d in detections], dtype=np.float32)
        if rows.size == 0:
            rows = np.empty((0, 6), dtype=np.float32)
        # BoxMOT : entrée N×6 [xyxy, score, class], sortie M×8
        # [x1,y1,x2,y2,track_id,score,class_id,detection_index].
        tracks = self.strongsort.update(rows, image)
        result = []
        for row in np.asarray(tracks):
            if len(row) < 8 or int(row[7]) < 0 or int(row[7]) >= len(detections):
                continue  # piste prédite sans détection : pas de profondeur actuelle
            source = detections[int(row[7])]
            result.append({
                "box": np.asarray(row[:4], dtype=np.float32),
                "score": float(row[5]),
                "class_id": int(row[6]),
                "label": source["label"],
                "track_id": int(row[4]),
                "mask": source["mask"],
            })
        return result
            
    def _track_ultralytics(self, image):
        result = self.model.track(
            image,
            persist=True,
            tracker=self.tracker_config,
            conf=self.confidence,
            classes=self.class_ids,
            retina_masks=True,
            verbose=False,
        )[0]
        if result.boxes is None or result.boxes.id is None:
            return []
        return self._read_detections(result)

    def track(self, image):
        detections = self._track_strongsort(image) if self.tracker_type == "strongsort" else self._track_ultralytics(image)
        return self.merge_cyclists(detections)

    def merge_cyclists(self, detections):
        merged, used = [], set()
        for p_i, person in enumerate(detections):
            if person["label"] != "person" or p_i in used:
                continue
            for b_i, bicycle in enumerate(detections):
                if bicycle["label"] == "bicycle" and b_i not in used and is_person_on_bicycle(person["box"], bicycle["box"]):
                    person_mask = person.get("mask")
                    bicycle_mask = bicycle.get("mask")

                    if person_mask is None:
                        cyclist_mask = bicycle_mask
                    elif bicycle_mask is None:
                        cyclist_mask = person_mask
                    else:
                        cyclist_mask = person_mask | bicycle_mask

                    merged.append({
                        "box": merge_box(person["box"], bicycle["box"]),
                        "score": max(person["score"], bicycle["score"]),
                        "class_id": -1,
                        "label": "cyclist",
                        "track_id": person["track_id"],
                        "mask": cyclist_mask,
                    })
                    used.update({p_i, b_i})
                    break
        return merged + [d for i, d in enumerate(detections) if i not in used]
