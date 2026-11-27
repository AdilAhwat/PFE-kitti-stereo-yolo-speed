from collections import Counter, defaultdict, deque


class TrackLabelStabilizer:
    """
    Stabilise la classe sémantique d'un objet suivi par ByteTrack.

    Exemple :
    - frame 1 : ID 4 = car
    - frame 2 : ID 4 = car
    - frame 3 : ID 4 = truck (hésitation YOLO)
    - frame 4 : ID 4 = car

    La classe affichée reste "car", car c'est la classe majoritaire.
    """

    def __init__(
        self,
        history_size=12,
        min_observations=3,
        min_majority_ratio=0.60,
        max_inactive_frames=40,
    ):
        self.history_size = history_size
        self.min_observations = min_observations
        self.min_majority_ratio = min_majority_ratio
        self.max_inactive_frames = max_inactive_frames

        self.histories = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )
        self.last_stable_label = {}
        self.last_seen_frame = {}

    def update(self, track_id, raw_label, confidence, frame_index):
        """
        Ajoute une observation YOLO et retourne une classe stabilisée.
        """
        if track_id is None:
            return raw_label

        track_id = int(track_id)
        confidence = float(confidence)

        self.histories[track_id].append({
            "label": raw_label,
            "confidence": confidence
        })

        self.last_seen_frame[track_id] = frame_index

        history = self.histories[track_id]

        # Au début, il n'y a pas encore assez de frames pour voter.
        if len(history) < self.min_observations:
            self.last_stable_label[track_id] = raw_label
            return raw_label

        # Vote pondéré : une détection confiante compte davantage.
        weighted_votes = Counter()

        for observation in history:
            label = observation["label"]
            score = observation["confidence"]

            # La détection participe au vote même avec une confiance faible,
            # mais une forte confiance a plus de poids.
            weighted_votes[label] += max(score, 0.05)

        stable_label, stable_weight = weighted_votes.most_common(1)[0]
        total_weight = sum(weighted_votes.values())
        majority_ratio = stable_weight / total_weight

        previous_label = self.last_stable_label.get(track_id, raw_label)

        # Si le vote n'est pas clair, nous gardons l'ancienne classe stable.
        if majority_ratio < self.min_majority_ratio:
            return previous_label

        self.last_stable_label[track_id] = stable_label
        return stable_label

    def cleanup(self, current_frame_index):
        """
        Supprime la mémoire des objets disparus depuis longtemps.
        Cela évite d'accumuler des IDs inutiles.
        """
        inactive_ids = []

        for track_id, last_frame in self.last_seen_frame.items():
            if current_frame_index - last_frame > self.max_inactive_frames:
                inactive_ids.append(track_id)

        for track_id in inactive_ids:
            self.histories.pop(track_id, None)
            self.last_stable_label.pop(track_id, None)
            self.last_seen_frame.pop(track_id, None)