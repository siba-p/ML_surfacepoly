"""Simple history container that mirrors the notebook helper."""
from __future__ import annotations

from typing import Dict, List, Sequence


class CustomHistory:
    def __init__(self, train_labels: Sequence[str], valid_labels: Sequence[str]):
        self.train_labels = list(train_labels)
        self.valid_labels = list(valid_labels)
        self.history: Dict[str, List[float]] = {label: [] for label in self.train_labels + self.valid_labels}

    def update(self, train_values, valid_values):
        for label, value in zip(self.train_labels, train_values):
            self._store(label, value)
        for label, value in zip(self.valid_labels, valid_values):
            self._store(label, value)

    def _store(self, label: str, value):
        processed = value.numpy() if hasattr(value, "numpy") else value
        self.history[label].append(processed)

    def to_dict(self):
        return self.history
