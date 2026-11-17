"""Collect calibration samples and train gaze models.

Two models are trained from the same calibration pass:
  * GazeModel     — classifies which screen you're looking at (KNN).
  * GazeRegressor — per-screen, maps gaze features -> fractional (fx, fy)
                    position within that screen, enabling within-screen control.
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

DEFAULT_PATH = os.path.abspath("calibration.json")

# 9-point grid as fractional (fx, fy) targets within a screen, each with a
# human-readable spot to physically look at. Margins of 0.1 keep the targets
# off the extreme bezel so they're natural to fixate.
GRID: List[Tuple[float, float, str]] = [
    (0.1, 0.1, "TOP-LEFT corner"),
    (0.5, 0.1, "TOP edge, middle"),
    (0.9, 0.1, "TOP-RIGHT corner"),
    (0.1, 0.5, "LEFT edge, middle"),
    (0.5, 0.5, "CENTER"),
    (0.9, 0.5, "RIGHT edge, middle"),
    (0.1, 0.9, "BOTTOM-LEFT corner"),
    (0.5, 0.9, "BOTTOM edge, middle"),
    (0.9, 0.9, "BOTTOM-RIGHT corner"),
]


class CalibrationData:
    def __init__(self):
        self.X: List[List[float]] = []   # feature vectors
        self.y: List[int] = []           # screen index labels
        self.fx: List[float] = []        # fractional x target within screen
        self.fy: List[float] = []        # fractional y target within screen

    def add(self, vector: np.ndarray, screen_index: int, fx: float = 0.5, fy: float = 0.5):
        self.X.append(vector.tolist())
        self.y.append(int(screen_index))
        self.fx.append(float(fx))
        self.fy.append(float(fy))

    @property
    def has_grid(self) -> bool:
        return len(self.fx) == len(self.X) and len(self.fx) > 0

    def save(self, path: str = DEFAULT_PATH):
        with open(path, "w") as f:
            json.dump({"X": self.X, "y": self.y, "fx": self.fx, "fy": self.fy}, f)

    @classmethod
    def load(cls, path: str = DEFAULT_PATH) -> "CalibrationData":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No calibration found at {path}. Run `uv run gaze-cursor --calibrate` first."
            )
        with open(path) as f:
            d = json.load(f)
        data = cls()
        data.X = d["X"]
        data.y = d["y"]
        data.fx = d.get("fx", [])
        data.fy = d.get("fy", [])
        return data


class GazeModel:
    """Scaler + KNN classifier: gaze features -> screen index."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = KNeighborsClassifier(n_neighbors=5, weights="distance")

    def train(self, data: CalibrationData):
        X = np.array(data.X)
        y = np.array(data.y)
        Xs = self.scaler.fit_transform(X)
        self.clf.set_params(n_neighbors=max(1, min(5, len(X))))
        self.clf.fit(Xs, y)

    def predict(self, vector: np.ndarray) -> Tuple[int, float]:
        Xs = self.scaler.transform(vector.reshape(1, -1))
        proba = self.clf.predict_proba(Xs)[0]
        best = int(np.argmax(proba))
        return int(self.clf.classes_[best]), float(proba[best])


class GazeRegressor:
    """Per-screen KNN regressors: gaze features -> (fx, fy) within that screen."""

    def __init__(self):
        self.scalers: Dict[int, StandardScaler] = {}
        self.models: Dict[int, KNeighborsRegressor] = {}

    def train(self, data: CalibrationData):
        X = np.array(data.X)
        y = np.array(data.y)
        targets = np.column_stack([data.fx, data.fy])
        for screen in sorted(set(y.tolist())):
            mask = y == screen
            Xs_raw = X[mask]
            if len(Xs_raw) < 2:
                continue
            scaler = StandardScaler()
            Xs = scaler.fit_transform(Xs_raw)
            reg = KNeighborsRegressor(
                n_neighbors=max(1, min(8, len(Xs_raw))), weights="distance"
            )
            reg.fit(Xs, targets[mask])
            self.scalers[screen] = scaler
            self.models[screen] = reg

    def predict(self, vector: np.ndarray, screen: int) -> Tuple[float, float]:
        if screen not in self.models:
            return 0.5, 0.5
        Xs = self.scalers[screen].transform(vector.reshape(1, -1))
        fx, fy = self.models[screen].predict(Xs)[0]
        return float(min(max(fx, 0.0), 1.0)), float(min(max(fy, 0.0), 1.0))
