"""Gaze feature extraction from a webcam frame using MediaPipe FaceLandmarker.

Uses the current MediaPipe Tasks API (the legacy `mp.solutions.face_mesh` module
was removed in recent releases). Landmark indices are unchanged from FaceMesh.
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

# Iris landmark indices (require refine_landmarks=True).
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Eye corner / lid landmarks for building relative iris position.
LEFT_EYE = {"outer": 33, "inner": 133, "top": 159, "bottom": 145}
RIGHT_EYE = {"outer": 263, "inner": 362, "top": 386, "bottom": 374}

# A few stable face points to estimate head orientation.
NOSE_TIP = 1
CHIN = 199
LEFT_FACE = 234
RIGHT_FACE = 454


@dataclass
class GazeFeatures:
    """A compact vector describing where the eyes/head are pointed.

    All values roughly normalized; used by the calibrated model, not meaningful
    in absolute units on their own.
    """

    vector: np.ndarray  # feature vector fed to the screen classifier


def _xy(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def _iris_center(landmarks, iris_idx, w, h):
    pts = np.array([_xy(landmarks, i, w, h) for i in iris_idx])
    return pts.mean(axis=0)


def _eye_ratios(landmarks, eye, w, h):
    """Return (horizontal, vertical) iris position within the eye, each ~0..1."""
    outer = _xy(landmarks, eye["outer"], w, h)
    inner = _xy(landmarks, eye["inner"], w, h)
    top = _xy(landmarks, eye["top"], w, h)
    bottom = _xy(landmarks, eye["bottom"], w, h)
    iris = _iris_center(landmarks, LEFT_IRIS if eye is LEFT_EYE else RIGHT_IRIS, w, h)

    eye_w = np.linalg.norm(inner - outer) + 1e-6
    eye_h = np.linalg.norm(bottom - top) + 1e-6
    # Project iris onto the horizontal (outer->inner) and vertical (top->bottom) axes.
    h_ratio = np.dot(iris - outer, inner - outer) / (eye_w ** 2)
    v_ratio = np.dot(iris - top, bottom - top) / (eye_h ** 2)
    return float(h_ratio), float(v_ratio)


class GazeEstimator:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Missing model: {model_path}\nDownload it with:\n"
                "  curl -sSL -o gaze_cursor/face_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task"
            )
        # VIDEO mode tracks landmarks across frames, which is far less jittery
        # than IMAGE mode (which re-detects from scratch every frame).
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._ts = 0  # monotonic timestamp (ms) required by VIDEO mode

    def extract(self, rgb_frame) -> Optional[GazeFeatures]:
        h, w = rgb_frame.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self._ts += 33  # ~30 fps; must strictly increase
        result = self.landmarker.detect_for_video(mp_image, self._ts)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]

        lh, lv = _eye_ratios(lm, LEFT_EYE, w, h)
        rh, rv = _eye_ratios(lm, RIGHT_EYE, w, h)

        # Head orientation proxies, normalized by face width so distance to the
        # camera doesn't dominate.
        nose = _xy(lm, NOSE_TIP, w, h)
        left_face = _xy(lm, LEFT_FACE, w, h)
        right_face = _xy(lm, RIGHT_FACE, w, h)
        chin = _xy(lm, CHIN, w, h)
        face_w = np.linalg.norm(right_face - left_face) + 1e-6
        mid_face = (left_face + right_face) / 2.0

        yaw = (nose[0] - mid_face[0]) / face_w          # left/right head turn
        pitch = (nose[1] - chin[1]) / face_w            # up/down head tilt

        vector = np.array([lh, lv, rh, rv, (lh + rh) / 2, (lv + rv) / 2, yaw, pitch])
        return GazeFeatures(vector=vector)

    def close(self):
        self.landmarker.close()
