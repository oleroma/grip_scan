"""MediaPipe hand landmarks (Tasks API) as pixel coordinates."""
import os

import cv2
import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")

NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
FINGERS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}
SEGMENTS = {  # segment names along each finger's landmark chain
    "thumb": ["metacarpal", "proximal", "distal"],
    "index": ["proximal", "middle", "distal"],
    "middle": ["proximal", "middle", "distal"],
    "ring": ["proximal", "middle", "distal"],
    "pinky": ["proximal", "middle", "distal"],
}


class Landmarker:
    def __init__(self, model_path=None):
        model_path = os.path.abspath(model_path or DEFAULT_MODEL)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe model not found at {model_path}.\nDownload it from {MODEL_URL}")
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_hands=1, min_hand_detection_confidence=0.3)
        self._det = vision.HandLandmarker.create_from_options(opts)

    def detect(self, image_bgr):
        """21x2 pixel coordinates, handedness label and score, or None."""
        import mediapipe as mp
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = self._det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.hand_landmarks:
            return None
        h, w = image_bgr.shape[:2]
        px = np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]])
        cat = res.handedness[0][0]
        return px, cat.category_name, float(cat.score)
