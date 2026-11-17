"""Optional on-device MediaPipe Gesture Recognizer.

Download the official gesture_recognizer.task from Google's MediaPipe model
documentation into storage/models/gesture_recognizer.task. No network requests
are made by this module. This model recognizes a limited set of gestures,
not arbitrary ASL vocabulary.
"""
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[2] / "storage/models/gesture_recognizer.task"

def available() -> bool:
    return MODEL_PATH.is_file()

class GestureRecognizer:
    def __init__(self, model_path=MODEL_PATH):
        import mediapipe as mp
        self.mp = mp
        opts = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
        )
        self.recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(opts)

    def predict(self, rgb_image):
        import numpy as np
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                              data=np.ascontiguousarray(rgb_image))
        result = self.recognizer.recognize(image)
        output = []
        for hand_idx, categories in enumerate(result.gestures):
            if not categories:
                continue
            best = categories[0]
            output.append({
                "gesture": best.category_name if best.category_name != "None" else None,
                "status": "candidate" if best.category_name != "None" else "uncertain",
                "confidence": round(float(best.score), 4),
                "method": "mediapipe_pretrained",
                "hand_index": hand_idx,
            })
        return output

    def close(self):
        self.recognizer.close()
