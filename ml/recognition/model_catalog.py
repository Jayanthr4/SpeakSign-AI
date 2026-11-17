"""Truthful catalog of local recognition capabilities and model availability."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def catalog():
    pretrained = ROOT / "storage/models/gesture_recognizer.task"
    return [
        {
            "id": "mediapipe_gesture_recognizer",
            "source": "Google MediaPipe",
            "task": "generic_static_gestures",
            "classes": [
                "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down",
                "Thumb_Up", "Victory", "ILoveYou",
            ],
            "installed": pretrained.is_file(),
            "model_path": str(pretrained.relative_to(ROOT)),
            "validated_asl_translation": False,
        },
        {
            "id": "local_geometry",
            "source": "SpeakSign experimental rules",
            "task": "experimental_gesture_candidates",
            "classes": [
                "ILY_HANDSHAPE", "MIDDLE_FINGER", "THUMBS_UP",
                "THUMBS_DOWN", "VICTORY", "OK", "HELLO_WAVE",
            ],
            "installed": True,
            "validated_asl_translation": False,
        },
    ]
