"""Dependency-free synthetic smoke tests for local ILY handshape logic.

Run: python -m unittest tests.test_ily_gesture
"""
import unittest
from ml.recognition.live_gestures import classify

class ILYGestureTests(unittest.TestCase):
    def test_invalid_landmarks_abstain(self):
        self.assertIsNone(classify([])["gesture"])

    def test_ily_handshape(self):
        # Synthetic upright palm; coordinates normalized to image.
        hand=[
            [0.50,0.85], # wrist 0
            [0.38,0.77],[0.28,0.68],[0.21,0.58],[0.15,0.49], # thumb
            [0.43,0.61],[0.41,0.47],[0.40,0.31],[0.39,0.17], # index
            [0.50,0.59],[0.51,0.55],[0.53,0.67],[0.53,0.72], # middle folded
            [0.57,0.61],[0.59,0.57],[0.60,0.69],[0.59,0.74], # ring folded
            [0.64,0.65],[0.70,0.52],[0.74,0.38],[0.77,0.27] # pinky
        ]
        self.assertEqual(classify(hand)["gesture"],"ILY_HANDSHAPE")
        self.assertEqual(classify(hand)["status"],"candidate")

if __name__=="__main__":
    unittest.main()
