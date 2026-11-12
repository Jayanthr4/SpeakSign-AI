"""Experimental everyday hand gestures from MediaPipe's 21 landmarks.

These are heuristic visual gestures, not validated ASL signs.
Coordinates are normalized to the full camera image.
"""
from __future__ import annotations
import math

def classify(hand:list[list[float]])->dict:
    result={"gesture":None,"status":"uncertain","method":"landmark_geometry","confidence":None}
    if len(hand)!=21 or any(len(p)<2 for p in hand):
        return result
    def d(a,b):
        return math.hypot(hand[a][0]-hand[b][0],hand[a][1]-hand[b][1])
    palm=max(d(0,9),1e-6)
    # Compare joint-to-tip distances in the hand's own scale.
    def straight(tip,pip,mcp):
        return d(tip,mcp)>1.35*d(pip,mcp) and d(0,tip)>d(0,pip)+0.12*palm
    index=straight(8,6,5)
    middle=straight(12,10,9)
    ring=straight(16,14,13)
    pinky=straight(20,18,17)
    folded=not any((index,middle,ring,pinky))
    thumb_extended=d(4,5)>0.85*palm and d(4,17)>1.05*palm
    thumb_delta=(hand[4][1]-hand[2][1])/palm

    gesture=None
    if folded and thumb_extended and thumb_delta< -0.5:
        gesture="THUMBS_UP"
    elif folded and thumb_extended and thumb_delta>0.5:
        gesture="THUMBS_DOWN"
    elif index and middle and not ring and not pinky and d(8,12)>0.48*palm:
        gesture="VICTORY"
    elif d(4,8)<0.32*palm and middle and ring and pinky:
        gesture="OK"
    if gesture:
        result.update(gesture=gesture,status="candidate")
    return result

def is_open_palm(hand:list[list[float]])->bool:
    if len(hand)!=21:return False
    def d(a,b):
        return math.hypot(hand[a][0]-hand[b][0],hand[a][1]-hand[b][1])
    palm=max(d(0,9),1e-6)
    return all(d(0,tip)>d(0,pip)+0.15*palm for tip,pip in ((8,6),(12,10),(16,14),(20,18)))
