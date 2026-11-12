"""Experimental static ASL handshape cues; NOT a trained A-Z classifier.

Only high-separation handshapes are emitted. Ambiguous configurations abstain.
J/Z require motion and are deliberately unsupported.
"""
from __future__ import annotations
import math

SUPPORTED=("B","D","I","L","O","V","Y")

def classify(hand:list[list[float]])->dict:
    result={"letter":None,"status":"uncertain","method":"experimental_geometry","supported_letters":list(SUPPORTED)}
    if len(hand)!=21 or any(len(p)<2 for p in hand):
        return result
    def dist(a,b):
        return math.hypot(hand[a][0]-hand[b][0],hand[a][1]-hand[b][1])
    palm=max(dist(0,9),1e-6)
    # Wrist-to-tip compared to wrist-to-middle-joint, with margin.
    extended=[]
    for tip,pip,mcp in ((4,3,2),(8,6,5),(12,10,9),(16,14,13),(20,18,17)):
        if tip==4:
            # Thumb openness measured against index MCP; palm-scale normalized.
            extended.append(dist(4,5)>0.85*palm and dist(4,17)>1.05*palm)
        else:
            extended.append(dist(0,tip)>dist(0,pip)+0.22*palm and dist(tip,mcp)>0.62*palm)
    thumb,index,middle,ring,pinky=extended
    # Fingerspelling is 3D, viewpoint dependent; do not claim definitive letters.
    letter=None
    if index and middle and ring and pinky and not thumb:
        letter="B"
    elif index and not middle and not ring and not pinky and not thumb:
        letter="D" if dist(8,12)>0.48*palm else None
    elif not index and not middle and not ring and pinky and not thumb:
        letter="I"
    elif thumb and index and not middle and not ring and not pinky:
        letter="L"
    elif index and middle and not ring and not pinky and not thumb and dist(8,12)>0.55*palm:
        letter="V"
    elif thumb and not index and not middle and not ring and pinky:
        letter="Y"
    # Rounded O: all four fingertips curl toward the thumb tip.
    # This is a coarse 2D cue and can confuse other closed handshapes.
    if letter is None and not index and not middle and not ring and not pinky:
        fingertips=(8,12,16,20)
        thumb_contact=sum(dist(4,t)<0.53*palm for t in fingertips)
        compact=sum(dist(t,9)<1.05*palm for t in fingertips)
        if thumb_contact>=2 and compact>=3:
            letter="O"
    if letter:
        result.update(letter=letter,status="candidate")
    return result

def classify_gesture(hand:list[list[float]])->dict:
    """Viewpoint-dependent thumbs-up pose; not a validated ASL sign."""
    result={"gesture":None,"status":"uncertain","method":"experimental_geometry"}
    if len(hand)!=21 or any(len(p)<2 for p in hand):return result
    def d(a,b):return math.hypot(hand[a][0]-hand[b][0],hand[a][1]-hand[b][1])
    palm=max(d(0,9),1e-6)
    folded=all(d(0,tip)<d(0,pip)+0.12*palm for tip,pip in ((8,6),(12,10),(16,14),(20,18)))
    thumb_open=d(4,5)>0.95*palm and d(4,17)>1.15*palm
    # Image y increases downward; thumb should point upward.
    thumb_up=hand[4][1]<hand[2][1]-0.35*palm
    if folded and thumb_open and thumb_up:
        result.update(gesture="THUMBS_UP",status="candidate")
    return result
