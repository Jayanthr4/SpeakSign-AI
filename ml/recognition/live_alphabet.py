"""Experimental static ASL handshape cues; NOT a trained A-Z classifier.

Only high-separation handshapes are emitted. Ambiguous configurations abstain.
J/Z require motion and are deliberately unsupported.
"""
from __future__ import annotations
import math

SUPPORTED=("B","D","I","L","V","Y")

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
    if letter:
        result.update(letter=letter,status="candidate")
    return result
