"""Experimental everyday gestures from normalized MediaPipe hand landmarks.

Rules intentionally abstain on ambiguous poses. Not validated ASL recognition.
"""
from __future__ import annotations
import math

def _geometry(hand):
    if len(hand)!=21 or any(len(p)<2 for p in hand):
        return None
    def d(a,b):
        return math.hypot(hand[a][0]-hand[b][0],hand[a][1]-hand[b][1])
    def angle(a,b,c):
        ux,uy=hand[a][0]-hand[b][0],hand[a][1]-hand[b][1]
        vx,vy=hand[c][0]-hand[b][0],hand[c][1]-hand[b][1]
        norm=max(math.hypot(ux,uy)*math.hypot(vx,vy),1e-9)
        return math.degrees(math.acos(max(-1,min(1,(ux*vx+uy*vy)/norm))))
    palm=max(d(0,9),1e-6)
    states={}
    for name,mcp,pip,dip,tip in (
        ("index",5,6,7,8),("middle",9,10,11,12),
        ("ring",13,14,15,16),("pinky",17,18,19,20)):
        # Two criteria help distinguish straight fingers from folded ones.
        states[name]=angle(mcp,pip,tip)>155 and d(0,tip)>d(0,pip)+0.10*palm
    return d,angle,palm,states

def classify(hand:list[list[float]])->dict:
    result={"gesture":None,"status":"uncertain","method":"landmark_geometry","confidence":None}
    g=_geometry(hand)
    if g is None:return result
    d,angle,palm,s=g
    folded=not any(s.values())
    # Thumb should be straight, visibly separated, and pointing vertically
    # relative to the image for thumbs-up/down.
    thumb_straight=angle(2,3,4)>145
    thumb_separated=d(4,5)>0.65*palm and d(4,17)>0.95*palm
    thumb_dx=(hand[4][0]-hand[2][0])/palm
    thumb_dy=(hand[4][1]-hand[2][1])/palm
    vertical=abs(thumb_dy)>abs(thumb_dx)*1.2
    gesture=None
    if folded and thumb_straight and thumb_separated and vertical:
        if thumb_dy< -0.48:gesture="THUMBS_UP"
        elif thumb_dy>0.48:gesture="THUMBS_DOWN"
    # Explicitly exclude thumb-up and require two *straight* raised fingers.
    elif s["index"] and s["middle"] and not s["ring"] and not s["pinky"] and d(8,12)>0.48*palm:
        gesture="VICTORY"
    elif d(4,8)<0.32*palm and s["middle"] and s["ring"] and s["pinky"]:
        gesture="OK"
    if gesture:result.update(gesture=gesture,status="candidate")
    return result

def is_open_palm(hand:list[list[float]])->bool:
    g=_geometry(hand)
    return g is not None and all(g[3].values())
