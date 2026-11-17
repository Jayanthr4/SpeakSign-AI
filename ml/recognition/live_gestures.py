"""Experimental everyday gesture candidates, not validated ASL recognition.

MediaPipe's 2D landmarks are viewpoint-dependent. Ambiguous poses abstain.
Diagnostic finger states are returned to aid webcam calibration.
"""
from __future__ import annotations
import math

def _measure(hand):
    if len(hand)!=21 or any(len(p)<2 for p in hand):return None
    def d(a,b):
        return math.hypot(hand[a][0]-hand[b][0],hand[a][1]-hand[b][1])
    palm=max(d(0,9),1e-6)
    states={}
    for name,mcp,pip,tip in (
        ("index",5,6,8),("middle",9,10,12),
        ("ring",13,14,16),("pinky",17,18,20)):
        # The earlier distance rule worked for Victory. Restore it with
        # a slightly relaxed margin instead of the overly strict angle test.
        states[name]=d(0,tip)>d(0,pip)+0.14*palm and d(tip,mcp)>0.60*palm
    return d,palm,states

def classify(hand:list[list[float]])->dict:
    result={"gesture":None,"status":"uncertain","method":"experimental_geometry","confidence":None}
    measured=_measure(hand)
    if measured is None:return result
    d,palm,s=measured
    result["diagnostics"]={"extended_fingers":s.copy()}
    thumb_dx=(hand[4][0]-hand[2][0])/palm
    thumb_dy=(hand[4][1]-hand[2][1])/palm
    result["diagnostics"]["thumb_direction"]={"dx":round(thumb_dx,2),"dy":round(thumb_dy,2)}
    # Finger folds should not be confused with a thumb pose. Thumbs
    # may be rotated relative to the webcam, so avoid strict joint angles.
    folded=not any(s.values())
    thumb_open=d(4,5)>0.70*palm and d(4,17)>0.95*palm
    result["diagnostics"]["ily_checks"]={
        "index_extended":bool(s["index"]),
        "pinky_extended":bool(s["pinky"]),
        "middle_folded":not s["middle"],
        "ring_folded":not s["ring"],
        "thumb_open":bool(thumb_open),
        "index_pinky_spread":round(d(8,20)/palm,2),
        "thumb_index_distance":round(d(4,5)/palm,2),
        "thumb_pinky_distance":round(d(4,17)/palm,2),
    }
    # A thumb can overlap the index/pinky in a 2D camera projection.
    # Use its own joints to estimate extension, independent of screen direction.
    thumb_tip_to_mcp=d(4,2)/palm
    thumb_ip_to_mcp=d(3,2)/palm
    thumb_joint_ratio=thumb_tip_to_mcp/max(thumb_ip_to_mcp,1e-6)
    thumb_tip_to_palm=d(4,9)/palm
    thumb_alt=(thumb_joint_ratio>1.55 and thumb_tip_to_mcp>0.38
               and thumb_tip_to_palm>0.50)
    result["diagnostics"]["ily_checks"].update({
        "thumb_joint_ratio":round(thumb_joint_ratio,2),
        "thumb_tip_to_mcp":round(thumb_tip_to_mcp,2),
        "thumb_tip_to_palm":round(thumb_tip_to_palm,2),
        "thumb_extended_by_joints":bool(thumb_alt),
    })
    gesture=None
    # ASL I-LOVE-YOU handshape: index and pinky extended, middle and ring
    # folded, thumb visibly separated from the palm. This is a static
    # handshape candidate, not a translated sentence.
    if (s["index"] and s["pinky"] and not s["middle"] and not s["ring"]
            and (thumb_open or thumb_alt) and d(8,20)>0.75*palm):
        gesture="ILY_HANDSHAPE"
        result["method"]="experimental_geometry_thumb_joints"
    elif folded and thumb_open and abs(thumb_dy)>0.35 and abs(thumb_dy)>abs(thumb_dx)*0.85:
        gesture="THUMBS_UP" if thumb_dy<0 else "THUMBS_DOWN"
    elif s["index"] and s["middle"] and not s["ring"] and not s["pinky"] and d(8,12)>0.48*palm:
        gesture="VICTORY"
    elif d(4,8)<0.32*palm and s["middle"] and s["ring"] and s["pinky"]:
        gesture="OK"
    if gesture:result.update(gesture=gesture,status="candidate")
    return result

def is_open_palm(hand:list[list[float]])->bool:
    measured=_measure(hand)
    return measured is not None and all(measured[2].values())
