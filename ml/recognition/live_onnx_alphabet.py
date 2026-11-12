"""Local ONNX ASL alphabet image classifier for tracked hand crops.

Model trained on Kaggle still images; probabilities are not calibrated.
"""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]/"storage/models/asl_alphabet"
_SESSION=None
_LABELS=None

def available()->bool:
    return (ROOT/"model.onnx").is_file() and (ROOT/"labels.json").is_file()

def _load():
    global _SESSION,_LABELS
    if _SESSION is None:
        import onnxruntime as ort
        _LABELS=json.loads((ROOT/"labels.json").read_text())
        _SESSION=ort.InferenceSession(str(ROOT/"model.onnx"),providers=["CPUExecutionProvider"])
    return _SESSION,_LABELS

def predict(frame:np.ndarray,hand:list[list[float]],threshold:float=0.85)->dict:
    out={"letter":None,"status":"uncertain","method":"mobilenetv3_onnx","confidence":0.0,"top_class":None}
    if len(hand)!=21 or not available():return out
    height,width=frame.shape[:2]
    xs=[p[0]*width for p in hand];ys=[p[1]*height for p in hand]
    x0,x1=min(xs),max(xs);y0,y1=min(ys),max(ys)
    span=max(x1-x0,y1-y0)
    if span<18:return out
    # Square crop with padding, clamped to image; keep hand centered.
    side=int(max(32,span*1.7))
    cx=(x0+x1)/2;cy=(y0+y1)/2
    left=int(round(cx-side/2));top=int(round(cy-side/2))
    right=left+side;bottom=top+side
    if left<0:right-=left;left=0
    if top<0:bottom-=top;top=0
    if right>width:left-=right-width;right=width
    if bottom>height:top-=bottom-height;bottom=height
    left=max(0,left);top=max(0,top)
    crop=frame[top:bottom,left:right]
    if crop.size==0:return out
    session,labels=_load()
    # Training: RGB, resize 224x224, ImageNet normalization.
    rgb=cv2.cvtColor(cv2.resize(crop,(224,224)),cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    mean=np.array([0.485,0.456,0.406],dtype=np.float32)
    std=np.array([0.229,0.224,0.225],dtype=np.float32)
    batch=np.transpose((rgb-mean)/std,(2,0,1))[None].astype(np.float32)
    logits=session.run(None,{session.get_inputs()[0].name:batch})[0][0]
    ex=np.exp(logits-np.max(logits));prob=ex/ex.sum()
    idx=int(prob.argmax());name=str(labels[idx]);confidence=float(prob[idx])
    out.update(top_class=name,confidence=round(confidence,4))
    # J/Z need temporal evidence, and nothing/del/space aren't alphabet letters.
    if name in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and len(name)==1 and name not in ("J","Z") and confidence>=threshold:
        out.update(letter=name,status="candidate")
    return out
