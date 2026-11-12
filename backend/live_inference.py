"""Local experimental frame-by-frame YOLO + per-person MediaPipe hands.

Frame uploads are resized in the browser. Track IDs are provisional and
associated only within the current live session. No ASL recognition.
"""
from collections import OrderedDict
from threading import Lock
import time
import cv2
import numpy as np

_lock=Lock()
_sessions=OrderedDict()
_MAX_SESSIONS=3

def infer_frame(session_id:str, encoded:bytes):
    if len(encoded)>2_000_000:raise ValueError("Frame exceeds 2MB")
    data=np.frombuffer(encoded,dtype=np.uint8)
    frame=cv2.imdecode(data,cv2.IMREAD_COLOR)
    if frame is None:raise ValueError("Invalid JPEG")
    h,w=frame.shape[:2]
    if max(w,h)>1280:raise ValueError("Frame too large")
    with _lock:
        if session_id not in _sessions:
            from ultralytics import YOLO
            import mediapipe as mp
            from pathlib import Path
            from ml.person_tracking.multisigner_landmarks import MODEL_URL
            import urllib.request
            path=Path(__file__).resolve().parents[1]/"storage/models/hand_landmarker.task"
            path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():urllib.request.urlretrieve(MODEL_URL,str(path))
            options=mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,num_hands=2)
            detector=mp.tasks.vision.HandLandmarker.create_from_options(options)
            _sessions[session_id]=(YOLO("yolo11n.pt"),detector,time.monotonic())
        model,detector,_=_sessions[session_id]
        _sessions[session_id]=(model,detector,time.monotonic())
        result=model.track(frame,persist=True,tracker="botsort.yaml",classes=[0],conf=0.25,verbose=False)[0]
        people=[]
        boxes=result.boxes
        if boxes is not None and boxes.id is not None:
            import mediapipe as mp
            for rect,tid,conf in zip(boxes.xyxy.cpu().tolist(),boxes.id.int().cpu().tolist(),boxes.conf.cpu().tolist()):
                x1,y1,x2,y2=[int(v) for v in rect]
                pad=int((x2-x1)*0.18)
                left=max(0,x1-pad);right=min(w,x2+pad)
                top=max(0,y1-pad);bottom=min(h,y2+pad)
                if right<=left or bottom<=top:continue
                crop=cv2.cvtColor(frame[top:bottom,left:right],cv2.COLOR_BGR2RGB)
                detected=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=crop))
                hands=[]
                for hand in detected.hand_landmarks:
                    hands.append([[round((left+p.x*(right-left))/w,4),round((top+p.y*(bottom-top))/h,4)] for p in hand])
                people.append({"track_id":int(tid),"box":[round(x1/w,4),round(y1/h,4),round(x2/w,4),round(y2/h,4)],"hands":hands,"confidence":round(float(conf),3)})
        now=time.monotonic()
        for key,(_,detector_obj,updated) in list(_sessions.items()):
            if key!=session_id and now-updated>90:
                detector_obj.close();del _sessions[key]
        while len(_sessions)>_MAX_SESSIONS:
            key,(_,detector_obj,_)=_sessions.popitem(last=False)
            detector_obj.close()
        return {"people":people,"width":w,"height":h,"note":"Provisional tracking IDs; hand landmarks are not sign recognition"}

def close_session(session_id:str):
    with _lock:
        entry=_sessions.pop(session_id,None)
        if entry:entry[1].close()
