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
_waves={}
_MAX_SESSIONS=3

def infer_frame(session_id:str, encoded:bytes):
    if len(encoded)>2_000_000:raise ValueError("Frame exceeds 2MB")
    data=np.frombuffer(encoded,dtype=np.uint8)
    frame=cv2.imdecode(data,cv2.IMREAD_COLOR)
    if frame is None:raise ValueError("Invalid JPEG")
    h,w=frame.shape[:2]
    if max(w,h)>1280:raise ValueError("Frame too large")
    started=time.perf_counter()
    with _lock:
        if session_id not in _sessions:
            from ultralytics import YOLO
            import mediapipe as mp
            from pathlib import Path
            from ml.person_tracking.multisigner_landmarks import MODEL_URL
            from ml.person_tracking.fullbody_landmarks import model_file
            import urllib.request
            path=Path(__file__).resolve().parents[1]/"storage/models/hand_landmarker.task"
            path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():urllib.request.urlretrieve(MODEL_URL,str(path))
            options=mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,num_hands=2)
            detector=mp.tasks.vision.HandLandmarker.create_from_options(options)
            pose_path=model_file(path.parent,"pose")
            face_path=model_file(path.parent,"face")
            pose=mp.tasks.vision.PoseLandmarker.create_from_options(
                mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(pose_path)),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_poses=1))
            face=mp.tasks.vision.FaceLandmarker.create_from_options(
                mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(face_path)),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_faces=1))
            _sessions[session_id]=(YOLO("yolo11n.pt"),detector,pose,face,time.monotonic())
            from ml.recognition.live_motion import WaveTracker
            _waves[session_id]=WaveTracker()
        model,detector,pose_detector,face_detector,_=_sessions[session_id]
        _sessions[session_id]=(model,detector,pose_detector,face_detector,time.monotonic())
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
                mp_crop=mp.Image(image_format=mp.ImageFormat.SRGB,data=crop)
                pose_result=pose_detector.detect(mp_crop)
                face_result=face_detector.detect(mp_crop)
                face_box=(left,top,right,bottom)
                # A head-focused second pass improves small-face detection.
                # It uses the upper portion of the tracked person, not another person.
                if not face_result.face_landmarks:
                    head_bottom=min(bottom,top+max(80,int((bottom-top)*0.46)))
                    head_left=max(left,left+int((right-left)*0.10))
                    head_right=min(right,right-int((right-left)*0.10))
                    if head_bottom-top>=48 and head_right-head_left>=48:
                        head_rgb=cv2.cvtColor(frame[top:head_bottom,head_left:head_right],cv2.COLOR_BGR2RGB)
                        head_result=face_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=head_rgb))
                        if head_result.face_landmarks:
                            face_result=head_result
                            face_box=(head_left,top,head_right,head_bottom)
                def remap(points):
                    return [[round((left+p.x*(right-left))/w,4),round((top+p.y*(bottom-top))/h,4)] for p in points]
                pose_points=remap(pose_result.pose_landmarks[0]) if pose_result.pose_landmarks else []
                if face_result.face_landmarks:
                    fx1,fy1,fx2,fy2=face_box
                    face_points=[[round((fx1+p.x*(fx2-fx1))/w,4),round((fy1+p.y*(fy2-fy1))/h,4)] for p in face_result.face_landmarks[0]]
                else:
                    face_points=[]
                from ml.recognition.live_alphabet import classify,classify_gesture
                from ml.recognition.live_onnx_alphabet import available,predict
                # Use trained classifier when model files exist.
                alphabet=[predict(frame,hand) for hand in hands] if available() else [classify(hand) for hand in hands]
                gestures=[classify_gesture(hand) for hand in hands]
                if hands:
                    motion=_waves[session_id].update(int(tid),hands[0],time.monotonic())
                    if motion["gesture"]:
                        gestures.append(motion)
                people.append({"alphabet":alphabet,"gestures":gestures,"pose":pose_points,"face":face_points,"track_id":int(tid),"box":[round(x1/w,4),round(y1/h,4),round(x2/w,4),round(y2/h,4)],"hands":hands,"confidence":round(float(conf),3)})
        now=time.monotonic()
        for key,(_,detector_obj,pose_obj,face_obj,updated) in list(_sessions.items()):
            if key!=session_id and now-updated>90:
                detector_obj.close();pose_obj.close();face_obj.close();del _sessions[key];_waves.pop(key,None)
        while len(_sessions)>_MAX_SESSIONS:
            key,(_,detector_obj,pose_obj,face_obj,_)=_sessions.popitem(last=False)
            detector_obj.close();pose_obj.close();face_obj.close();_waves.pop(key,None)
        return {"people":people,"width":w,"height":h,"inference_ms":round((time.perf_counter()-started)*1000),"recognition_status":"trained_static_alphabet_candidates" if __import__("ml.recognition.live_onnx_alphabet",fromlist=["available"]).available() else "experimental_geometry_fallback","note":"Provisional tracking IDs; hand landmarks are not sign recognition"}

def close_session(session_id:str):
    with _lock:
        entry=_sessions.pop(session_id,None)
        _waves.pop(session_id,None)
        if entry:
            for detector in entry[1:4]:detector.close()
