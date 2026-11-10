"""Track all people and extract hand landmarks independently for each signer.

Usage:
 python -m ml.person_tracking.multisigner_landmarks --input /path/video.mp4 --output storage/multisigner
Requires: pip install -r ml/requirements-landmarks.txt
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from ultralytics import YOLO

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
               (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
               (15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]


def padded_box(box, width, height, margin):
    x1,y1,x2,y2 = map(int, box)
    px=int((x2-x1)*margin)
    py=int((y2-y1)*margin)
    return max(0,x1-px),max(0,y1-py),min(width,x2+px),min(height,y2+py)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("storage/multisigner"))
    p.add_argument("--model",default="yolo11n.pt")
    p.add_argument("--tracker",default="botsort.yaml")
    p.add_argument("--margin",type=float,default=0.3)
    p.add_argument("--max-frames",type=int,default=None)
    p.add_argument("--hand-model",type=Path,default=Path("storage/models/hand_landmarker.task"))
    args=p.parse_args()
    if not args.input.is_file():p.error(f"Video not found: {args.input}")
    if not 0<=args.margin<=1:p.error("--margin must be between 0 and 1")
    args.output.mkdir(parents=True,exist_ok=True)
    args.hand_model.parent.mkdir(parents=True,exist_ok=True)
    if not args.hand_model.exists():
        print("Downloading MediaPipe hand landmark model...")
        urllib.request.urlretrieve(MODEL_URL,str(args.hand_model))
    detector_model=YOLO(args.model)
    BaseOptions=mp.tasks.BaseOptions
    HandLandmarker=mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions=mp.tasks.vision.HandLandmarkerOptions
    RunningMode=mp.tasks.vision.RunningMode
    # Independent VIDEO-mode instances prevent landmark tracking from jumping
    # between two people. Reappearing IDs get a fresh instance.
    def new_landmarker():
        return HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(args.hand_model)),
            running_mode=RunningMode.VIDEO,num_hands=2,
            min_hand_detection_confidence=0.35,
            min_tracking_confidence=0.4))
    landmarks={}
    last_seen={}
    cap=cv2.VideoCapture(str(args.input))
    if not cap.isOpened():raise RuntimeError(f"Cannot open {args.input}")
    fps=cap.get(cv2.CAP_PROP_FPS)
    if fps<=0 or fps>240:fps=30.0
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer=cv2.VideoWriter(str(args.output/"multisigner_preview.mp4"),
                           cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
    if not writer.isOpened():raise RuntimeError("Could not create output video")
    output=args.output/"multisigner_landmarks.jsonl"
    count=0
    try:
        with output.open("w") as out:
            while args.max_frames is None or count<args.max_frames:
                ok,frame=cap.read()
                if not ok:break
                detection=detector_model.track(frame,persist=True,classes=[0],
                                               tracker=args.tracker,conf=0.25,verbose=False)[0]
                observations=[]
                boxes=detection.boxes
                if boxes is not None and boxes.id is not None:
                    for rect,tid,conf in zip(boxes.xyxy.cpu().tolist(),
                                             boxes.id.int().cpu().tolist(),
                                             boxes.conf.cpu().tolist()):
                        tid=int(tid)
                        x1,y1,x2,y2=padded_box(rect,width,height,args.margin)
                        if x2<=x1 or y2<=y1:continue
                        # Never pass the whole scene into a hand detector:
                        # each signer gets a separate spatial crop and model state.
                        crop=frame[y1:y2,x1:x2]
                        rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
                        if tid not in landmarks:
                            landmarks[tid]=new_landmarker()
                        # Time in milliseconds is monotonic across video frames.
                        timestamp=int(round(count*1000/fps))
                        result=landmarks[tid].detect_for_video(
                            mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),timestamp)
                        hands=[]
                        for i,hand in enumerate(result.hand_landmarks):
                            pts=[]
                            for lm in hand:
                                px=x1+lm.x*(x2-x1)
                                py=y1+lm.y*(y2-y1)
                                pts.append([round(px,2),round(py,2),round(lm.z,5)])
                            for a,b in CONNECTIONS:
                                cv2.line(frame,(int(pts[a][0]),int(pts[a][1])),
                                         (int(pts[b][0]),int(pts[b][1])),(60,230,80),2)
                            for x,y,_ in pts:
                                cv2.circle(frame,(int(x),int(y)),2,(20,80,255),-1)
                            handedness=(result.handedness[i][0].category_name
                                        if i<len(result.handedness) and result.handedness[i]
                                        else None)
                            hands.append({"handedness":handedness,"landmarks_xy_pixels_z_relative":pts})
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(230,160,50),2)
                        cv2.putText(frame,f"Signer {tid}: {len(hands)} hands",
                                    (x1,max(22,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,(230,160,50),2)
                        observations.append({"track_id":tid,"confidence":round(conf,4),
                                             "crop_xyxy":[x1,y1,x2,y2],"hands":hands})
                        last_seen[tid]=count
                # Evict dormant detector state; a reappearing ID is reinitialized.
                for tid in list(landmarks):
                    if count-last_seen.get(tid,count)>int(fps*3):
                        landmarks.pop(tid).close()
                        last_seen.pop(tid,None)
                out.write(json.dumps({"frame":count,"time_seconds":round(count/fps,4),
                                      "signers":observations})+"\n")
                writer.write(frame)
                count+=1
    finally:
        cap.release()
        writer.release()
        for model in landmarks.values():model.close()
    print(f"Processed {count} frames. Video: {args.output/'multisigner_preview.mp4'}")
    print(f"Landmark sequences: {output}")
    print("Note: person IDs can switch; hands may be missed in small/occluded crops.")


if __name__=="__main__":
    main()
