"""V2.5: augment V2.4 tracked hand data with face and pose landmarks.

Run after multisigner_landmarks. Track IDs are provisional, not biometric identity.
This is feature extraction, not ASL recognition or translation.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp

BASE="https://storage.googleapis.com/mediapipe-models/"
MODELS={
 "pose":BASE+"pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
 "face":BASE+"face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}


def model_file(folder:Path,name:str)->Path:
    path=folder/(name+"_landmarker.task")
    if not path.exists():
        folder.mkdir(parents=True,exist_ok=True)
        print(f"Downloading {name} model...")
        urllib.request.urlretrieve(MODELS[name],str(path))
    return path


def landmarks_to_frame(points,box):
    x1,y1,x2,y2=box
    return [[round(x1+p.x*(x2-x1),2),
             round(y1+p.y*(y2-y1),2),round(p.z,6)] for p in points]


def make_detector(kind,model_path):
    options=dict(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
    )
    if kind=="pose":
        return mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(num_poses=1,**options))
    return mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(num_faces=1,**options))


def draw_points(frame,points,indices,color):
    for idx in indices:
        if idx<len(points):
            x,y,_=points[idx]
            cv2.circle(frame,(int(x),int(y)),2,color,-1)


HAND_CONNECTIONS=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
POSE_CONNECTIONS=[(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),(27,29),(29,31),(28,30),(30,32),(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(9,10)]
FACE_CONNECTIONS=[(10,338),(338,297),(297,332),(332,284),(284,251),(251,389),(389,356),(356,454),(454,323),(323,361),(361,288),(288,397),(397,365),(365,379),(379,378),(378,400),(400,377),(377,152),(152,148),(148,176),(176,149),(149,150),(150,136),(136,172),(172,58),(58,132),(132,93),(93,234),(234,127),(127,162),(162,21),(21,54),(54,103),(103,67),(67,109),(109,10),(33,133),(362,263),(61,40),(40,37),(37,0),(0,267),(267,270),(270,291),(61,146),(146,91),(91,181),(181,84),(84,17),(17,314),(314,405),(405,321),(321,375),(375,291)]

def draw_connections(frame,points,edges,color,thickness=2):
    height,width=frame.shape[:2]
    for a,b in edges:
        if a>=len(points) or b>=len(points):
            continue
        ax,ay=points[a][:2]
        bx,by=points[b][:2]
        if all(0<=x<width and 0<=y<height for x,y in ((ax,ay),(bx,by))):
            cv2.line(frame,(int(ax),int(ay)),(int(bx),int(by)),color,thickness)
    for x,y,*_ in points:
        if 0<=x<width and 0<=y<height:
            cv2.circle(frame,(int(x),int(y)),2,color,-1)


def head_crop_box(pose_points, person_box, width, height, padding=0.45):
    """Conservative head ROI in original pixels, from visible pose landmarks.

    Pose points already use original-frame pixels. The ROI stays within the
    tracked person's bounding box to reduce cross-person face contamination.
    Returns None when there is insufficient evidence of head location.
    """
    if len(pose_points)<13:return None
    import math
    x1,y1,x2,y2=map(float,person_box)
    head=[pose_points[i] for i in (0,1,2,3,4,5,6,7,8) if i<len(pose_points)]
    head=[p for p in head if len(p)>=2 and all(math.isfinite(v) for v in p[:2])
          and x1<=p[0]<=x2 and y1<=p[1]<=y2]
    if len(head)<2:return None
    xs=[p[0] for p in head];ys=[p[1] for p in head]
    shoulders=[pose_points[i] for i in (11,12)]
    shoulder_width=abs(shoulders[0][0]-shoulders[1][0])
    span=max(max(xs)-min(xs),max(ys)-min(ys),shoulder_width*0.45,20)
    cx=(min(xs)+max(xs))/2
    cy=(min(ys)+max(ys))/2
    radius=span*(0.5+padding)
    a=max(0,int(max(x1,cx-radius)));b=max(0,int(max(y1,cy-radius)))
    c=min(width,int(min(x2,cx+radius)));d=min(height,int(min(y2,cy+radius)))
    if c-a<24 or d-b<24:return None
    return [a,b,c,d]


def face_quality(points, frame_width, frame_height):
    """Geometry checks, not a calibrated visibility/occlusion estimate."""
    import math
    if len(points)<468:return {"usable":False,"reason":"insufficient_points"}
    lips=(0,13,14,17,61,291)
    values=[points[i] for i in lips]
    if not all(len(p)>=2 and all(math.isfinite(v) for v in p[:2]) for p in values):
        return {"usable":False,"reason":"invalid_lips"}
    inside=sum(0<=p[0]<frame_width and 0<=p[1]<frame_height for p in values)
    return {"usable":inside==len(values),"reason":"geometry_only" if inside==len(values) else "lips_outside_frame"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True,help="Same source video used in V2.4")
    p.add_argument("--landmarks",type=Path,default=Path("storage/multisigner/multisigner_landmarks.jsonl"))
    p.add_argument("--output",type=Path,default=Path("storage/fullbody"))
    p.add_argument("--signer-ids",type=int,nargs="+",default=None,
                   help="Track IDs for the two main signers; otherwise use two most frequent IDs")
    p.add_argument("--enable-face",action="store_true",help="Enable optional face mesh extraction (off by default)")
    p.add_argument("--head-crop-fallback",action="store_true",help="Try pose-guided head ROI if face missing on person crop")
    p.add_argument("--max-frames",type=int,default=None)
    p.add_argument("--model-dir",type=Path,default=Path("storage/models"))
    args=p.parse_args()
    if not args.input.is_file():p.error(f"Video not found: {args.input}")
    if not args.landmarks.is_file():p.error(f"V2.4 landmarks not found: {args.landmarks}")
    if args.signer_ids is not None and len(set(args.signer_ids))!=len(args.signer_ids):
        p.error("Signer IDs must be distinct")
    args.output.mkdir(parents=True,exist_ok=True)
    # Count how often a track appears, not its box size in individual frames.
    with args.landmarks.open() as f:
        frequency=Counter(
            person["track_id"] for line in f
            for person in json.loads(line).get("signers",[])
        )
    selected=(args.signer_ids if args.signer_ids is not None
              else [tid for tid,_ in frequency.most_common(2)])
    if not selected:p.error("No tracked people found in the V2.4 JSONL")
    print(f"Processing track IDs: {selected} (review identity continuity manually)")
    pose_path=model_file(args.model_dir,"pose")
    face_path=model_file(args.model_dir,"face") if args.enable_face else None
    cap=cv2.VideoCapture(str(args.input))
    if not cap.isOpened():raise RuntimeError(f"Cannot open video: {args.input}")
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not 1<=fps<=240:fps=30.
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer=cv2.VideoWriter(str(args.output/"fullbody_preview.mp4"),
                           cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot write preview video")
    detectors={}
    fallback_detector=(mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(face_path)),num_faces=1,
            running_mode=mp.tasks.vision.RunningMode.IMAGE))
        if args.enable_face and args.head_crop_fallback else None)
    fallback_attempts=Counter()
    fallback_success=Counter()
    last_timestamp={}
    frames=0
    missing=Counter()
    detected=Counter()
    try:
        with args.landmarks.open() as f, (args.output/"signer_sequences.jsonl").open("w") as output:
            for line in f:
                if args.max_frames is not None and frames>=args.max_frames:break
                ok,frame=cap.read()
                if not ok:break
                record=json.loads(line)
                if record["frame"]!=frames:
                    raise ValueError(f"Frame mismatch at {frames}: got {record['frame']}")
                by_id={p["track_id"]:p for p in record.get("signers",[])}
                timestamp=max(0,int(round(frames*1000/fps)))
                entries=[]
                for tid in selected:
                    person=by_id.get(tid)
                    item={"track_id":tid,"visible":person is not None,
                          "hands":person.get("hands",[]) if person else [],
                          "pose":[],"face":[],"quality":{"hands_detected":0,
                          "pose_detected":False,"face_detected":False,
                          "missing":person is None}}
                    if person:
                        x1,y1,x2,y2=map(int,person["crop_xyxy"])
                        x1=max(0,x1);y1=max(0,y1)
                        x2=min(width,x2);y2=min(height,y2)
                        if x2>x1 and y2>y1:
                            crop=frame[y1:y2,x1:x2]
                            rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
                            mpimage=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                            if tid not in detectors:
                                detectors[tid]={
                                    "pose":make_detector("pose",pose_path),
                                    **({"face":make_detector("face",face_path)} if args.enable_face else {})}
                                last_timestamp[tid]=-1
                            stamp=max(timestamp,last_timestamp[tid]+1)
                            last_timestamp[tid]=stamp
                            pose=detectors[tid]["pose"].detect_for_video(mpimage,stamp)
                            face=detectors[tid]["face"].detect_for_video(mpimage,stamp) if args.enable_face else None
                            if pose.pose_landmarks:
                                item["pose"]=landmarks_to_frame(pose.pose_landmarks[0],[x1,y1,x2,y2])
                                draw_connections(frame,item["pose"],POSE_CONNECTIONS,(255,190,40))
                            face_source="person_crop" if face is not None and face.face_landmarks else "none"
                            if face is not None and face.face_landmarks:
                                item["face"]=landmarks_to_frame(face.face_landmarks[0],[x1,y1,x2,y2])
                            elif fallback_detector is not None and item["pose"]:
                                head_box=head_crop_box(item["pose"],[x1,y1,x2,y2],width,height)
                                if head_box:
                                    fallback_attempts[tid]+=1
                                    hx1,hy1,hx2,hy2=head_box
                                    head_rgb=cv2.cvtColor(frame[hy1:hy2,hx1:hx2],cv2.COLOR_BGR2RGB)
                                    head_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=head_rgb)
                                    head_result=fallback_detector.detect(head_image)
                                    if head_result.face_landmarks:
                                        candidate=landmarks_to_frame(head_result.face_landmarks[0],head_box)
                                        if face_quality(candidate,width,height)["usable"]:
                                            item["face"]=candidate
                                            face_source="pose_head_crop"
                                            fallback_success[tid]+=1
                            item["quality"]["face_source"]=face_source
                            item["quality"]["face_geometry"]=face_quality(item["face"],width,height)
                            if item["face"]:
                                # Sparse face points for readable preview; full mesh exported.
                                draw_connections(frame,item["face"],FACE_CONNECTIONS,(255,90,190),1)
                            for hand in item["hands"]:
                                pts=hand.get("landmarks_xy_pixels_z_relative",[])
                                if len(pts)==21:
                                    draw_connections(frame,pts,HAND_CONNECTIONS,(65,235,85),2)
                            item["quality"]={
                                "hands_detected":len(item["hands"]),
                                "pose_detected":bool(item["pose"]),
                                "face_detected":bool(item["face"]),
                                "missing":False,
                            }
                            detected[tid]+=1
                            cv2.rectangle(frame,(x1,y1),(x2,y2),(60,210,110),2)
                            cv2.putText(frame,f"Signer {tid}: H{len(item['hands'])} P{int(bool(item['pose']))} F{int(bool(item['face']))}",
                                        (x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(60,210,110),2)
                    else:
                        missing[tid]+=1
                    entries.append(item)
                output.write(json.dumps({"frame":frames,"time_seconds":round(frames/fps,4),
                                         "signers":entries})+"\n")
                writer.write(frame)
                frames+=1
    finally:
        cap.release()
        writer.release()
        if fallback_detector is not None:fallback_detector.close()
        for pair in detectors.values():
            for detector in pair.values():detector.close()
    summary={
        "source":str(args.input),"frames":frames,"fps":fps,
        "selected_track_ids":selected,"face_enabled":args.enable_face,
        "head_crop_fallback_enabled":bool(fallback_detector is not None),
        "head_crop_attempts":dict(fallback_attempts),
        "head_crop_successes":dict(fallback_success),
        "present_frames":dict(detected),"missing_frames":dict(missing),
        "limitations":[
            "Track IDs may switch across occlusions; this is not identity verification.",
            "Person crops may contain other people's hands, faces or bodies.",
            "No temporal smoothing, signing activity detection or ASL translation.",
            "Coordinates are image pixels; z is model-relative, not metric depth.",
        ],
    }
    (args.output/"quality_summary.json").write_text(json.dumps(summary,indent=2))
    print(f"Preview: {args.output/'fullbody_preview.mp4'}")
    print(f"Sequences: {args.output/'signer_sequences.jsonl'}")
    print(f"Quality: {args.output/'quality_summary.json'}")


if __name__=="__main__":
    main()
