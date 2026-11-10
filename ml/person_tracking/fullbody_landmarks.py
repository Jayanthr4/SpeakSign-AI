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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True,help="Same source video used in V2.4")
    p.add_argument("--landmarks",type=Path,default=Path("storage/multisigner/multisigner_landmarks.jsonl"))
    p.add_argument("--output",type=Path,default=Path("storage/fullbody"))
    p.add_argument("--signer-ids",type=int,nargs="+",default=None,
                   help="Track IDs for the two main signers; otherwise use two most frequent IDs")
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
    face_path=model_file(args.model_dir,"face")
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
                                    "face":make_detector("face",face_path)}
                                last_timestamp[tid]=-1
                            stamp=max(timestamp,last_timestamp[tid]+1)
                            last_timestamp[tid]=stamp
                            pose=detectors[tid]["pose"].detect_for_video(mpimage,stamp)
                            face=detectors[tid]["face"].detect_for_video(mpimage,stamp)
                            if pose.pose_landmarks:
                                item["pose"]=landmarks_to_frame(pose.pose_landmarks[0],[x1,y1,x2,y2])
                                draw_points(frame,item["pose"],range(33),(255,190,40))
                            if face.face_landmarks:
                                item["face"]=landmarks_to_frame(face.face_landmarks[0],[x1,y1,x2,y2])
                                # Sparse face points for readable preview; full mesh exported.
                                draw_points(frame,item["face"],[1,4,33,61,133,168,263,291,362],(255,90,190))
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
        for pair in detectors.values():
            for detector in pair.values():detector.close()
    summary={
        "source":str(args.input),"frames":frames,"fps":fps,
        "selected_track_ids":selected,
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
