"""V2.6: general-purpose, multi-signer sequence QA and window export.

Consumes V2.5 signer_sequences.jsonl, supports any number of tracked people,
and never interprets movement as ASL translation or proven signing activity.
Standard-library only. Missing landmarks stay missing (no zero-fill).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def distance(a,b):
    return math.hypot(a[0]-b[0],a[1]-b[1])


def hand_centers(hands):
    centers=[]
    for hand in hands:
        pts=hand.get("landmarks_xy_pixels_z_relative",[])
        if len(pts)==21:
            centers.append([sum(p[0] for p in pts)/21,
                            sum(p[1] for p in pts)/21])
    return centers


def process(source:Path,output:Path,window:int,stride:int,min_visibility:float,
            signer_ids:list[int]|None=None):
    if window<=0 or stride<=0:raise ValueError("window and stride must be positive")
    if not 0<=min_visibility<=1:raise ValueError("min_visibility must be in [0,1]")
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    with source.open() as f:
        for line_number,line in enumerate(f,1):
            if not line.strip():continue
            row=json.loads(line)
            if rows and row["frame"]<=rows[-1]["frame"]:
                raise ValueError(f"Frames must be strictly increasing (line {line_number})")
            rows.append(row)
    if not rows:raise ValueError("No input frames")
    tracks=defaultdict(dict)
    for row in rows:
        for signer in row.get("signers",[]):
            tid=int(signer["track_id"])
            if signer_ids is None or tid in signer_ids:
                tracks[tid][row["frame"]]=signer
    if signer_ids is not None:
        for tid in signer_ids:tracks.setdefault(tid,{})
    summary={"source":str(source),"total_frames":len(rows),"window_size":window,
             "stride":stride,"signers":{},"warnings":[
                 "Tracking IDs can switch; a track ID is not a verified identity.",
                 "Motion is an unvalidated activity proxy, not signing recognition.",
                 "Coordinates are original-video pixels, not camera-normalized.",
                 "Windows with missing landmarks preserve null/missing entries.",
             ]}
    windows_path=output/"temporal_windows.jsonl"
    with windows_path.open("w") as out:
        for tid,by_frame in sorted(tracks.items()):
            sequence=[]
            visible=hands_ok=pose_ok=face_ok=0
            movements=[]
            previous=None
            for row in rows:
                signer=by_frame.get(row["frame"])
                present=bool(signer and signer.get("visible",True))
                hands=signer.get("hands",[]) if present else []
                pose=signer.get("pose",[]) if present else []
                face=signer.get("face",[]) if present else []
                centers=hand_centers(hands)
                visible+=int(present)
                hands_ok+=int(bool(centers))
                pose_ok+=int(bool(pose))
                face_ok+=int(bool(face))
                motion=None
                if previous is not None and centers and len(centers)==len(previous):
                    # Nearest-center association, only a rough proxy. The
                    # assignment can still swap when hands cross.
                    motion=sum(min(distance(c,p) for p in previous) for c in centers)/len(centers)
                    movements.append(motion)
                previous=centers if present and centers else None
                sequence.append({
                    "frame":row["frame"],"time_seconds":row["time_seconds"],
                    "visible":present,"hands":hands,"pose":pose,"face":face,
                    "quality":{"hand_count":len(centers),"pose_present":bool(pose),
                               "face_present":bool(face),"hand_motion_px":motion},
                })
            n=len(rows)
            emitted=0
            for start in range(0,n-window+1,stride):
                clip=sequence[start:start+window]
                coverage=sum(x["visible"] for x in clip)/window
                if coverage<min_visibility:continue
                out.write(json.dumps({
                    "track_id":tid,"start_frame":clip[0]["frame"],
                    "end_frame":clip[-1]["frame"],
                    "start_seconds":clip[0]["time_seconds"],
                    "end_seconds":clip[-1]["time_seconds"],
                    "visibility_ratio":round(coverage,4),
                    "frames":clip,
                })+"\n")
                emitted+=1
            summary["signers"][str(tid)]={
                "visible_frames":visible,
                "visibility_ratio":round(visible/n,4),
                "frames_with_hands":hands_ok,
                "frames_with_pose":pose_ok,
                "frames_with_face":face_ok,
                "hand_detection_rate_when_visible":round(hands_ok/visible,4) if visible else None,
                "pose_detection_rate_when_visible":round(pose_ok/visible,4) if visible else None,
                "face_detection_rate_when_visible":round(face_ok/visible,4) if visible else None,
                "mean_hand_motion_px":round(sum(movements)/len(movements),3) if movements else None,
                "exported_windows":emitted,
            }
    (output/"sequence_quality.json").write_text(json.dumps(summary,indent=2))
    print(f"Tracks: {len(tracks)}; frames: {len(rows)}")
    print(f"Quality: {output/'sequence_quality.json'}")
    print(f"Temporal windows: {windows_path}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=Path("storage/fullbody/signer_sequences.jsonl"))
    p.add_argument("--output",type=Path,default=Path("storage/sequences"))
    p.add_argument("--window",type=int,default=32)
    p.add_argument("--stride",type=int,default=16)
    p.add_argument("--min-visibility",type=float,default=0.6)
    p.add_argument("--signer-ids",type=int,nargs="+",default=None)
    args=p.parse_args()
    if not args.input.is_file():p.error(f"Input not found: {args.input}")
    process(args.input,args.output,args.window,args.stride,args.min_visibility,args.signer_ids)


if __name__=="__main__":
    main()
