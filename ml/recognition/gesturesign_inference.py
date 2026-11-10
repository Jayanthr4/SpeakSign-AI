"""Run GestureSign on SpeakSign temporal windows with upstream feature code.

Requires the upstream GestureSign Python package on PYTHONPATH. Only isolated
sign clips with adequate face/hand/pose coverage are eligible. A continuous
video's arbitrary sliding windows are NOT ground-truth isolated signs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

FACE=468
LEFT=468
POSE=489
RIGHT=522


def to_holistic(frames:list[dict],width:int,height:int):
    if width<=0 or height<=0:raise ValueError("Positive source width and height required")
    data=np.full((len(frames),543,3),np.nan,dtype=np.float32)
    coverage={"hands":0,"pose":0,"face":0}
    for t,frame in enumerate(frames):
        pose=frame.get("pose",[])
        face=frame.get("face",[])
        hands=frame.get("hands",[])
        if len(pose)==33:
            data[t,POSE:POSE+33]=np.asarray(pose,dtype=np.float32)
            coverage["pose"]+=1
        if len(face)>=468:
            data[t,:FACE]=np.asarray(face[:468],dtype=np.float32)
            coverage["face"]+=1
        valid=[np.asarray(h["landmarks_xy_pixels_z_relative"],dtype=np.float32)
               for h in hands if len(h.get("landmarks_xy_pixels_z_relative",[]))==21][:2]
        if valid:
            coverage["hands"]+=1
        if len(valid)==2 and len(pose)==33:
            wrists=[data[t,POSE+15,:2],data[t,POSE+16,:2]]
            straight=np.linalg.norm(valid[0][0,:2]-wrists[0])+np.linalg.norm(valid[1][0,:2]-wrists[1])
            swapped=np.linalg.norm(valid[0][0,:2]-wrists[1])+np.linalg.norm(valid[1][0,:2]-wrists[0])
            sides=[LEFT,RIGHT] if straight<=swapped else [RIGHT,LEFT]
        else:
            sides=[]
            for h in valid:
                if len(pose)==33:
                    wrist=h[0,:2]
                    side=LEFT if np.linalg.norm(wrist-data[t,POSE+15,:2])<np.linalg.norm(wrist-data[t,POSE+16,:2]) else RIGHT
                else:
                    # MediaPipe handedness assumes mirrored camera input.
                    label=next((item.get("handedness","") for item in hands
                                if len(item.get("landmarks_xy_pixels_z_relative",[]))==21
                                and np.allclose(np.asarray(item["landmarks_xy_pixels_z_relative"])[0,:2],h[0,:2])), "")
                    side=RIGHT if str(label).lower().startswith("left") else LEFT
                sides.append(side)
        for hand,side in zip(valid,sides):
            if np.isfinite(data[t,side,0]):continue
            data[t,side:side+21]=hand
    data[:,:,0]/=width
    data[:,:,1]/=height
    # Original z is relative to a cropped person, not Holistic-compatible.
    # Upstream feature version 2 only uses x and y.
    data[:,:,2]=np.nan
    return data,{k:round(v/len(frames),4) for k,v in coverage.items()}


def predict(windows:Path,model:Path,metadata:Path,labels:Path,
            width:int,height:int,fps:float,min_coverage:float,output:Path):
    from ml.recognition.inspect_gesturesign import inspect
    inspect(model,metadata)
    try:
        from gesturesign.features import sequence_features
        from gesturesign.resample import fit_length,resample
    except ImportError as exc:
        raise RuntimeError("Clone GestureSign and set PYTHONPATH to its pipeline/src (see docs)") from exc
    import onnxruntime as ort
    meta=json.loads(metadata.read_text())
    label_data=json.loads(labels.read_text())
    names={int(item["index"]):item["label"] for item in label_data["signs"]}
    session=ort.InferenceSession(str(model),providers=["CPUExecutionProvider"])
    output.parent.mkdir(parents=True,exist_ok=True)
    total=eligible=0
    with windows.open() as src,output.open("w") as dst:
        for line in src:
            row=json.loads(line)
            total+=1
            frames=row["frames"]
            holistic,coverage=to_holistic(frames,width,height)
            result={"track_id":row["track_id"],"start_frame":row["start_frame"],
                    "end_frame":row["end_frame"],"coverage":coverage}
            if min(coverage.values())<min_coverage:
                result.update(status="skipped_low_coverage",prediction=None)
            else:
                fitted=fit_length(resample(holistic,fps,float(meta["fps"])),int(meta["maxFrames"]))
                if len(fitted)<int(meta["minFrames"]):
                    result.update(status="skipped_short",prediction=None)
                else:
                    features=sequence_features(fitted,"hands_pose_lips")
                    if features.shape[1]!=460 or not np.isfinite(features).all():
                        raise ValueError("Unexpected non-finite or mismatched features")
                    logits=session.run(["logits"],{"frames":features[None]})[0][0]
                    temperature=float(meta.get("calibration",{}).get("temperature",1.0))
                    z=(logits/temperature).astype(np.float64)
                    probs=np.exp(z-z.max());probs/=probs.sum()
                    top=np.argsort(probs)[-3:][::-1]
                    best=float(probs[top[0]])
                    threshold=float(meta.get("calibration",{}).get("notSureBelow",0.29))
                    result.update(status="predicted" if best>=threshold else "uncertain",
                                  prediction=names.get(int(top[0]),str(top[0])) if best>=threshold else None,
                                  top3=[{"sign":names.get(int(i),str(i)),
                                         "probability":round(float(probs[i]),4)} for i in top])
                    eligible+=1
            dst.write(json.dumps(result)+"\n")
    print(f"Processed {total} windows; ran model on {eligible}; saved {output}")
    print("Caution: isolated-sign model predictions are not continuous ASL translation.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--windows",type=Path,default=Path("storage/sequences/temporal_windows.jsonl"))
    p.add_argument("--model",type=Path,default=Path("storage/models/gesturesign/model.onnx"))
    p.add_argument("--metadata",type=Path,default=Path("storage/models/gesturesign/model-meta.json"))
    p.add_argument("--labels",type=Path,default=Path("storage/models/gesturesign/labels.json"))
    p.add_argument("--width",type=int,required=True,help="Original source video pixel width")
    p.add_argument("--height",type=int,required=True,help="Original source video pixel height")
    p.add_argument("--fps",type=float,required=True,help="Original video FPS")
    p.add_argument("--min-coverage",type=float,default=0.7)
    p.add_argument("--output",type=Path,default=Path("storage/recognition/predictions.jsonl"))
    args=p.parse_args()
    if args.fps<=0 or not 0<=args.min_coverage<=1:p.error("Invalid fps or coverage")
    predict(args.windows,args.model,args.metadata,args.labels,args.width,args.height,
            args.fps,args.min_coverage,args.output)


if __name__=="__main__":
    main()
