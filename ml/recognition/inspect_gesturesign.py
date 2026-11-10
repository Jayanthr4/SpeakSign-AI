"""Validate a pretrained GestureSign ONNX model against its published metadata.

This deliberately does not produce predictions from incompatible raw landmarks.
Model files must be supplied by the user from the upstream project.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def inspect(model:Path, metadata:Path):
    meta=json.loads(metadata.read_text())
    if meta.get("featureVersion")!=2 or meta.get("featureSet")!="hands_pose_lips":
        raise ValueError("Unsupported GestureSign feature schema")
    if meta.get("featureDim")!=460:
        raise ValueError("Expected 460 features per frame")
    if not model.is_file():
        raise FileNotFoundError(model)
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install onnxruntime: python -m pip install onnxruntime") from exc
    session=ort.InferenceSession(str(model),providers=["CPUExecutionProvider"])
    inputs=[{"name":x.name,"shape":x.shape,"type":x.type} for x in session.get_inputs()]
    outputs=[{"name":x.name,"shape":x.shape,"type":x.type} for x in session.get_outputs()]
    if len(inputs)!=1 or inputs[0]["name"]!=meta["io"]["input"]:
        raise ValueError("Unexpected model input signature")
    shape=inputs[0]["shape"]
    if len(shape)!=3 or isinstance(shape[-1],int) and shape[-1]!=460:
        raise ValueError(f"Unexpected input shape: {shape}")
    expected=set(meta["io"]["outputs"])
    if not expected.issubset({x["name"] for x in outputs}):
        raise ValueError("Model outputs do not match metadata")
    return {"model":str(model),"version":meta["version"],
            "features":meta["featureSet"],"feature_dim":meta["featureDim"],
            "min_frames":meta["minFrames"],"max_frames":meta["maxFrames"],
            "inputs":inputs,"outputs":outputs,
            "ready_for_inference":False,
            "reason":"Feature adapter not implemented: requires exact upstream preprocessing, including lips and temporal features."}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--metadata",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("storage/model_checks/gesturesign.json"))
    args=p.parse_args()
    result=inspect(args.model,args.metadata)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
