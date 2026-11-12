"""Export an already-trained SpeakSign ASL alphabet checkpoint to ONNX."""
from __future__ import annotations
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,default=Path("storage/models/asl_alphabet/best.pt"))
    p.add_argument("--output",type=Path,default=Path("storage/models/asl_alphabet/model.onnx"))
    args=p.parse_args()
    try:
        import torch
        from torchvision import models
        import onnxscript  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}. Install with: python -m pip install onnx onnxscript") from exc
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    labels=checkpoint["labels"]
    size=int(checkpoint["image_size"])
    net=models.mobilenet_v3_small(weights=None)
    net.classifier[-1]=torch.nn.Linear(net.classifier[-1].in_features,len(labels))
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(net,torch.randn(1,3,size,size),str(args.output),
        input_names=["image"],output_names=["logits"],opset_version=17,
        dynamic_axes={"image":{0:"batch"},"logits":{0:"batch"}})
    print(f"Exported {args.output} ({len(labels)} classes, input {size}x{size})")

if __name__=="__main__":
    main()
