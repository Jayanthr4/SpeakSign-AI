"""Train an experimental ASL alphabet image classifier.

Dataset: Kaggle grassknoted/asl-alphabet, unpacked under datasets/asl_alphabet.
A random image split is for development only; final evaluation needs new signers.
"""
from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

def find_root(root:Path)->Path:
    """Find the actual training class directory inside a Kaggle extraction."""
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist: {root.resolve()}\\n"
            "Download and extract the Kaggle ASL Alphabet dataset first. "
            "Use: find datasets -maxdepth 6 -type d | head -60"
        )
    candidates=[root]
    if root.is_dir():
        candidates.extend(p for p in root.rglob("*") if p.is_dir() and len(p.relative_to(root).parts)<=5)
    for path in candidates:
        children=[p for p in path.iterdir() if p.is_dir()]
        classes=[p for p in children if any(p.glob("*.jpg")) or any(p.glob("*.jpeg")) or any(p.glob("*.png"))]
        if len(classes)>=20 and any(p.name.upper()=="A" for p in classes):
            print(f"Discovered training classes: {path.resolve()}",flush=True)
            return path
    raise FileNotFoundError(
        f"No ASL Alphabet training class folders found under {root.resolve()}.\\n"
        "Expected 20+ class folders (A, B, C, ...) containing JPG/PNG images.\\n"
        "Inspect extraction with: find datasets -maxdepth 6 -type d | head -60"
    )

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=Path("datasets/asl_alphabet"))
    p.add_argument("--output",type=Path,default=Path("storage/models/asl_alphabet"))
    p.add_argument("--epochs",type=int,default=12)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--image-size",type=int,default=224)
    p.add_argument("--lr",type=float,default=0.0003)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--workers",type=int,default=0)
    p.add_argument("--pretrained",action="store_true",help="Download ImageNet pretrained weights")
    p.add_argument("--export-onnx",action="store_true")
    args=p.parse_args()
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader,Dataset
    from torchvision import models,transforms
    if args.epochs<1 or args.batch_size<1: p.error("epochs and batch-size must be positive")
    random.seed(args.seed);torch.manual_seed(args.seed)
    root=find_root(args.data)
    labels=sorted(d.name for d in root.iterdir() if d.is_dir() and any(d.glob("*.jpg")) or any(d.glob("*.jpeg")) or any(d.glob("*.png")))
    if not labels:raise ValueError("No labeled JPG images found")
    label_to_id={name:i for i,name in enumerate(labels)}
    train_rows=[];val_rows=[]
    rng=random.Random(args.seed)
    for label in labels:
        paths=sorted(path for path in (root/label).iterdir() if path.suffix.lower() in {".jpg",".jpeg",".png"})
        rng.shuffle(paths)
        if len(paths)<5:raise ValueError(f"Not enough examples for {label}")
        n_val=max(1,round(len(paths)*0.15))
        val_rows.extend((path,label_to_id[label]) for path in paths[:n_val])
        train_rows.extend((path,label_to_id[label]) for path in paths[n_val:])
    normalize=transforms.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))
    train_transform=transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.15,contrast=0.15),
        transforms.ToTensor(),normalize])
    eval_transform=transforms.Compose([transforms.Resize((args.image_size,args.image_size)),transforms.ToTensor(),normalize])
    class Images(Dataset):
        def __init__(self,rows,transform):self.rows=rows;self.transform=transform
        def __len__(self):return len(self.rows)
        def __getitem__(self,index):
            path,label=self.rows[index]
            with Image.open(path) as image:
                return self.transform(image.convert("RGB")),label
    train=DataLoader(Images(train_rows,train_transform),batch_size=args.batch_size,shuffle=True,num_workers=args.workers)
    val=DataLoader(Images(val_rows,eval_transform),batch_size=args.batch_size,num_workers=args.workers)
    device=torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    weights=models.MobileNet_V3_Small_Weights.DEFAULT if args.pretrained else None
    net=models.mobilenet_v3_small(weights=weights)
    net.classifier[-1]=torch.nn.Linear(net.classifier[-1].in_features,len(labels))
    net.to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=args.lr)
    criterion=torch.nn.CrossEntropyLoss()
    args.output.mkdir(parents=True,exist_ok=True)
    best=-1.0
    for epoch in range(1,args.epochs+1):
        net.train();running=0;seen=0
        for images,targets in train:
            images=images.to(device);targets=targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(net(images),targets)
            loss.backward();optimizer.step()
            running+=float(loss.item())*len(targets);seen+=len(targets)
        net.eval();correct=total=0
        class_total=defaultdict(int);class_correct=defaultdict(int)
        with torch.inference_mode():
            for images,targets in val:
                logits=net(images.to(device))
                predictions=logits.argmax(dim=1).cpu()
                for target,prediction in zip(targets.tolist(),predictions.tolist()):
                    total+=1;correct+=int(target==prediction)
                    class_total[labels[target]]+=1
                    class_correct[labels[target]]+=int(target==prediction)
        accuracy=correct/max(total,1)
        print(f"epoch {epoch}/{args.epochs} train_loss={running/max(seen,1):.4f} val_accuracy={accuracy:.4f}",flush=True)
        if accuracy>best:
            best=accuracy
            torch.save({"state_dict":net.state_dict(),"labels":labels,"image_size":args.image_size},args.output/"best.pt")
            (args.output/"metrics.json").write_text(json.dumps({
                "validation_accuracy":accuracy,"epoch":epoch,"train_images":len(train_rows),
                "validation_images":len(val_rows),"device":str(device),
                "per_class_accuracy":{k:class_correct[k]/v for k,v in class_total.items()},
                "warning":"Random image validation is not signer-independent; collect an external test set."
            },indent=2))
    (args.output/"labels.json").write_text(json.dumps(labels,indent=2))
    if args.export_onnx:
        checkpoint=torch.load(args.output/"best.pt",map_location="cpu",weights_only=True)
        net.load_state_dict(checkpoint["state_dict"]);net.cpu().eval()
        torch.onnx.export(net,torch.randn(1,3,args.image_size,args.image_size),
            str(args.output/"model.onnx"),input_names=["image"],output_names=["logits"],
            dynamic_axes={"image":{0:"batch"},"logits":{0:"batch"}},opset_version=17)
        print("Exported",args.output/"model.onnx")
    print("Best validation accuracy:",round(best,4),"output:",args.output)

if __name__=="__main__":
    main()
