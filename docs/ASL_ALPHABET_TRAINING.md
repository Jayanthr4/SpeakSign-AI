# ASL Alphabet model — training experiment

This trains a **static image classifier**, not an ASL translator. The Kaggle
dataset includes 29 classes (26 letters plus SPACE, DELETE, NOTHING).
J and Z are motion-dependent in real fingerspelling; static-image predictions
must not be presented as reliable J/Z recognition.

## 1. Install

Activate the project's existing Conda environment, then:

```bash
python -m pip install torch torchvision pillow onnx
```

## 2. Download dataset

Download the **ASL Alphabet** dataset by grassknoted from:
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Extract to `datasets/asl_alphabet/` so that the training directory is either
`datasets/asl_alphabet/asl_alphabet_train/asl_alphabet_train/A/` or
`datasets/asl_alphabet/asl_alphabet_train/A/`.

Do not commit images or trained weights; `datasets/` and `storage/` are ignored.

## 3. Train on Apple Silicon

```bash
python -m ml.recognition.train_asl_alphabet --data datasets/asl_alphabet --pretrained --epochs 12 --batch-size 32 --export-onnx
```

The script automatically uses Apple MPS when available, otherwise CUDA or CPU.
Use `--batch-size 16` if memory is tight. Pretrained weights download on first use.

Artifacts in `storage/models/asl_alphabet/`:
- `best.pt` — best validation checkpoint
- `labels.json` — ordered class names
- `metrics.json` — development validation metrics and per-class accuracy
- `model.onnx` — ONNX export when requested

**Evaluation caveat:** The script uses a seeded 85/15 random image split.
This is a development metric, NOT an independent signer-generalization score.
Before claiming recognition accuracy, test on a new person and camera setup.
Use a separate gesture dataset for thumbs-up and temporal video for waving/BYE.

**Integration status:** This training script does not yet replace the current
live heuristic recognizer. Evaluate the trained checkpoint first.
