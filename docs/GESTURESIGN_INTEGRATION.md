# V2.6.2 — GestureSign model compatibility gate

Upstream project: https://github.com/shashank11yadav/GestureSign

We inspected the upstream `web/public/models/model-meta.json`,
`web/src/ml/features.ts`, and `web/src/ml/assemble.ts`.
The published ONNX model uses 460-dimensional features per frame with
`hands_pose_lips`, featureVersion 2, and model inputs named `frames`.
The input features are **not** raw MediaPipe coordinates.

## Download upstream model assets yourself

```bash
mkdir -p storage/models/gesturesign
curl -L --fail -o storage/models/gesturesign/model.onnx \
  https://raw.githubusercontent.com/shashank11yadav/GestureSign/main/web/public/models/model.onnx
curl -L --fail -o storage/models/gesturesign/model-meta.json \
  https://raw.githubusercontent.com/shashank11yadav/GestureSign/main/web/public/models/model-meta.json
python -m pip install onnxruntime
python -m ml.recognition.inspect_gesturesign \
  --model storage/models/gesturesign/model.onnx \
  --metadata storage/models/gesturesign/model-meta.json
```

This validates the model's ONNX input/output signature and writes
`storage/model_checks/gesturesign.json`. It **does not run recognition**:
we must port the upstream exact normalization, hand assignment, lip feature
selection, sequence resampling, and temporal velocity computation first.
Never present arbitrary 460-dimensional arrays as valid input.

The fullbody CLI now defaults to **face off**, with `--enable-face` for
optional facial mesh extraction. The pretrained GestureSign model still
requires compatible lip features. Its accuracy cannot be assumed to transfer
to multi-signer continuous video.
