# V2.6.3 — Experimental GestureSign ONNX inference

This integrates **upstream GestureSign's exact Python feature extraction**
(`gesturesign.features.sequence_features` and `gesturesign.resample`).
Do not replace it with arbitrary 460-column input.

## Setup

```bash
conda activate SpeakSign_conda_env
python -m pip install numpy onnxruntime
mkdir -p storage/models/gesturesign
curl -L --fail -o storage/models/gesturesign/model.onnx https://raw.githubusercontent.com/shashank11yadav/GestureSign/main/web/public/models/model.onnx
curl -L --fail -o storage/models/gesturesign/model-meta.json https://raw.githubusercontent.com/shashank11yadav/GestureSign/main/web/public/models/model-meta.json
curl -L --fail -o storage/models/gesturesign/labels.json https://raw.githubusercontent.com/shashank11yadav/GestureSign/main/web/public/models/labels.json
git clone https://github.com/shashank11yadav/GestureSign.git /tmp/GestureSign
export PYTHONPATH="$PWD:/tmp/GestureSign/pipeline/src:${PYTHONPATH:-}"
```

Run using source video dimensions and FPS (e.g. portrait 720 x 1280,
and **replace 30 with your video's actual FPS**):

```bash
python -m ml.recognition.gesturesign_inference \
  --windows storage/sequences/temporal_windows.jsonl \
  --width 720 --height 1280 --fps 30 \
  --min-coverage 0.7 \
  --output storage/recognition/predictions.jsonl
```

Run tests: `python -m unittest discover -s tests -p 'test_*.py'`

## Important limitations

- The upstream model is for **isolated ASL signs**, not continuous video
  translation. Predictions from sliding windows are experimental.
- The adapter maps fullbody V2.5 pixel coordinates to 543-point Holistic
  coordinates, assigns hands by pose wrists, and uses the upstream exact
  sequence normalization/resampling. It does **not** reproduce the original
  MediaPipe Holistic detector; domain shift can reduce accuracy.
- Face extraction must be enabled upstream (`--enable-face`) to provide lip
  features. The default coverage gate skips windows with insufficient face,
  hands or pose data. Your earlier video has low face coverage, so many or
  all windows may be skipped. This is intentional.
- The adapter requires original source-video width, height and FPS. Supplying
  incorrect values invalidates feature normalization.
- Track ID switches, background people and person-crop overlap remain
  unresolved. A high confidence score does not prove a valid ASL sign.
- Upstream license/attribution must be reviewed before redistributing the
  model. The upstream source is loaded from a separate checkout rather than
  vendored into this repository.
