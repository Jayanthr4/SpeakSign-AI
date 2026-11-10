# Multi-signer landmark extraction (V2.4 experimental)

This is an offline experiment, not ASL translation. It processes **all**
tracked people independently, rather than selecting one signer.

```bash
conda activate SpeakSign_conda_env
python -m pip install -r ml/requirements-landmarks.txt
python -m ml.person_tracking.multisigner_landmarks \
  --input /absolute/path/to/testimony-4.mp4 \
  --output storage/multisigner
```

Outputs:
- `multisigner_preview.mp4`: person boxes, signer IDs and hand skeletons
- `multisigner_landmarks.jsonl`: per-frame, per-person, per-hand keypoints
  in original-video pixel coordinates; empty hands indicate no detection

The first run downloads the MediaPipe Hand Landmarker model and YOLO weights.

**Important limitations:** MediaPipe is run independently on a padded crop for
each tracked person. If people overlap, a crop can still contain someone
else's hands. Track IDs can change after occlusion; neither detector guarantees
identity consistency. Handedness is the model's output, not a verified physical
left/right assignment. Small hands in portrait video may require higher
resolution or improved hand-region detection. This stage does not yet
extract face or pose landmarks, identify signing activity, or translate ASL.

Check both signer IDs, both hand skeletons, and whether a person's hands are
wrongly attributed to the other before moving to temporal modeling.
