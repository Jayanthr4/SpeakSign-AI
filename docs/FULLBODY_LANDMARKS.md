# V2.5 — Multi-signer face + body pose landmark extraction

V2.5 builds on the validated V2.4 hand-tracking output. It adds independent
MediaPipe Pose and Face Landmarker models for the two most frequently tracked
people, or for two explicitly specified track IDs.

## Run

```bash
conda activate SpeakSign_conda_env
git pull origin feat/speaksign-v2-foundation
python -m pip install -r ml/requirements-landmarks.txt

# Run V2.4 first if you haven't already:
python -m ml.person_tracking.multisigner_landmarks \
  --input /absolute/path/to/testimony-4.mp4 \
  --output storage/multisigner

python -m ml.person_tracking.fullbody_landmarks \
  --input /absolute/path/to/testimony-4.mp4 \
  --landmarks storage/multisigner/multisigner_landmarks.jsonl \
  --output storage/fullbody
```

If the two most frequent IDs are not your two signers, pass
`--signer-ids 1 2` with the actual IDs visible in the V2.4 preview.

## Outputs

- `fullbody_preview.mp4` — boxes, pose points, and sparse face points
- `signer_sequences.jsonl` — synchronized frame records for each selected
  signer: hands, 33 pose landmarks, face mesh, missing-data flags
- `quality_summary.json` — selected IDs, frame counts and known limitations

Full face mesh coordinates are in JSONL but only a few face points are drawn
to avoid obscuring the signers. Hand points remain in the JSONL and the V2.4
preview; V2.5 preview does not redraw the hand skeletons.

## Validation and caveats

- **Track ID is not a guaranteed person identity**. Verify it doesn't change
  after crossings, occlusions or re-entry. Auto-selecting the two most frequent
  IDs is only a heuristic.
- Independent crops can include other people's faces or hands when people
  overlap. Face and pose results are not guaranteed to belong to the track.
- This is an **offline CLI** module, not yet connected to the React dashboard.
- Landmarks are model estimates; missing observations are represented explicitly.
- A person can be visible without signing. V2.5 does not detect signing
  activity or translate ASL.
- The source video and V2.4 JSONL must match exactly in frame order.
