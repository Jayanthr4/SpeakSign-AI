# V2.6.4 — Pose-guided head crop fallback

The existing full-person face detector remains the primary extraction path.
Use `--enable-face --head-crop-fallback` to try an additional head-focused
crop **only when the first detector fails** and pose head landmarks exist.

The fallback:
- Estimates a conservative head box using nose/eyes/ears and shoulder width.
- Clips that box to the tracked person and original image boundaries.
- Runs a separate MediaPipe FaceLandmarker in IMAGE mode.
- Maps the 478/468 detected points back to original image pixel coordinates.
- Stores `quality.face_source` and `quality.face_geometry`.
- Counts attempts/successes in `quality_summary.json`.

This is **not** a specialized side-profile model. Geometry checks only
verify finite, in-frame lip landmarks; they do not establish whether
occluded lips are accurately estimated. No hidden landmarks are fabricated.

Run:

```bash
python -m ml.person_tracking.fullbody_landmarks \
  --input /absolute/path/to/video.mp4 \
  --landmarks storage/multisigner/multisigner_landmarks.jsonl \
  --output storage/fullbody_v264 \
  --enable-face --head-crop-fallback

python -m ml.person_tracking.sequence_engine \
  --input storage/fullbody_v264/signer_sequences.jsonl \
  --output storage/sequences_v264 \
  --window 32 --stride 16 --min-visibility 0.6
```

Then use the GestureSign inference CLI with
`--windows storage/sequences_v264/temporal_windows.jsonl` and the
actual video width, height and FPS.

Compare `head_crop_successes` and the resulting inference eligibility
against V2.6.3. A higher detection rate does not demonstrate higher
recognition accuracy; evaluate with labeled isolated-sign videos.
