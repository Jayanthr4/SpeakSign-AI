# V2.7 — Temporal consistency and labeled evaluation

Uses V2.6.4 **corrected-FPS** JSONL inference results. Does not infer
signs from skipped or uncertain windows. A stable candidate requires
at least two nearby, agreeing accepted windows from the same track.
Overlapping windows are correlated, so this is a *display filter*, not
a confidence calibration or proof of signing.

```bash
python -m ml.recognition.temporal_consistency \
  --input storage/recognition/predictions_v264_corrected.jsonl \
  --fps 59.98509316770186 \
  --output storage/evaluation_v27
```

Outputs: `candidate_timeline.jsonl`, `evaluation_summary.json`.
Optional `--annotations labeled_signs.jsonl` runs one-to-one
exact-label interval matching at minimum IoU 0.5 (configurable
`--min-iou`). Each annotation must have
`{"track_id":1,"start_frame":0,"end_frame":47,"sign":"hello"}`.

**Do not compute or report accuracy** from a few cherry-picked labels:
the annotation file must exhaustively cover the evaluation region,
including non-signing intervals. Current optional evaluation treats
unannotated regions as negatives; if only part of a video is annotated,
restrict both predictions and annotations to that region first.

Tests: `python -m unittest discover -s tests -p 'test_*.py'`

Next: compare timeline candidates with ground-truth isolated-sign
clips, calibrate thresholds, and build a frontend review interface.
