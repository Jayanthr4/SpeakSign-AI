# SpeakSign AI V2.6 — generalized sequence QA

This milestone adds **dynamic multi-signer sequence processing** and quality
reports, without requiring exactly two signers. It consumes the V2.5 JSONL
export and can handle any number of track IDs in that input.

## Run on your existing V2.5 output

```bash
python -m ml.person_tracking.sequence_engine \
  --input storage/fullbody/signer_sequences.jsonl \
  --output storage/sequences \
  --window 32 --stride 16 --min-visibility 0.6
```

Outputs:

- `sequence_quality.json` — per-track visibility, hands/pose/face detection
  coverage and a rough hand-motion statistic.
- `temporal_windows.jsonl` — synchronized sliding windows for each track,
  preserving missing data, suitable for downstream feature engineering.

Run regression tests:

```bash
python -m unittest discover -s tests -p 'test_sequence_engine.py'
```

## What is and is not generalized

The **sequence engine** supports any number of signers. The upstream V2.5
fullbody extraction still defaults to the two most frequent IDs; to include
more signers, use `--signer-ids` with all desired IDs. V2.4 tracks all
detected people. This separation is deliberate: running heavy face/pose
models on every background person can be expensive.

This is not yet automatic signer activity classification, stable re-ID after
occlusion, ASL recognition or translation. Motion statistics must not be
presented as a sign-language detector. Tracker ID changes can fragment a
person's sequence, and crop overlap can misattribute landmarks.

For generalization, test with multiple recordings (1, 2, and 3+ signers,
different lighting and background people). Measure identity switches,
hand/pose/face detection rates, and missing-data coverage before training.
