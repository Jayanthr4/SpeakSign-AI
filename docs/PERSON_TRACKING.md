# Multi-person signer tracking baseline (V2.3 research)

## Why not OpenPose as the default?

OpenPose can provide multi-person body/hand/face keypoints, but its native setup
is heavier on Apple Silicon. The modular baseline here uses YOLO person detection
plus BoT-SORT track IDs. A later stage will run detailed hand/face/pose
landmarkers on the selected person's crop.

## Mac setup

From the repository root (Python 3.11 recommended):

```bash
pip install -r ml/requirements-tracking.txt
python -m ml.person_tracking.track_video --input /absolute/path/to/testimony-4.mp4 --output storage/tracking
```

Open `storage/tracking/tracked_preview.mp4` and inspect the visible Person IDs.
Choose the signer's ID, then run:

```bash
python -m ml.person_tracking.track_video --input /absolute/path/to/testimony-4.mp4 --output storage/tracking --signer-id 2
```

Replace 2 with the ID you actually observe. This generates
`selected_signer.mp4` and `tracking_manifest.json`.

## Limitations

- Track IDs are estimates, not stable identities; occlusion and crossings can
  cause ID switches. Review the annotated video and JSON before using crops.
- If a signer is missing, the crop contains a black frame and the JSON
  explicitly marks the signer absent. Do not interpret it as valid landmark data.
- This is an **offline CLI baseline**, not yet wired to the React interface.
- Cropping can cut off hands when they extend beyond the person bounding box;
  increase `--margin` and inspect the output.
- The crop is stretched to 512x512 for this initial experiment; future
  landmark extraction should use aspect-preserving padding.
- The first run downloads YOLO weights; network access is required.
- Review Ultralytics licensing before commercial deployment.
