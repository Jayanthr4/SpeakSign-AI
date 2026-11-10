"""Offline multi-person tracking and selected-signer crop generation.

Run from repository root:
  pip install -r ml/requirements-tracking.txt
  python -m ml.person_tracking.track_video --input /path/to/video.mp4 --output storage/tracking

Inspect the annotated preview, then re-run with --signer-id N.
This is a person-tracking baseline, NOT sign-language recognition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Local video file")
    parser.add_argument("--output", type=Path, default=Path("storage/tracking"))
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO detection checkpoint")
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--signer-id", type=int, default=None, help="Track ID from preview")
    parser.add_argument("--margin", type=float, default=0.22, help="Extra fraction around box")
    parser.add_argument("--max-frames", type=positive_int, default=None)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    if args.signer_id is not None and args.signer_id < 0:
        parser.error("--signer-id must be nonnegative")
    if not 0 <= args.margin <= 1:
        parser.error("--margin must be between 0 and 1")
    args.output.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode video: {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not 1 <= fps <= 240:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Video has invalid dimensions")

    model = YOLO(args.model)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    preview_path = args.output / "tracked_preview.mp4"
    preview = cv2.VideoWriter(str(preview_path), fourcc, fps, (width, height))
    if not preview.isOpened():
        capture.release()
        raise RuntimeError("Cannot create preview output video")

    # Crop stays at a fixed resolution so the output is usable for a
    # future landmark extraction pipeline. No other person's crop is
    # substituted when the chosen signer is temporarily missing.
    crop_path = args.output / "selected_signer.mp4"
    crop_writer = None
    if args.signer_id is not None:
        crop_writer = cv2.VideoWriter(str(crop_path), fourcc, fps, (512, 512))
        if not crop_writer.isOpened():
            preview.release()
            capture.release()
            raise RuntimeError("Cannot create signer crop video")

    manifest = []
    frames = 0
    try:
        while args.max_frames is None or frames < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            original_frame = frame.copy()
            result = model.track(
                frame, persist=True, classes=[0], tracker=args.tracker,
                conf=0.25, verbose=False,
            )[0]
            people = []
            selected_box = None
            if result.boxes is not None and result.boxes.id is not None:
                for xyxy, track_id, conf in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.id.int().cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                ):
                    box = [int(round(v)) for v in xyxy]
                    people.append({"track_id":track_id,"bbox_xyxy":box,"confidence":round(conf,4)})
                    if track_id == args.signer_id:
                        selected_box = box
                    x1,y1,x2,y2 = box
                    chosen = track_id == args.signer_id
                    color = (70,210,95) if chosen else (235,170,60)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    cv2.putText(frame,f"Person {track_id}",(x1,max(20,y1-8)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.65,color,2)
            if crop_writer is not None:
                # Black frame explicitly represents a missing signer, not
                # a substitute person's image. Manifest marks it missing.
                crop = original_frame[:1,:1].copy()
                crop = cv2.resize(crop,(512,512))
                crop[:] = 0
                if selected_box:
                    x1,y1,x2,y2 = selected_box
                    pad_x=int((x2-x1)*args.margin)
                    pad_y=int((y2-y1)*args.margin)
                    x1=max(0,x1-pad_x);y1=max(0,y1-pad_y)
                    x2=min(width,x2+pad_x);y2=min(height,y2+pad_y)
                    if x2>x1 and y2>y1:
                        crop=cv2.resize(original_frame[y1:y2,x1:x2],(512,512))
                crop_writer.write(crop)
            manifest.append({
                "frame":frames,"time_seconds":round(frames/fps,4),
                "people":people,"selected_signer_visible":selected_box is not None,
            })
            preview.write(frame)
            frames+=1
    finally:
        capture.release()
        preview.release()
        if crop_writer is not None:
            crop_writer.release()
    metadata = {
        "source":str(args.input),"fps":fps,"width":width,"height":height,
        "signer_id":args.signer_id,"frames":frames,
        "warning":"Person IDs are tracker estimates and may switch after occlusion.",
        "observations":manifest,
    }
    (args.output / "tracking_manifest.json").write_text(json.dumps(metadata,indent=2))
    print(f"Tracked {frames} frames. Preview: {preview_path}")
    if args.signer_id is not None:
        print(f"Signer crop: {crop_path}")


if __name__ == "__main__":
    main()
