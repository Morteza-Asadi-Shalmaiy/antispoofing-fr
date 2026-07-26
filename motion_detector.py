#!/usr/bin/env python3
"""
Motion Detector
===============

Detects motion in a video (file or webcam) using frame-differencing
against a running background average, draws bounding boxes around
moving regions, and optionally saves the annotated result to disk.

This is a cleaned-up, CLI-friendly version of an original Colab
prototype (see notebooks/original_colab.ipynb). Key improvements over
the prototype:

  - No dependency on Colab-only helpers (cv2_imshow, files.upload).
  - Uses a running weighted average as the background model instead of
    a single first frame, which is more robust to gradual lighting
    changes and camera noise.
  - Reports progress on a single updating line instead of dumping an
    image per frame.
  - Always writes an annotated output video; on-screen display is
    optional via --display.
  - Fails loudly (clear error) instead of silently doing nothing when
    the input source can't be opened.

Usage:
    python motion_detector.py --source assets/test-video.mp4
    python motion_detector.py --source 0 --display
    python motion_detector.py --source clip.mp4 --output result.mp4 --min-area 800
"""

import argparse
import datetime
import os
import sys
from typing import Optional, Tuple

import cv2
import imutils
import numpy as np


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect motion in a video file or webcam stream."
    )
    parser.add_argument(
        "--source",
        default="assets/test-video.mp4",
        help="Path to a video file, or an integer webcam index (e.g. 0). "
        "Default: assets/test-video.mp4",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the annotated output video. Defaults to "
        "'result-<source filename>.mp4' in the current directory.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=500,
        help="Minimum contour area (in pixels) to count as motion. Default: 500",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=500,
        help="Width (px) frames are resized to before processing. Default: 500",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show a live window while processing (requires a GUI environment).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving the annotated output video (progress is still printed).",
    )
    return parser.parse_args(argv)


def resolve_source(source: str):
    """Return an int (webcam index) or the original string (file path)."""
    try:
        return int(source)
    except ValueError:
        return source


def open_capture(source) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(
            f"Could not open video source: {source!r}. "
            "Check the path is correct or that the webcam index is valid."
        )
    return cap


def detect_motion(
    background: np.ndarray,
    frame: np.ndarray,
    min_area: int,
) -> Tuple[np.ndarray, str, list]:
    """
    Compare a single (already resized, grayscale, blurred) frame against
    a background frame and annotate `frame` in place with bounding boxes.

    Returns (annotated_frame, status_text, boxes) where boxes is a list
    of (x, y, w, h) tuples for each detected motion region.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    frame_delta = cv2.absdiff(background, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    status = "No motion"
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        boxes.append((x, y, w, h))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        status = "Motion detected"

    return frame, status, boxes


def annotate_overlay(frame: np.ndarray, status: str) -> np.ndarray:
    cv2.putText(
        frame, f"Room Status: {status}", (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
    )
    cv2.putText(
        frame,
        datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1,
    )
    return frame


def default_output_path(source) -> str:
    if isinstance(source, int):
        return "result-webcam.mp4"
    name = os.path.basename(str(source))
    root, _ext = os.path.splitext(name)
    return f"result-{root}.mp4"


def run(args: argparse.Namespace) -> None:
    source = resolve_source(args.source)
    cap = open_capture(source)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    output_path = args.output or default_output_path(source)
    save_output = not args.no_save

    background: Optional[np.ndarray] = None
    writer: Optional[cv2.VideoWriter] = None
    frame_idx = 0
    motion_frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            frame = imutils.resize(frame, width=args.resize_width)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0).astype("float")

            if background is None:
                background = gray.copy()

            # Running weighted average background model. More robust to
            # gradual lighting drift than freezing on a single first frame.
            cv2.accumulateWeighted(gray, background, 0.05)
            bg_uint8 = cv2.convertScaleAbs(background)

            frame, status, _boxes = detect_motion(bg_uint8, frame, args.min_area)
            if status == "Motion detected":
                motion_frame_count += 1
            frame = annotate_overlay(frame, status)

            if save_output:
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                writer.write(frame)

            if args.display:
                cv2.imshow("Motion Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            total_str = f"/{total_frames}" if total_frames else ""
            print(
                f"\rProcessing frame {frame_idx}{total_str} - {status}",
                end="",
                flush=True,
            )
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    print()  # newline after the progress line
    print(f"Done. {motion_frame_count}/{frame_idx} frames had motion.")
    if save_output:
        print(f"Saved annotated video to: {output_path}")


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
