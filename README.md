# Motion-Detection 🎥
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Background%20Subtraction-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://img.shields.io/badge/Tests-pytest-0A9EDC?logo=pytest&logoColor=white)

> **Results at a glance:** Running-weighted-average background model (robust to gradual
> lighting drift) · CLI with configurable sensitivity, resize, and webcam support ·
> Unit-tested on synthetic frames, no sample video required

A lightweight **motion detector** built with OpenCV, using classic background
subtraction rather than a single frozen reference frame: each frame is blurred,
diffed against a continuously updated background average (`cv2.accumulateWeighted`),
thresholded into a motion mask, and boxed via contour detection. Started as a Colab
notebook exercise and refactored into a tested, configurable CLI tool that runs on
video files or a live webcam feed.

## Preview

![Motion detection demo](assets/result-test-video.gif)

*Green boxes mark detected motion; the overlay reports "Occupied" vs.
"Unoccupied" (or "Motion detected" / "No motion" in the CLI version) per frame.*

## How it works

1. Each frame is resized, converted to grayscale, and blurred to reduce noise.
2. A **running weighted average** of past frames is maintained as the
   background model (`cv2.accumulateWeighted`), rather than freezing on a
   single first frame — this makes the detector more robust to gradual
   lighting changes.
3. The absolute difference between the current frame and the background is
   thresholded to produce a binary motion mask.
4. Contours are extracted from the mask; any contour above a minimum area
   is treated as a motion region and boxed.
5. The annotated frame is written to an output video, and progress is
   printed as a single live-updating line.

## Installation

```bash
git clone https://github.com/Morteza-Asadi-Shalmaiy/Motion-Detection.git
cd Motion-Detection
pip install -r requirements.txt
```

## Usage

Run on the bundled sample video:

```bash
python motion_detector.py --source assets/test-video.mp4
```

Run on your own video, with a custom output path and sensitivity:

```bash
python motion_detector.py --source my-clip.mp4 --output my-result.mp4 --min-area 800
```

Run on a webcam (index 0) with a live preview window:

```bash
python motion_detector.py --source 0 --display
```

All options:

```
--source        Path to a video file, or webcam index (default: assets/test-video.mp4)
--output        Output video path (default: result-<source name>.mp4)
--min-area      Minimum contour area counted as motion (default: 500)
--resize-width  Width frames are resized to before processing (default: 500)
--display       Show a live window while processing
--no-save       Skip saving the output video
```

## Running tests

```bash
pytest tests/
```

Tests use synthetic in-memory frames (a static background vs. one with an
injected moving block), so they run without needing the sample video.

## Limitations & possible improvements

- Background-subtraction via frame differencing is sensitive to fast
  lighting changes (e.g. clouds passing, lights turning on) and can produce
  false positives from shadows.
- A dedicated background subtractor such as `cv2.createBackgroundSubtractorMOG2`
  or `...KNN` would likely generalize better than the running-average
  approach used here, especially for longer or noisier videos.
- This detects motion regions per frame; it doesn't track individual objects
  across frames (no object identity/tracking).
- Works on pre-recorded video or a single webcam; no multi-camera support.

## Origin

This project started as a Colab notebook exercise
(`motion-detection.ipynb`) and was refactored into a standalone,
tested CLI tool for this repository.

## License

MIT — see [LICENSE](LICENSE).
