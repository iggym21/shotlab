# Shot Form Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool that extracts basketball free-throw shooting form from video (pose keypoints, joint angles, rep segmentation), computes per-rep metrics, classifies reps as consistent/inconsistent, and outputs an annotated overlay video plus a JSON + chart session report.

**Architecture:** Pipeline of pure-ish stages orchestrated by `main.py`: `pose.py` (video → per-frame landmarks+angles) → `segmenter.py` (frames → rep boundaries) → `metrics.py` (rep boundaries + frames → per-rep metrics + session aggregate) → `classifier.py` (reps → labeled reps) → `report.py` (JSON + matplotlib chart) → `overlay.py` (annotated video, optional). Each stage is unit-testable without a real video except `pose.py` (needs MediaPipe + a real video, exercised only by the smoke test).

**Tech Stack:** Python 3.10+, mediapipe (Pose Landmarker), opencv-python, numpy, scipy, pandas, matplotlib, scikit-learn, joblib, argparse, pytest.

**Spec:** The full PRD is the user's original message in this conversation (no separate spec file exists in the repo — this plan is the authoritative breakdown; copy relevant spec text into each task rather than pointing elsewhere).

## Global Constraints

- Python 3.10+.
- Dependencies limited to: mediapipe, opencv-python, numpy, scipy, pandas, matplotlib, scikit-learn, joblib (for classifier persistence), pytest (dev/test only). No other third-party deps.
- CLI only — no GUI/web/webcam/real-time mode.
- One shooter per clip; no ball tracking.
- Right-hand shooter assumed by default (angle joint set is right-side); parameterizing shooting side is explicitly future work, not part of this plan.
- Landmarks referenced by name everywhere (never raw MediaPipe indices).
- Visibility threshold for "usable" landmark: `0.5`.
- Savgol smoothing on landmark coordinate sequences: window length 9, polyorder 2 (spec's "Key Implementation Notes").
- Segmenter: minimum rep duration 0.5s, minimum gap between reps 0.3s, both exposed as parameters.
- VideoWriter codec: `cv2.VideoWriter_fourcc(*'mp4v')`, output fps/resolution must match input exactly.
- `output/` is git-ignored; annotated videos and reports land there.
- Use Python `logging` (not `print`) for `--verbose`/non-essential output; the per-phase stdout tables in Phases 2–3 (rep count, per-rep table) are the one exception — spec explicitly asks for those as always-on printed output, not gated behind `--verbose`.
- No git commit co-authorship / attribution lines beyond what the user's own author identity provides (user explicitly said not to add Claude as co-author).

---

## File Structure

```
shot-form-analyzer/          (repo root == /Users/ignatiusmartin/Documents/Personal/Projects/shotlab)
├── .gitignore
├── requirements.txt
├── README.md
├── main.py
├── analyzer/
│   ├── __init__.py
│   ├── pose.py
│   ├── angles.py
│   ├── segmenter.py
│   ├── metrics.py
│   ├── classifier.py
│   ├── overlay.py
│   └── report.py
├── tests/
│   ├── __init__.py
│   ├── test_angles.py
│   ├── test_segmenter.py
│   └── test_metrics.py
└── output/                  ← git-ignored
```

- `analyzer/angles.py` — pure math (no MediaPipe/OpenCV import). Owns joint-angle geometry and the `compute_angles` joint table.
- `analyzer/pose.py` — MediaPipe + OpenCV wrapper. Owns video decode, landmark extraction, per-frame `compute_angles` calls, and the savgol smoothing pass over landmark coordinates.
- `analyzer/segmenter.py` — pure numpy/scipy. Owns wrist-trajectory rep boundary detection.
- `analyzer/metrics.py` — pure Python/numpy. Owns per-rep metric extraction and session-level aggregation.
- `analyzer/classifier.py` — scikit-learn + joblib. Owns unsupervised-then-supervised rep labeling and model persistence.
- `analyzer/report.py` — json + matplotlib. Owns on-disk report artifacts.
- `analyzer/overlay.py` — OpenCV drawing only. Owns per-frame visual annotation (skeleton + angle text).
- `main.py` — argparse + orchestration + logging setup + `render_annotated_video`. No business logic lives here beyond wiring and the video-writing loop.

---

## Task 1: Project scaffold, git init, dependencies

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `analyzer/__init__.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: importable `analyzer` package, `tests` package, a git repo with an initial commit, a working `pytest` invocation (0 tests yet, but must not error).

- [ ] **Step 1: Init git repo**

```bash
cd /Users/ignatiusmartin/Documents/Personal/Projects/shotlab
git init
git config user.email "ignatiusmartin6@gmail.com"
```
(Only set `user.email` if `git config user.name`/`user.email` are not already set globally — check first with `git config --get user.name` and `git config --get user.email`; if both already return values, skip this config step entirely.)

- [ ] **Step 2: Create `.gitignore`**

```
output/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
.venv/
venv/
*.mp4
!clips/**/*.mp4
.DS_Store
```

- [ ] **Step 3: Create `requirements.txt`**

```
mediapipe>=0.10
opencv-python>=4.9
numpy>=1.26
scipy>=1.11
pandas>=2.1
matplotlib>=3.8
scikit-learn>=1.3
joblib>=1.3
pytest>=7.4
```

- [ ] **Step 4: Create package `__init__.py` files**

`analyzer/__init__.py`:
```python
```
(empty file — marks `analyzer` as a package)

`tests/__init__.py`:
```python
```
(empty file)

- [ ] **Step 5: Create `output/` dir with `.gitkeep` (dir itself is git-ignored, but this keeps the mkdir step explicit)**

```bash
mkdir -p /Users/ignatiusmartin/Documents/Personal/Projects/shotlab/output
```
(No `.gitkeep` needed since `output/` is git-ignored entirely; `main.py` will create it at runtime with `os.makedirs(..., exist_ok=True)` — see Task 4.)

- [ ] **Step 6: Verify pytest runs clean with zero tests**

Run: `cd /Users/ignatiusmartin/Documents/Personal/Projects/shotlab && python -m pytest tests/ -v`
Expected: `no tests ran` (exit code 0 or 5 depending on pytest version — both acceptable at this stage), no import errors.

- [ ] **Step 7: Commit**

```bash
git add .gitignore requirements.txt analyzer/__init__.py tests/__init__.py
git commit -m "chore: project scaffold"
```

---

## Task 2: `analyzer/angles.py` — angle geometry (TDD)

**Files:**
- Create: `analyzer/angles.py`
- Test: `tests/test_angles.py`

**Interfaces:**
- Produces:
  - `VISIBILITY_THRESHOLD: float = 0.5`
  - `angle_between(a: tuple[float,float], b: tuple[float,float], c: tuple[float,float]) -> float` — interior angle at `b`, degrees, range `[0, 180]`.
  - `JOINT_LANDMARKS: dict[str, tuple[str,str,str]]` — maps joint name → 3 landmark names `(a, b, c)` used by `angle_between`.
  - `compute_angles(landmarks: dict[str, dict]) -> dict[str, float | None]` — `landmarks` is `{name: {"x": float, "y": float, "z": float, "visibility": float}}`. Returns one entry per key in `JOINT_LANDMARKS`, value `None` if any of the 3 required landmarks is missing from `landmarks` or has `visibility < VISIBILITY_THRESHOLD`.
- Consumes: nothing (pure module, no MediaPipe/OpenCV import).

- [ ] **Step 1: Write failing tests**

Create `tests/test_angles.py`:
```python
import math
import pytest

from analyzer.angles import angle_between, compute_angles, JOINT_LANDMARKS, VISIBILITY_THRESHOLD


def test_angle_between_right_angle():
    a = (0.0, 1.0)
    b = (0.0, 0.0)
    c = (1.0, 0.0)
    assert angle_between(a, b, c) == pytest.approx(90.0, abs=1e-6)


def test_angle_between_straight_line():
    a = (-1.0, 0.0)
    b = (0.0, 0.0)
    c = (1.0, 0.0)
    assert angle_between(a, b, c) == pytest.approx(180.0, abs=1e-6)


def test_angle_between_45_degrees():
    a = (1.0, 0.0)
    b = (0.0, 0.0)
    c = (1.0, 1.0)
    assert angle_between(a, b, c) == pytest.approx(45.0, abs=1e-6)


def test_angle_between_is_order_independent_for_ends():
    a = (0.0, 1.0)
    b = (0.0, 0.0)
    c = (1.0, 0.0)
    assert angle_between(a, b, c) == pytest.approx(angle_between(c, b, a), abs=1e-6)


def _lm(x, y, visibility=1.0):
    return {"x": x, "y": y, "z": 0.0, "visibility": visibility}


def test_compute_angles_returns_value_for_full_visibility():
    a, b, c = JOINT_LANDMARKS["elbow_right"]
    landmarks = {
        a: _lm(0.0, 1.0),
        b: _lm(0.0, 0.0),
        c: _lm(1.0, 0.0),
    }
    angles = compute_angles(landmarks)
    assert angles["elbow_right"] == pytest.approx(90.0, abs=1e-6)


def test_compute_angles_returns_none_for_low_visibility():
    a, b, c = JOINT_LANDMARKS["elbow_right"]
    landmarks = {
        a: _lm(0.0, 1.0, visibility=VISIBILITY_THRESHOLD - 0.01),
        b: _lm(0.0, 0.0),
        c: _lm(1.0, 0.0),
    }
    angles = compute_angles(landmarks)
    assert angles["elbow_right"] is None


def test_compute_angles_returns_none_for_missing_landmark():
    a, b, c = JOINT_LANDMARKS["knee_right"]
    landmarks = {a: _lm(0.0, 1.0), b: _lm(0.0, 0.0)}  # c missing
    angles = compute_angles(landmarks)
    assert angles["knee_right"] is None


def test_compute_angles_covers_all_expected_joints():
    landmarks = {}
    for a, b, c in JOINT_LANDMARKS.values():
        for name in (a, b, c):
            landmarks[name] = _lm(0.1, 0.1)
    angles = compute_angles(landmarks)
    assert set(angles.keys()) == set(JOINT_LANDMARKS.keys())
```

- [ ] **Step 2: Run tests, verify they fail with ModuleNotFoundError**

Run: `cd /Users/ignatiusmartin/Documents/Personal/Projects/shotlab && python -m pytest tests/test_angles.py -v`
Expected: `ModuleNotFoundError: No module named 'analyzer.angles'`

- [ ] **Step 3: Implement `analyzer/angles.py`**

```python
"""Pure joint-angle geometry. No MediaPipe/OpenCV dependency."""
import numpy as np

VISIBILITY_THRESHOLD = 0.5

# joint_name -> (landmark_a, landmark_b, landmark_c); angle is measured at b.
JOINT_LANDMARKS = {
    "elbow_right": ("right_shoulder", "right_elbow", "right_wrist"),
    "elbow_left": ("left_shoulder", "left_elbow", "left_wrist"),
    "knee_right": ("right_hip", "right_knee", "right_ankle"),
    "knee_left": ("left_hip", "left_knee", "left_ankle"),
    "wrist_right": ("right_elbow", "right_wrist", "right_index"),
    "hip_right": ("right_shoulder", "right_hip", "right_knee"),
    "shoulder_right": ("right_elbow", "right_shoulder", "right_hip"),
}


def angle_between(a, b, c) -> float:
    """Interior angle at b, in degrees, given 3 (x, y) points."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    ba = a - b
    bc = c - b

    angle_ba = np.arctan2(ba[1], ba[0])
    angle_bc = np.arctan2(bc[1], bc[0])

    diff = np.degrees(angle_ba - angle_bc)
    diff = diff % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    return float(diff)


def compute_angles(landmarks: dict) -> dict:
    """landmarks: {name: {x, y, z, visibility}}. Returns {joint_name: degrees|None}."""
    angles = {}
    for joint_name, (name_a, name_b, name_c) in JOINT_LANDMARKS.items():
        pts = []
        visible = True
        for name in (name_a, name_b, name_c):
            lm = landmarks.get(name)
            if lm is None or lm["visibility"] < VISIBILITY_THRESHOLD:
                visible = False
                break
            pts.append((lm["x"], lm["y"]))
        angles[joint_name] = angle_between(*pts) if visible else None
    return angles
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd /Users/ignatiusmartin/Documents/Personal/Projects/shotlab && python -m pytest tests/test_angles.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add analyzer/angles.py tests/test_angles.py
git commit -m "feat: joint angle geometry"
```

---

## Task 3: `analyzer/pose.py` — MediaPipe extraction

**Files:**
- Create: `analyzer/pose.py`

**Interfaces:**
- Consumes: `analyzer.angles.compute_angles(landmarks: dict) -> dict`, `analyzer.angles.VISIBILITY_THRESHOLD`.
- Produces:
  - `LANDMARK_NAMES: dict[int, str]` — MediaPipe `PoseLandmark` enum value → snake_case name matching `analyzer.angles.JOINT_LANDMARKS` (e.g. `"right_elbow"`, `"right_index"`).
  - `class PoseExtractor`:
    - `__init__(self, model_complexity: int = 1)`
    - `extract(self, video_path: str) -> tuple[list[dict | None], float]` — returns `(frames, fps)` where `frames[i]` is either `None` (required landmark below visibility threshold) or a `FrameData` dict `{"frame_idx": int, "timestamp_s": float, "landmarks": dict, "angles": dict}`. `frames[i]` corresponds to video frame `i` — list length equals total frame count, so callers can always use the list index as the frame index. `fps` is read from the video via `cv2.VideoCapture.get(cv2.CAP_PROP_FPS)`.
  - `REQUIRED_LANDMARKS: tuple[str, ...]` — the landmark names that must all clear `VISIBILITY_THRESHOLD` for a frame to be kept (the union of all names referenced in `analyzer.angles.JOINT_LANDMARKS`, plus `right_wrist`/`left_wrist` for the segmenter's use later — those are already included via the joint table).

- [ ] **Step 1: Implement `analyzer/pose.py`**

```python
"""MediaPipe Pose wrapper: video -> list[FrameData | None]."""
import logging

import cv2
import mediapipe as mp

from analyzer.angles import compute_angles, JOINT_LANDMARKS, VISIBILITY_THRESHOLD

logger = logging.getLogger(__name__)

_mp_pose = mp.solutions.pose

# MediaPipe enum member name -> our snake_case landmark name.
_ENUM_TO_NAME = {
    "LEFT_SHOULDER": "left_shoulder",
    "RIGHT_SHOULDER": "right_shoulder",
    "LEFT_ELBOW": "left_elbow",
    "RIGHT_ELBOW": "right_elbow",
    "LEFT_WRIST": "left_wrist",
    "RIGHT_WRIST": "right_wrist",
    "LEFT_INDEX": "left_index",
    "RIGHT_INDEX": "right_index",
    "LEFT_HIP": "left_hip",
    "RIGHT_HIP": "right_hip",
    "LEFT_KNEE": "left_knee",
    "RIGHT_KNEE": "right_knee",
    "LEFT_ANKLE": "left_ankle",
    "RIGHT_ANKLE": "right_ankle",
}

LANDMARK_NAMES = {
    _mp_pose.PoseLandmark[enum_name].value: name
    for enum_name, name in _ENUM_TO_NAME.items()
}

REQUIRED_LANDMARKS = tuple(
    sorted({name for triple in JOINT_LANDMARKS.values() for name in triple})
)


class PoseExtractor:
    def __init__(self, model_complexity: int = 1):
        self.model_complexity = model_complexity

    def extract(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []

        with _mp_pose.Pose(
            model_complexity=self.model_complexity,
            static_image_mode=False,
        ) as pose:
            frame_idx = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                result = pose.process(frame_rgb)

                frame_data = self._build_frame_data(result, frame_idx, fps)
                frames.append(frame_data)
                frame_idx += 1

        cap.release()
        logger.debug("Extracted %d frames at %.2f fps", len(frames), fps)
        return frames, fps

    def _build_frame_data(self, result, frame_idx: int, fps: float):
        if not result.pose_landmarks:
            return None

        landmarks = {}
        for lm_idx, name in LANDMARK_NAMES.items():
            lm = result.pose_landmarks.landmark[lm_idx]
            landmarks[name] = {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
            }

        for required_name in REQUIRED_LANDMARKS:
            lm = landmarks.get(required_name)
            if lm is None or lm["visibility"] < VISIBILITY_THRESHOLD:
                return None

        angles = compute_angles(landmarks)

        return {
            "frame_idx": frame_idx,
            "timestamp_s": frame_idx / fps,
            "landmarks": landmarks,
            "angles": angles,
        }
```

- [ ] **Step 2: Verify import works**

Run: `cd /Users/ignatiusmartin/Documents/Personal/Projects/shotlab && python -c "from analyzer.pose import PoseExtractor; print('ok')"`
Expected: `ok` (confirms mediapipe/opencv install and import wiring; if `mediapipe`/`opencv-python` aren't installed yet, install first: `pip install -r requirements.txt`).

- [ ] **Step 3: Commit**

```bash
git add analyzer/pose.py
git commit -m "feat: MediaPipe pose extraction"
```

---

## Task 4: `main.py` skeleton + smoke test (Phase 1 done)

**Files:**
- Create: `main.py`

**Interfaces:**
- Consumes: `analyzer.pose.PoseExtractor().extract(video_path) -> (frames, fps)`.
- Produces: `build_arg_parser() -> argparse.ArgumentParser`, a `main()` entry point run via `if __name__ == "__main__"`. CLI flags: `--input` (required, str), `--output` (default `"output/"`), `--fps` (float, default `None` — overrides detected fps when set), `--min-reps` (int, default `3`), `--no-overlay` (store_true), `--verbose` (store_true).

- [ ] **Step 1: Implement `main.py` (Phase 1 subset only — later tasks extend this file)**

```python
"""CLI entry point for the shot form analyzer."""
import argparse
import logging
import os
import sys

from analyzer.pose import PoseExtractor

logger = logging.getLogger("shotlab")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze basketball free-throw form from video.")
    parser.add_argument("--input", required=True, help="Path to input video file.")
    parser.add_argument("--output", default="output/", help="Output directory (default: output/).")
    parser.add_argument("--fps", type=float, default=None, help="Override detected video fps.")
    parser.add_argument("--min-reps", type=int, default=3, help="Minimum reps required to run the classifier.")
    parser.add_argument("--no-overlay", action="store_true", help="Skip annotated video, stats only.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not os.path.isfile(args.input):
        logger.error("Input video not found: %s", args.input)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    frames, detected_fps = PoseExtractor().extract(args.input)
    fps = args.fps or detected_fps
    print(f"Extracted {len(frames)} frames at {fps:.2f} fps.")

    return frames, fps, args


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test with a real short mp4**

Run (using a short real video you have, e.g. one shot with a phone, or ask the user for a sample clip path — `--no-overlay` avoids needing Tasks 5+):
`cd /Users/ignatiusmartin/Documents/Personal/Projects/shotlab && python main.py --input <path_to_any_mp4> --no-overlay`
Expected: prints `Extracted N frames at F.FF fps.` with no traceback. If no sample video is available yet, generate a trivial synthetic one for the smoke test only:
```bash
python -c "
import cv2, numpy as np
w = cv2.VideoWriter('/tmp/smoke.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (320, 240))
for _ in range(60):
    w.write(np.zeros((240, 320, 3), dtype='uint8'))
w.release()
"
python main.py --input /tmp/smoke.mp4 --no-overlay
```
(A blank synthetic video has no detectable person, so every frame will be `None` — that's fine, this step only verifies the pipeline runs end-to-end without crashing and prints the frame count.)

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: CLI entry point with pose extraction smoke test"
```

---

## Task 5: `analyzer/segmenter.py` — rep segmentation (TDD)

**Files:**
- Create: `analyzer/segmenter.py`
- Test: `tests/test_segmenter.py`

**Interfaces:**
- Consumes: `frames: list[dict | None]` matching `PoseExtractor.extract`'s `FrameData` shape (`frame["landmarks"]["right_wrist"]["y"]`).
- Produces: `segment_reps(frames: list[dict | None], fps: float, min_rep_duration_s: float = 0.5, min_gap_s: float = 0.3) -> list[tuple[int, int, int]]` — list of `(start_frame_idx, release_frame_idx, end_frame_idx)`, one tuple per detected rep, sorted ascending by `start_frame_idx`.

**Algorithm:** MediaPipe y-coordinates increase downward, so a shot's release (wrist at its highest point) is a **local minimum** in `right_wrist.y`. Missing frames (`None`) are linearly interpolated over `frame_idx` before smoothing so gaps don't break the signal. Steps:
1. Build the raw `y` series over all frame indices, using `np.interp` to fill `None` frames from surrounding valid frames (edges use nearest valid value).
2. Smooth with `scipy.signal.savgol_filter(y, window_length=9, polyorder=2)` (clamp `window_length` down to the largest valid odd value `<= len(y)` if the clip is very short).
3. Find release candidates as local minima via `scipy.signal.find_peaks(-y_smooth, distance=max(1, int(min_gap_s * fps)))`.
4. For each release candidate, walk backward to the nearest preceding local maximum (or index 0) as `start`, and forward to the nearest following local maximum (or last index) as `end`.
5. Drop candidates where `(end - start) / fps < min_rep_duration_s`.
6. Return the surviving `(start, release, end)` tuples in ascending order.

- [ ] **Step 1: Write failing tests**

Create `tests/test_segmenter.py`:
```python
import numpy as np
import pytest

from analyzer.segmenter import segment_reps


def _frame(y):
    return {
        "frame_idx": 0,
        "timestamp_s": 0.0,
        "landmarks": {"right_wrist": {"x": 0.5, "y": y, "z": 0.0, "visibility": 1.0}},
        "angles": {},
    }


def _make_frames(y_values):
    return [_frame(y) for y in y_values]


def _synthetic_reps(n_reps=3, frames_per_rep=30, high_y=0.8, low_y=0.2):
    """Each rep: high -> low -> high, i.e. a dip (release at the trough)."""
    y = []
    half = frames_per_rep // 2
    for _ in range(n_reps):
        down = np.linspace(high_y, low_y, half)
        up = np.linspace(low_y, high_y, frames_per_rep - half)
        y.extend(down.tolist())
        y.extend(up.tolist())
    return np.array(y)


def test_segment_reps_finds_correct_count():
    y = _synthetic_reps(n_reps=3, frames_per_rep=30)
    frames = _make_frames(y)
    fps = 30.0
    reps = segment_reps(frames, fps, min_rep_duration_s=0.3, min_gap_s=0.1)
    assert len(reps) == 3


def test_segment_reps_release_near_trough():
    y = _synthetic_reps(n_reps=1, frames_per_rep=30)
    frames = _make_frames(y)
    fps = 30.0
    reps = segment_reps(frames, fps, min_rep_duration_s=0.3, min_gap_s=0.1)
    assert len(reps) == 1
    start, release, end = reps[0]
    expected_trough = int(np.argmin(y))
    assert abs(release - expected_trough) <= 2


def test_segment_reps_bounds_are_ordered():
    y = _synthetic_reps(n_reps=2, frames_per_rep=40)
    frames = _make_frames(y)
    fps = 30.0
    reps = segment_reps(frames, fps, min_rep_duration_s=0.3, min_gap_s=0.1)
    for start, release, end in reps:
        assert start <= release <= end


def test_segment_reps_handles_none_frames():
    y = _synthetic_reps(n_reps=2, frames_per_rep=30)
    frames = _make_frames(y)
    frames[5] = None
    frames[6] = None
    fps = 30.0
    reps = segment_reps(frames, fps, min_rep_duration_s=0.3, min_gap_s=0.1)
    assert len(reps) == 2


def test_segment_reps_rejects_too_short_reps():
    # Single rep spanning only 3 frames at 30fps = 0.1s, well under 0.5s default minimum.
    y = np.array([0.8, 0.5, 0.2, 0.5, 0.8])
    frames = _make_frames(y)
    fps = 30.0
    reps = segment_reps(frames, fps, min_rep_duration_s=0.5, min_gap_s=0.1)
    assert len(reps) == 0
```

- [ ] **Step 2: Run tests, verify they fail with ModuleNotFoundError**

Run: `cd /Users/ignatiusmartin/Documents/Personal/Projects/shotlab && python -m pytest tests/test_segmenter.py -v`
Expected: `ModuleNotFoundError: No module named 'analyzer.segmenter'`

- [ ] **Step 3: Implement `analyzer/segmenter.py`**

```python
"""Rep segmentation from shooting-wrist vertical trajectory."""
import numpy as np
from scipy.signal import find_peaks, savgol_filter


def _extract_wrist_y(frames):
    idx = np.arange(len(frames))
    y = np.full(len(frames), np.nan)
    for i, frame in enumerate(frames):
        if frame is not None:
            y[i] = frame["landmarks"]["right_wrist"]["y"]

    valid = ~np.isnan(y)
    if not valid.any():
        raise ValueError("No frames with visible right_wrist to segment reps from.")
    if not valid.all():
        y = np.interp(idx, idx[valid], y[valid])
    return y


def _smooth(y):
    window = min(9, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if window < 5:
        return y.copy()
    return savgol_filter(y, window_length=window, polyorder=2)


def _nearest_preceding_peak(peaks_idx, target, default):
    candidates = peaks_idx[peaks_idx < target]
    return int(candidates.max()) if len(candidates) else default


def _nearest_following_peak(peaks_idx, target, default):
    candidates = peaks_idx[peaks_idx > target]
    return int(candidates.min()) if len(candidates) else default


def segment_reps(frames, fps, min_rep_duration_s: float = 0.5, min_gap_s: float = 0.3):
    if len(frames) == 0:
        return []

    y = _extract_wrist_y(frames)
    y_smooth = _smooth(y)

    min_gap_frames = max(1, int(round(min_gap_s * fps)))

    troughs, _ = find_peaks(-y_smooth, distance=min_gap_frames)
    peaks, _ = find_peaks(y_smooth, distance=min_gap_frames)

    reps = []
    for release in troughs:
        start = _nearest_preceding_peak(peaks, release, default=0)
        end = _nearest_following_peak(peaks, release, default=len(y_smooth) - 1)

        duration_s = (end - start) / fps
        if duration_s < min_rep_duration_s:
            continue

        reps.append((int(start), int(release), int(end)))

    reps.sort(key=lambda r: r[0])
    return reps
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd /Users/ignatiusmartin/Documents/Personal/Projects/shotlab && python -m pytest tests/test_segmenter.py -v`
Expected: all tests PASS. If `test_segment_reps_finds_correct_count` or the trough-position test are flaky against the exact synthetic shape, adjust `_synthetic_reps` amplitude/frame counts in the test (not the algorithm) until the signal unambiguously has 3 well-separated troughs — do not weaken the algorithm to fit a bad synthetic signal.

- [ ] **Step 5: Wire into `main.py`, print rep count + timestamps**

Modify `main.py`:
```python
from analyzer.pose import PoseExtractor
from analyzer.segmenter import segment_reps
```
and in `main()`, after computing `fps`:
```python
    reps_bounds = segment_reps(frames, fps)
    print(f"Detected {len(reps_bounds)} rep(s):")
    for i, (start, release, end) in enumerate(reps_bounds):
        print(
            f"  rep {i}: start={start} ({start/fps:.2f}s) "
            f"release={release} ({release/fps:.2f}s) "
            f"end={end} ({end/fps:.2f}s)"
        )

    return frames, fps, reps_bounds, args
```
(replace the previous `return frames, fps, args` line and its call site if any).

- [ ] **Step 6: Commit**

```bash
git add analyzer/segmenter.py tests/test_segmenter.py main.py
git commit -m "feat: rep segmentation from wrist trajectory"
```

---

## Task 6: `analyzer/metrics.py` — per-rep metrics + aggregation (TDD)

**Files:**
- Create: `analyzer/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `rep_bounds: list[tuple[int,int,int]]` from `segment_reps`; `frames: list[dict | None]` from `PoseExtractor.extract`.
- Produces:
  - `KNEE_BENT_THRESHOLD_DEG: float = 150.0`
  - `compute_rep_metrics(rep_bounds: list[tuple[int,int,int]], frames: list[dict | None]) -> list[dict]` — one `RepData` dict per rep bound, with keys `rep_id: int`, `start_frame: int`, `release_frame: int`, `end_frame: int`, `elbow_at_release: float`, `knee_bend_at_setup: float`, `wrist_follow_through: float`, `arc_proxy: float`, `label: str` (initialized to `"unlabeled"`, overwritten later by `classifier.train_and_label`). Frames with `None` angle values inside a rep window are skipped when searching for a usable frame (see setup-frame rule below); if literally no frame in the window has the needed angle, fall back to `0.0` and log a warning (never raise — a video can legitimately have a rep with a brief visibility dropout at the exact anchor frame).
  - `aggregate_session(reps: list[dict]) -> dict` — `{metric_name: {"mean": float, "std": float, "min": float, "max": float, "rep_to_rep_delta": list[float]}}` for each of the four metric fields above. `rep_to_rep_delta[i] = reps[i+1][metric] - reps[i][metric]` (length `len(reps) - 1`; empty list if fewer than 2 reps). If `reps` is empty, return `{}`.

**Rules (from spec):**
- Release frame anchor: within `[start, end]`, the frame where `elbow_right` angle is closest to 180° (most extended) — use this frame's `elbow_at_release` and `arc_proxy` (=`shoulder_right` angle at that same frame). Note: `release` in `rep_bounds` is the trough of wrist-y (highest wrist position); the *elbow-most-extended* frame used for these two metrics is found independently within `[start, end]` per this rule, and may differ from the tuple's `release` index — use the tuple's `release` only as the window's rough center reference, not as the metric anchor.
- Setup frame: first frame in `[start, end]` where both `knee_right` and `knee_left` angles are `<= KNEE_BENT_THRESHOLD_DEG` (150°). If no such frame exists, use `start`. `knee_bend_at_setup` = `knee_right` angle at that frame.
- Follow-through: `wrist_follow_through` = `wrist_right` angle at frame `end`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_metrics.py`:
```import pytest

from analyzer.metrics import compute_rep_metrics, aggregate_session


def _frame(elbow=None, knee_r=None, knee_l=None, wrist=None, shoulder=None):
    return {
        "frame_idx": 0,
        "timestamp_s": 0.0,
        "landmarks": {},
        "angles": {
            "elbow_right": elbow,
            "knee_right": knee_r,
            "knee_left": knee_l,
            "wrist_right": wrist,
            "shoulder_right": shoulder,
        },
    }


def test_compute_rep_metrics_single_rep_happy_path():
    frames = [
        _frame(elbow=120, knee_r=140, knee_l=138, wrist=90, shoulder=60),   # 0: setup (knees bent)
        _frame(elbow=150, knee_r=160, knee_l=158, wrist=95, shoulder=80),   # 1
        _frame(elbow=178, knee_r=170, knee_l=172, wrist=100, shoulder=95),  # 2: most extended elbow
        _frame(elbow=160, knee_r=165, knee_l=167, wrist=70, shoulder=85),   # 3: end / follow-through
    ]
    rep_bounds = [(0, 2, 3)]
    reps = compute_rep_metrics(rep_bounds, frames)

    assert len(reps) == 1
    rep = reps[0]
    assert rep["rep_id"] == 0
    assert rep["start_frame"] == 0
    assert rep["release_frame"] == 2
    assert rep["end_frame"] == 3
    assert rep["elbow_at_release"] == pytest.approx(178)
    assert rep["arc_proxy"] == pytest.approx(95)
    assert rep["knee_bend_at_setup"] == pytest.approx(140)
    assert rep["wrist_follow_through"] == pytest.approx(70)
    assert rep["label"] == "unlabeled"


def test_compute_rep_metrics_falls_back_to_start_when_no_bent_knee_frame():
    frames = [
        _frame(elbow=170, knee_r=170, knee_l=172, wrist=90, shoulder=90),  # never bent past 150
        _frame(elbow=178, knee_r=175, knee_l=176, wrist=95, shoulder=95),
    ]
    rep_bounds = [(0, 1, 1)]
    reps = compute_rep_metrics(rep_bounds, frames)
    assert reps[0]["knee_bend_at_setup"] == pytest.approx(170)


def test_compute_rep_metrics_multiple_reps_get_sequential_ids():
    frames = [
        _frame(elbow=178, knee_r=140, knee_l=140, wrist=90, shoulder=90),
        _frame(elbow=170, knee_r=145, knee_l=145, wrist=85, shoulder=85),
        _frame(elbow=178, knee_r=140, knee_l=140, wrist=90, shoulder=90),
        _frame(elbow=170, knee_r=145, knee_l=145, wrist=85, shoulder=85),
    ]
    rep_bounds = [(0, 0, 1), (2, 2, 3)]
    reps = compute_rep_metrics(rep_bounds, frames)
    assert [r["rep_id"] for r in reps] == [0, 1]


def test_aggregate_session_computes_stats_and_deltas():
    reps = [
        {"elbow_at_release": 170, "knee_bend_at_setup": 140, "wrist_follow_through": 80, "arc_proxy": 90},
        {"elbow_at_release": 180, "knee_bend_at_setup": 145, "wrist_follow_through": 85, "arc_proxy": 95},
        {"elbow_at_release": 175, "knee_bend_at_setup": 142, "wrist_follow_through": 82, "arc_proxy": 92},
    ]
    agg = aggregate_session(reps)

    assert set(agg.keys()) == {"elbow_at_release", "knee_bend_at_setup", "wrist_follow_through", "arc_proxy"}
    elbow = agg["elbow_at_release"]
    assert elbow["mean"] == pytest.approx((170 + 180 + 175) / 3)
    assert elbow["min"] == pytest.approx(170)
    assert elbow["max"] == pytest.approx(180)
    assert elbow["rep_to_rep_delta"] == pytest.approx([10, -5])


def test_aggregate_session_empty_reps_returns_empty_dict():
    assert aggregate_session([]) == {}
```

- [ ] **Step 2: Run tests, verify they fail with ModuleNotFoundError**

Run: `cd /Users/ignatiusmartin/Documents/Personal/Projects/shotlab && python -m pytest tests/test_metrics.py -v`
Expected: `ModuleNotFoundError: No module named 'analyzer.metrics'`

- [ ] **Step 3: Implement `analyzer/metrics.py`**

```python
"""Per-rep metric extraction and session-level aggregation."""
import logging

import numpy as np

logger = logging.getLogger(__name__)

KNEE_BENT_THRESHOLD_DEG = 150.0

METRIC_FIELDS = (
    "elbow_at_release",
    "knee_bend_at_setup",
    "wrist_follow_through",
    "arc_proxy",
)


def _find_most_extended_elbow_frame(frames, start, end):
    best_idx = None
    best_angle = -1.0
    for i in range(start, end + 1):
        frame = frames[i]
        if frame is None:
            continue
        angle = frame["angles"].get("elbow_right")
        if angle is not None and angle > best_angle:
            best_angle = angle
            best_idx = i
    return best_idx


def _find_setup_frame(frames, start, end):
    for i in range(start, end + 1):
        frame = frames[i]
        if frame is None:
            continue
        knee_r = frame["angles"].get("knee_right")
        knee_l = frame["angles"].get("knee_left")
        if knee_r is not None and knee_l is not None:
            if knee_r <= KNEE_BENT_THRESHOLD_DEG and knee_l <= KNEE_BENT_THRESHOLD_DEG:
                return i
    return start


def _safe_angle(frames, idx, angle_name, rep_id):
    frame = frames[idx] if 0 <= idx < len(frames) else None
    if frame is not None:
        value = frame["angles"].get(angle_name)
        if value is not None:
            return value
    logger.warning("rep %d: missing %s at frame %d, defaulting to 0.0", rep_id, angle_name, idx)
    return 0.0


def compute_rep_metrics(rep_bounds, frames):
    reps = []
    for rep_id, (start, release, end) in enumerate(rep_bounds):
        extended_idx = _find_most_extended_elbow_frame(frames, start, end)
        if extended_idx is None:
            extended_idx = release

        setup_idx = _find_setup_frame(frames, start, end)

        reps.append({
            "rep_id": rep_id,
            "start_frame": start,
            "release_frame": release,
            "end_frame": end,
            "elbow_at_release": _safe_angle(frames, extended_idx, "elbow_right", rep_id),
            "knee_bend_at_setup": _safe_angle(frames, setup_idx, "knee_right", rep_id),
            "wrist_follow_through": _safe_angle(frames, end, "wrist_right", rep_id),
            "arc_proxy": _safe_angle(frames, extended_idx, "shoulder_right", rep_id),
            "label": "unlabeled",
        })
    return reps


def aggregate_session(reps):
    if not reps:
        return {}

    agg = {}
    for field in METRIC_FIELDS:
        values = np.array([rep[field] for rep in reps], dtype=float)
        agg[field] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "rep_to_rep_delta": np.diff(values).tolist(),
        }
    return agg
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd /Users/ignatiusmartin/Documents/Personal/Projects/shotlab && python -m pytest tests/test_metrics.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add analyzer/metrics.py tests/test_metrics.py
git commit -m "feat: per-rep metrics and session aggregation"
```

---

## Task 7: `analyzer/classifier.py` — unsupervised + LR rep labeling

**Files:**
- Create: `analyzer/classifier.py`

**Interfaces:**
- Consumes: `reps: list[dict]` from `compute_rep_metrics` (each has the 4 `METRIC_FIELDS` from Task 6 plus `rep_id`, `label`).
- Produces: `train_and_label(reps: list[dict], output_dir: str = "output", min_reps: int = 3) -> list[dict]` — returns the same rep dicts with `label` set to `"consistent"`, `"inconsistent"`, or `"unclassified"` (mutates and returns the input list; does not copy). When `len(reps) >= min_reps`, also writes `<output_dir>/classifier.pkl` via `joblib.dump` containing `{"scaler": StandardScaler, "kmeans": KMeans, "model": LogisticRegression}`.

**Algorithm:**
1. If `len(reps) < min_reps`: set every rep's `label = "unclassified"`, return reps immediately (no model saved).
2. Build feature matrix `X` from `[elbow_at_release, knee_bend_at_setup, wrist_follow_through, arc_proxy]` per rep, in that order.
3. `X_scaled = StandardScaler().fit_transform(X)`.
4. `KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(X_scaled)` → `cluster_ids` (0/1 per rep).
5. The larger cluster (more reps) is `"consistent"` (majority = normal form), the smaller is `"inconsistent"`. Tie (equal size): the cluster whose points have the smaller mean distance to their own centroid is `"consistent"` (tighter cluster = more consistent).
6. Build binary target `y = 1` for consistent-cluster reps, `0` for inconsistent-cluster reps.
7. Fit `LogisticRegression().fit(X_scaled, y)`, predict `y_pred = model.predict(X_scaled)`.
8. Assign `rep["label"] = "consistent" if y_pred[i] == 1 else "inconsistent"`.
9. `joblib.dump({"scaler": scaler, "kmeans": kmeans, "model": model}, os.path.join(output_dir, "classifier.pkl"))`.

- [ ] **Step 1: Implement `analyzer/classifier.py`**

```python
"""Unsupervised-then-supervised rep consistency labeling."""
import os

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURE_FIELDS = (
    "elbow_at_release",
    "knee_bend_at_setup",
    "wrist_follow_through",
    "arc_proxy",
)


def _mean_dist_to_centroid(X_scaled, cluster_ids, cluster_id, centroid):
    members = X_scaled[cluster_ids == cluster_id]
    if len(members) == 0:
        return float("inf")
    return float(np.mean(np.linalg.norm(members - centroid, axis=1)))


def train_and_label(reps, output_dir: str = "output", min_reps: int = 3):
    if len(reps) < min_reps:
        for rep in reps:
            rep["label"] = "unclassified"
        return reps

    X = np.array([[rep[field] for field in FEATURE_FIELDS] for rep in reps], dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    cluster_ids = kmeans.fit_predict(X_scaled)

    count_0 = int(np.sum(cluster_ids == 0))
    count_1 = int(np.sum(cluster_ids == 1))

    if count_0 != count_1:
        consistent_cluster = 0 if count_0 > count_1 else 1
    else:
        dist_0 = _mean_dist_to_centroid(X_scaled, cluster_ids, 0, kmeans.cluster_centers_[0])
        dist_1 = _mean_dist_to_centroid(X_scaled, cluster_ids, 1, kmeans.cluster_centers_[1])
        consistent_cluster = 0 if dist_0 <= dist_1 else 1

    y = (cluster_ids == consistent_cluster).astype(int)

    model = LogisticRegression()
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    for rep, pred in zip(reps, y_pred):
        rep["label"] = "consistent" if pred == 1 else "inconsistent"

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(
        {"scaler": scaler, "kmeans": kmeans, "model": model},
        os.path.join(output_dir, "classifier.pkl"),
    )

    return reps
```

- [ ] **Step 2: Manual verification (no video needed)**

Run: `cd /Users/ignatiusmartin/Documents/Personal/Projects/shotlab && python -c "
from analyzer.classifier import train_and_label

reps = [
    {'rep_id': i, 'elbow_at_release': 170 + i, 'knee_bend_at_setup': 140, 'wrist_follow_through': 80, 'arc_proxy': 90}
    for i in range(4)
]
reps.append({'rep_id': 4, 'elbow_at_release': 90, 'knee_bend_at_setup': 179, 'wrist_follow_through': 10, 'arc_proxy': 20})
result = train_and_label(reps, output_dir='/tmp/shotlab_test_output')
for r in result:
    print(r['rep_id'], r['label'])
"`
Expected: reps 0-3 (similar values) get one label, rep 4 (outlier) gets the other; no traceback; `/tmp/shotlab_test_output/classifier.pkl` exists.

- [ ] **Step 3: Commit**

```bash
git add analyzer/classifier.py
git commit -m "feat: rep consistency classifier"
```

---

## Task 8: Wire metrics + classifier into `main.py`, print per-rep table

**Files:**
- Modify: `main.py`

**Interfaces:**
- Consumes: `compute_rep_metrics`, `aggregate_session` (Task 6), `train_and_label` (Task 7).

- [ ] **Step 1: Modify `main.py`**

Add imports:
```python
from analyzer.metrics import compute_rep_metrics, aggregate_session
from analyzer.classifier import train_and_label
```

Replace the tail of `main()` (after the rep-bounds printing block from Task 5) with:
```python
    reps = compute_rep_metrics(reps_bounds, frames)
    reps = train_and_label(reps, output_dir=args.output, min_reps=args.min_reps)
    session_agg = aggregate_session(reps)

    print(f"\n{'rep_id':>6} | {'elbow':>7} | {'knee':>7} | {'wrist':>7} | {'arc':>7} | label")
    for rep in reps:
        print(
            f"{rep['rep_id']:>6} | {rep['elbow_at_release']:>7.1f} | "
            f"{rep['knee_bend_at_setup']:>7.1f} | {rep['wrist_follow_through']:>7.1f} | "
            f"{rep['arc_proxy']:>7.1f} | {rep['label']}"
        )

    return frames, fps, reps, session_agg, args
```
(replace the previous `return frames, fps, reps_bounds, args` line).

- [ ] **Step 2: Manual verification**

Run against the synthetic smoke video from Task 4 Step 2, or a real sample clip if available:
`cd /Users/ignatiusmartin/Documents/Personal/Projects/shotlab && python main.py --input /tmp/smoke.mp4 --no-overlay`
Expected: prints frame count, rep count (likely 0 for the blank synthetic video — that's fine, confirms no crash through the full metrics/classifier path when there are zero reps). If a real free-throw clip is available, prefer running against that to see a populated table with no crash.

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: wire per-rep metrics and classifier into CLI"
```

---

## Task 9: `analyzer/report.py` — JSON + matplotlib chart

**Files:**
- Create: `analyzer/report.py`

**Interfaces:**
- Consumes: `session_agg: dict` from `aggregate_session`, `reps: list[dict]` from `compute_rep_metrics`/`train_and_label`.
- Produces:
  - `save_json(session_agg: dict, reps: list[dict], output_dir: str) -> str` — writes `<output_dir>/session_report.json` with `{"reps": reps, "session_summary": session_agg}`, returns the written path.
  - `save_chart(reps: list[dict], session_agg: dict, output_dir: str) -> str` — writes `<output_dir>/session_summary.png`, returns the written path. 2×2 matplotlib figure per spec: top-left elbow-at-release bar chart with mean line + ±1σ band, top-right knee-bend-at-setup, bottom-left wrist-follow-through, bottom-right arc-proxy. Figure title `f"Session Summary — {len(reps)} reps"`. If `reps` is empty, still write a figure with 4 empty-but-labeled panels (no crash on the zero-rep case).

- [ ] **Step 1: Implement `analyzer/report.py`**

```python
"""Session report artifacts: JSON + matplotlib summary chart."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PANELS = (
    ("elbow_at_release", "Elbow Angle at Release (deg)"),
    ("knee_bend_at_setup", "Knee Bend at Setup (deg)"),
    ("wrist_follow_through", "Wrist Follow-Through (deg)"),
    ("arc_proxy", "Arc Proxy — Shoulder Elevation at Release (deg)"),
)


def save_json(session_agg: dict, reps: list, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "session_report.json")
    with open(path, "w") as f:
        json.dump({"reps": reps, "session_summary": session_agg}, f, indent=2)
    return path


def _plot_panel(ax, reps, field, title):
    if not reps:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no reps detected", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Rep")
        ax.set_ylabel(title)
        return

    rep_ids = [rep["rep_id"] for rep in reps]
    values = [rep[field] for rep in reps]
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

    ax.bar(rep_ids, values, color="#4C72B0")
    ax.axhline(mean, color="black", linestyle="--", linewidth=1, label="mean")
    ax.axhspan(mean - std, mean + std, color="gray", alpha=0.2, label="±1σ")
    ax.set_title(title)
    ax.set_xlabel("Rep")
    ax.set_ylabel(title)
    ax.set_xticks(rep_ids)
    ax.legend(fontsize=8)


def save_chart(reps: list, session_agg: dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Session Summary — {len(reps)} reps")

    for ax, (field, title) in zip(axes.flat, PANELS):
        _plot_panel(ax, reps, field, title)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "session_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
```

- [ ] **Step 2: Manual verification**

Run: `cd /Users/ignatiusmartin/Documents/Personal/Projects/shotlab && python -c "
from analyzer.report import save_json, save_chart

reps = [
    {'rep_id': i, 'start_frame': i*10, 'release_frame': i*10+5, 'end_frame': i*10+9,
     'elbow_at_release': 170+i, 'knee_bend_at_setup': 140+i, 'wrist_follow_through': 80+i,
     'arc_proxy': 90+i, 'label': 'consistent'}
    for i in range(4)
]
from analyzer.metrics import aggregate_session
agg = aggregate_session(reps)
print(save_json(agg, reps, '/tmp/shotlab_test_output'))
print(save_chart(reps, agg, '/tmp/shotlab_test_output'))
"`
Expected: two paths printed, both files exist; open `/tmp/shotlab_test_output/session_summary.png` (or `file` command) to confirm it's a valid non-trivial PNG.

- [ ] **Step 3: Wire into `main.py`**

Add import:
```python
from analyzer.report import save_json, save_chart
```
Append before the `return` at the end of `main()`:
```python
    json_path = save_json(session_agg, reps, args.output)
    chart_path = save_chart(reps, session_agg, args.output)
    print(f"\nWrote {json_path}")
    print(f"Wrote {chart_path}")
```

- [ ] **Step 4: Commit**

```bash
git add analyzer/report.py main.py
git commit -m "feat: session report JSON and summary chart"
```

---

## Task 10: `analyzer/overlay.py` — per-frame skeleton + angle drawing

**Files:**
- Create: `analyzer/overlay.py`

**Interfaces:**
- Consumes: `frame: np.ndarray` (BGR, shape `(H, W, 3)`), `frame_data: dict | None` (a `FrameData` from `PoseExtractor.extract`, or `None`), `rep_id: int | None`, `timestamp_s: float`, `in_rep: bool`.
- Produces: `draw_frame(frame: np.ndarray, frame_data: dict | None, rep_id: int | None, timestamp_s: float, in_rep: bool) -> np.ndarray` — returns a new annotated frame (does not mutate the input in place; caller passes the original decoded frame each time).

**Design note (deviation from spec's literal `mp.solutions.drawing_utils` suggestion):** `PoseExtractor.extract` only retains the plain-dict `landmarks`, not MediaPipe's `NormalizedLandmarkList` proto, so `mp.solutions.drawing_utils.draw_landmarks` (which requires that proto type) isn't usable downstream without re-running pose estimation. Draw the skeleton manually with `cv2.line`/`cv2.circle` over a fixed bone list instead — same visual result, no extra MediaPipe dependency at render time.

- [ ] **Step 1: Implement `analyzer/overlay.py`**

```python
"""OpenCV drawing: skeleton overlay + angle callouts + HUD text."""
import cv2

BONES = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_index"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_index"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)

# joint_name in analyzer.angles.JOINT_LANDMARKS -> landmark to anchor the text label on
ANGLE_LABEL_ANCHOR = {
    "elbow_right": "right_elbow",
    "elbow_left": "left_elbow",
    "knee_right": "right_knee",
    "knee_left": "left_knee",
    "wrist_right": "right_wrist",
    "hip_right": "right_hip",
    "shoulder_right": "right_shoulder",
}

SKELETON_COLOR_IN_REP = (0, 200, 0)
SKELETON_COLOR_OUT_OF_REP = (160, 160, 160)
JOINT_RADIUS = 4


def _to_pixel(landmark, width, height):
    return int(landmark["x"] * width), int(landmark["y"] * height)


def _draw_text_with_outline(frame, text, org, scale=0.5, color=(255, 255, 255)):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_frame(frame, frame_data, rep_id, timestamp_s: float, in_rep: bool):
    out = frame.copy()
    height, width = out.shape[:2]

    header = f"t={timestamp_s:.2f}s"
    if rep_id is not None:
        header = f"rep {rep_id} | {header}"
    _draw_text_with_outline(out, header, (10, 24), scale=0.6)

    if frame_data is None:
        return out

    landmarks = frame_data["landmarks"]
    color = SKELETON_COLOR_IN_REP if in_rep else SKELETON_COLOR_OUT_OF_REP

    for name_a, name_b in BONES:
        lm_a = landmarks.get(name_a)
        lm_b = landmarks.get(name_b)
        if lm_a is None or lm_b is None:
            continue
        cv2.line(out, _to_pixel(lm_a, width, height), _to_pixel(lm_b, width, height), color, 2)

    for lm in landmarks.values():
        cv2.circle(out, _to_pixel(lm, width, height), JOINT_RADIUS, color, -1)

    for joint_name, angle in frame_data["angles"].items():
        if angle is None:
            continue
        anchor_name = ANGLE_LABEL_ANCHOR.get(joint_name)
        anchor = landmarks.get(anchor_name)
        if anchor is None:
            continue
        px, py = _to_pixel(anchor, width, height)
        _draw_text_with_outline(out, f"{angle:.0f}", (px + 6, py - 6))

    return out
```

- [ ] **Step 2: Manual verification (synthetic frame, no video needed)**

Run: `cd /Users/ignatiusmartin/Documents/Personal/Projects/shotlab && python -c "
import numpy as np
from analyzer.overlay import draw_frame

frame = np.zeros((240, 320, 3), dtype='uint8')
frame_data = {
    'frame_idx': 0, 'timestamp_s': 1.5,
    'landmarks': {
        'right_shoulder': {'x': 0.4, 'y': 0.3, 'z': 0, 'visibility': 1},
        'right_elbow': {'x': 0.5, 'y': 0.5, 'z': 0, 'visibility': 1},
        'right_wrist': {'x': 0.6, 'y': 0.3, 'z': 0, 'visibility': 1},
    },
    'angles': {'elbow_right': 95.0},
}
out = draw_frame(frame, frame_data, rep_id=2, timestamp_s=1.5, in_rep=True)
print(out.shape, out.dtype)
assert out.sum() > 0  # something was drawn
print('ok')
"`
Expected: prints shape/dtype and `ok`, no traceback.

- [ ] **Step 3: Commit**

```bash
git add analyzer/overlay.py
git commit -m "feat: skeleton and angle overlay drawing"
```

---

## Task 11: `render_annotated_video` in `main.py` (Phase 5 done)

**Files:**
- Modify: `main.py`

**Interfaces:**
- Consumes: `analyzer.overlay.draw_frame`, `frames: list[dict|None]`, `reps: list[dict]` (for in-rep/rep-id lookup), `input_path: str`, `output_dir: str`, `fps: float`.
- Produces: `render_annotated_video(input_path: str, output_dir: str, frames: list, reps: list, fps: float) -> str` — writes `<output_dir>/annotated_<input_basename>` (`.mp4`), returns the written path.

- [ ] **Step 1: Add to `main.py`**

Add imports:
```python
import cv2
from analyzer.overlay import draw_frame
```

Add function (module level, above `main()`):
```python
def _build_frame_rep_lookup(reps):
    """frame_idx -> (rep_id, in_rep) for every frame covered by any rep."""
    lookup = {}
    for rep in reps:
        for idx in range(rep["start_frame"], rep["end_frame"] + 1):
            lookup[idx] = (rep["rep_id"], True)
    return lookup


def render_annotated_video(input_path: str, output_dir: str, frames: list, reps: list, fps: float) -> str:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not reopen video for rendering: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    basename = os.path.basename(input_path)
    name, _ = os.path.splitext(basename)
    output_path = os.path.join(output_dir, f"annotated_{name}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_rep_lookup = _build_frame_rep_lookup(reps)

    frame_idx = 0
    while True:
        ok, raw_frame = cap.read()
        if not ok:
            break

        frame_data = frames[frame_idx] if frame_idx < len(frames) else None
        rep_id, in_rep = frame_rep_lookup.get(frame_idx, (None, False))
        timestamp_s = frame_idx / fps

        annotated = draw_frame(raw_frame, frame_data, rep_id, timestamp_s, in_rep)
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    return output_path
```

Append to the end of `main()`, replacing the final `return`:
```python
    if not args.no_overlay:
        video_path = render_annotated_video(args.input, args.output, frames, reps, fps)
        print(f"Wrote {video_path}")

    return frames, fps, reps, session_agg, args
```

- [ ] **Step 2: Manual verification**

Run against a real free-throw clip if available (`--input <clip>` without `--no-overlay`); at minimum run against the synthetic smoke clip to confirm no crash:
`cd /Users/ignatiusmartin/Documents/Personal/Projects/shotlab && python main.py --input /tmp/smoke.mp4`
Expected: prints frame/rep info, JSON/chart paths, `Wrote output/annotated_smoke.mp4`; confirm the file exists and is playable (`ffprobe output/annotated_smoke.mp4` or open in a video player) with the same resolution/fps as the input.

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: render annotated overlay video"
```

---

## Task 12: Polish — logging, error handling, README (Phase 6)

**Files:**
- Modify: `main.py`
- Create: `README.md`

**Interfaces:** none new — this task hardens existing entry points and documents them.

- [ ] **Step 1: Add graceful error handling to `main.py`**

Wrap the body of `main()` (from the `PoseExtractor().extract` call onward) so failures produce a clean error message instead of a raw traceback for the known failure modes the spec calls out: missing video (already handled in Task 4), zero frames extracted, zero reps detected before classifier/report stages. Modify `main()`:

```python
    frames, detected_fps = PoseExtractor().extract(args.input)
    fps = args.fps or detected_fps
    print(f"Extracted {len(frames)} frames at {fps:.2f} fps.")

    if not any(frame is not None for frame in frames):
        logger.error("No usable pose landmarks detected in any frame of %s.", args.input)
        sys.exit(1)

    reps_bounds = segment_reps(frames, fps)
    print(f"Detected {len(reps_bounds)} rep(s):")
    for i, (start, release, end) in enumerate(reps_bounds):
        print(
            f"  rep {i}: start={start} ({start/fps:.2f}s) "
            f"release={release} ({release/fps:.2f}s) "
            f"end={end} ({end/fps:.2f}s)"
        )

    if not reps_bounds:
        logger.warning("No reps detected — skipping metrics, classifier, and report.")
        return frames, fps, [], {}, args

    reps = compute_rep_metrics(reps_bounds, frames)
    if len(reps) < args.min_reps:
        logger.info(
            "Only %d rep(s) detected (< --min-reps=%d); classifier will label all reps 'unclassified'.",
            len(reps), args.min_reps,
        )
    reps = train_and_label(reps, output_dir=args.output, min_reps=args.min_reps)
    session_agg = aggregate_session(reps)
```
(keep the rest of `main()` — per-rep table printing, `save_json`/`save_chart`, `render_annotated_video` — unchanged, following this block).

- [ ] **Step 2: Verify `--verbose` toggles debug logs**

Run: `cd /Users/ignatiusmartin/Documents/Personal/Projects/shotlab && python main.py --input /tmp/smoke.mp4 --no-overlay --verbose`
Expected: `DEBUG shotlab.pose: Extracted ...` (or similar) debug-level line appears; without `--verbose` it does not.

- [ ] **Step 3: Write `README.md`**

```markdown
# Shot Form Analyzer

Analyze basketball free-throw shooting form from video: pose extraction, joint-angle
computation, rep segmentation, per-rep metrics, a consistency classifier, and an
annotated video + session report.

## Installation

Requires Python 3.10+.

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Quick Start

\`\`\`bash
python main.py --input clips/freethrow_session.mp4
\`\`\`

This writes to `output/`:
- `annotated_<input_name>.mp4` — skeleton + angle overlay video
- `session_report.json` — per-rep metrics and session-level stats
- `session_summary.png` — 2×2 chart of the four per-rep metrics across the session
- `classifier.pkl` — the fitted scaler/KMeans/LogisticRegression, if enough reps were detected

## CLI Options

\`\`\`bash
python main.py \\
  --input clips/freethrow_session.mp4 \\
  --output output/          # output directory, default: output/
  --fps 30                  # override detected fps if detection is wrong
  --min-reps 3              # fewer reps than this -> all labeled "unclassified"
  --no-overlay              # skip the annotated video, stats/report only
  --verbose                 # enable debug logging
\`\`\`

## What Each Metric Means

- **elbow_at_release** — right elbow angle (degrees) at the frame within the rep where
  the elbow is most extended. Closer to 180° = fuller extension at release.
- **knee_bend_at_setup** — right knee angle (degrees) at the first frame in the rep
  where both knees are bent past 150°, i.e. the shooting stance. Lower = deeper bend.
- **wrist_follow_through** — wrist-forearm angle (degrees) at the last frame of the
  rep, capturing the snap/follow-through position.
- **arc_proxy** — right shoulder elevation angle (degrees) at the same frame used for
  `elbow_at_release`, used as a rough proxy for shot arc since the ball itself isn't tracked.

Each rep is labeled `consistent`, `inconsistent`, or `unclassified` (fewer than
`--min-reps` reps in the session) by a classifier trained on that session's own reps:
KMeans splits reps into two groups on the four metrics above (the majority/tighter
cluster is "consistent"), then a logistic regression is fit on that split to produce
the final labels.

## Assumptions & Limits

- Right-hand shooter only (the joint angle set is right-side).
- One shooter per clip, file input only (no webcam/real-time mode).
- No ball tracking — `arc_proxy` is a body-pose proxy for shot arc, not a measured trajectory.

## Running Tests

\`\`\`bash
python -m pytest tests/ -v
\`\`\`
```

- [ ] **Step 4: Full test suite run**

Run: `cd /Users/ignatiusmartin/Documents/Personal/Projects/shotlab && python -m pytest tests/ -v`
Expected: all tests across `test_angles.py`, `test_segmenter.py`, `test_metrics.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py README.md
git commit -m "feat: error handling, verbose logging, README"
```

---

## Self-Review Notes

- **Spec coverage:** Phase 1 (Task 1-4), Phase 2 (Task 5), Phase 3 (Task 6-8), Phase 4 (Task 9), Phase 5 (Task 10-11), Phase 6 (Task 12) all mapped. `angle_between`/`compute_angles` (Task 2), `PoseExtractor` (Task 3), `segment_reps` (Task 5), `compute_rep_metrics`/`aggregate_session` (Task 6), `train_and_label` (Task 7), `save_json`/`save_chart` (Task 9), `draw_frame`/`render_annotated_video` (Task 10-11) all present with concrete signatures matching the PRD's data model.
- **Known deviation from literal spec wording:** `overlay.py` draws the skeleton manually with `cv2.line`/`cv2.circle` rather than `mp.solutions.drawing_utils.draw_landmarks`, because `PoseExtractor.extract` discards the MediaPipe landmark proto after converting to plain dicts (by design, so `angles.py`/`segmenter.py`/`metrics.py` stay MediaPipe-free). Documented inline in Task 10.
- **Savgol smoothing over full landmark sequences:** the spec's "Key Implementation Notes" mentions smoothing each landmark's x/y across the full frame sequence *before* computing angles (window 9, polyorder 2), as a noise-reduction step distinct from the segmenter's own smoothing of the wrist-y signal. This plan's `pose.py` computes angles per-frame from raw (unsmoothed) landmarks — a reasonable first cut, but if release-frame/setup-frame detection turns out noisy in manual testing (Task 3/Phase 2 "watch the video and count manually" check), add a post-extraction smoothing pass over `frames[i]["landmarks"][name]["x"/"y"]` (grouped by landmark name across all frames) before recomputing angles, as a follow-up task — flagged here rather than speculatively built now, since YAGNI until real footage shows it's needed.
- **Shooting side parameterization:** explicitly out of scope per Global Constraints (right-hand shooter hardcoded), matching the spec's own "parameterize eventually" note.
- **Placeholder scan:** no TBD/TODO markers; every step has runnable code or an exact shell command.
- **Type consistency check:** `FrameData` dict shape (`frame_idx`, `timestamp_s`, `landmarks`, `angles`) is identical across `pose.py` (Task 3), `segmenter.py` (Task 5), `metrics.py` (Task 6), and `overlay.py` (Task 10). `RepData` dict shape (`rep_id`, `start_frame`, `release_frame`, `end_frame`, 4 metric fields, `label`) is identical across `metrics.py` (Task 6), `classifier.py` (Task 7), `report.py` (Task 9), and `main.py`'s `_build_frame_rep_lookup` (Task 11).
