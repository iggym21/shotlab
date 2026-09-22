# Shot Form Analyzer

Analyze basketball free-throw shooting form from video: pose extraction, joint-angle
computation, rep segmentation, per-rep metrics, a consistency classifier, and an
annotated video + session report.

## Installation

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependency versions are pinned in `requirements.txt` to a combination that's actually
been verified to resolve and import together (mediapipe's newest releases dropped the
legacy `mp.solutions.pose` API this project uses, and unpinned opencv/numpy can drift
out of sync with mediapipe's own numpy ceiling — don't loosen the pins without
re-verifying `python -c "import cv2, mediapipe; print(mediapipe.solutions.pose)"`).

## Quick Start

```bash
python main.py --input clips/freethrow_session.mp4
```

This writes to `output/`:
- `annotated_<input_name>.mp4` — skeleton + angle overlay video
- `session_report.json` — per-rep metrics and session-level stats
- `session_summary.png` — 2×2 chart of the four per-rep metrics across the session
- `classifier.pkl` — the fitted scaler/KMeans/LogisticRegression, if enough reps were detected

## CLI Options

```bash
python main.py \
  --input clips/freethrow_session.mp4 \
  --output output/          # output directory, default: output/
  --fps 30                  # override detected fps if detection is wrong
  --min-reps 3              # fewer reps than this -> all labeled "unclassified"
  --no-overlay               # skip the annotated video, stats/report only
  --verbose                  # enable debug logging
```

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

## Example Output

```
Extracted 412 frames at 29.97 fps.
Detected 5 rep(s):
  rep 0: start=18 (0.60s) release=34 (1.13s) end=55 (1.84s)
  rep 1: start=90 (3.00s) release=108 (3.60s) end=131 (4.37s)
  ...

rep_id |   elbow |    knee |   wrist |     arc | label
     0 |   172.3 |   138.1 |    88.4 |    91.2 | consistent
     1 |   169.8 |   142.6 |    85.1 |    89.7 | consistent
     2 |   140.5 |   170.2 |    60.3 |    70.1 | inconsistent
     ...

Wrote output/session_report.json
Wrote output/session_summary.png
Wrote output/annotated_freethrow_session.mp4
```

## Assumptions & Limits

- Right-hand shooter only (the joint angle set is right-side).
- One shooter per clip, file input only (no webcam/real-time mode).
- No ball tracking — `arc_proxy` is a body-pose proxy for shot arc, not a measured trajectory.
- Multi-person video is not supported; MediaPipe tracks a single detected pose per frame.

## Running Tests

```bash
python -m pytest tests/ -v
```
