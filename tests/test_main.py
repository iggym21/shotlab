import json

import cv2
import numpy as np
import pytest

import main as main_module


def _write_blank_video(path, n_frames=60, fps=30, size=(320, 240)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for _ in range(n_frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype="uint8"))
    writer.release()


def _synthetic_wrist_y(n_reps=2, frames_per_rep=30, high_y=0.8, low_y=0.2):
    y = []
    half = frames_per_rep // 2
    for _ in range(n_reps):
        down = np.linspace(high_y, low_y, half)
        up = np.linspace(low_y, high_y, frames_per_rep - half)
        y.extend(down.tolist())
        y.extend(up.tolist())
    return y


def _synthetic_frames(n_reps=2, frames_per_rep=30, fps=30.0):
    wrist_y = _synthetic_wrist_y(n_reps, frames_per_rep)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-3.0, 3.0, size=len(wrist_y))
    peak_pos = frames_per_rep * 0.6  # interior peak, away from the shared rep boundary frame
    frames = []
    for i, y in enumerate(wrist_y):
        pos_in_cycle = i % frames_per_rep
        knee = 140.0 if pos_in_cycle < frames_per_rep // 3 else 170.0
        elbow = 100.0 + 78.0 * np.exp(-((pos_in_cycle - peak_pos) ** 2) / (2 * 6.0 ** 2)) + jitter[i]
        frames.append({
            "frame_idx": i,
            "timestamp_s": i / fps,
            "landmarks": {
                "left_shoulder": {"x": 0.3, "y": 0.3, "z": 0.0, "visibility": 1.0},
                "right_shoulder": {"x": 0.4, "y": 0.3, "z": 0.0, "visibility": 1.0},
                "right_elbow": {"x": 0.5, "y": 0.4, "z": 0.0, "visibility": 1.0},
                "right_wrist": {"x": 0.5, "y": y, "z": 0.0, "visibility": 1.0},
                "right_hip": {"x": 0.4, "y": 0.6, "z": 0.0, "visibility": 1.0},
                "right_knee": {"x": 0.4, "y": 0.8, "z": 0.0, "visibility": 1.0},
                "right_ankle": {"x": 0.4, "y": 0.95, "z": 0.0, "visibility": 1.0},
            },
            "angles": {
                "elbow_right": elbow,
                "knee_right": knee,
                "knee_left": knee,
                "wrist_right": 70.0 + pos_in_cycle,
                "shoulder_right": 80.0 + pos_in_cycle * 0.3,
                "hip_right": 90.0,
                "elbow_left": None,
            },
        })
    return frames


class _StubPoseExtractor:
    def __init__(self, model_complexity=1):
        pass

    def extract(self, video_path):
        return _synthetic_frames(), 30.0


def test_main_end_to_end_writes_all_artifacts(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    _write_blank_video(video_path)
    output_dir = tmp_path / "output"

    monkeypatch.setattr(main_module, "PoseExtractor", _StubPoseExtractor)

    frames, fps, reps, session_agg, args = main_module.main([
        "--input", str(video_path),
        "--output", str(output_dir),
        "--min-reps", "2",
    ])

    assert len(reps) == 2
    assert set(r["label"] for r in reps) <= {"consistent", "inconsistent"}
    assert set(session_agg.keys()) == {
        "elbow_at_release", "knee_bend_at_setup", "wrist_follow_through", "arc_proxy", "elbow_flare",
    }

    assert (output_dir / "session_report.json").exists()
    assert (output_dir / "session_report.csv").exists()
    assert (output_dir / "session_summary.png").exists()
    assert (output_dir / "session_history.json").exists()
    assert (output_dir / "session_trend.png").exists()
    assert (output_dir / "classifier.pkl").exists()
    assert (output_dir / "annotated_clip.mp4").exists()

    with open(output_dir / "session_report.json") as f:
        report = json.load(f)
    assert len(report["reps"]) == 2

    annotated = cv2.VideoCapture(str(output_dir / "annotated_clip.mp4"))
    assert annotated.get(cv2.CAP_PROP_FRAME_COUNT) == 60
    annotated.release()


def test_main_no_overlay_skips_video(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    _write_blank_video(video_path)
    output_dir = tmp_path / "output"

    monkeypatch.setattr(main_module, "PoseExtractor", _StubPoseExtractor)

    main_module.main([
        "--input", str(video_path),
        "--output", str(output_dir),
        "--min-reps", "2",
        "--no-overlay",
    ])

    assert (output_dir / "session_report.json").exists()
    assert not (output_dir / "annotated_clip.mp4").exists()


def test_main_missing_input_file_exits(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        main_module.main(["--input", str(tmp_path / "does_not_exist.mp4")])
    assert exc_info.value.code == 1


def test_main_zero_reps_detected_skips_downstream_stages(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    _write_blank_video(video_path, n_frames=10)
    output_dir = tmp_path / "output"

    class _StaticWristPoseExtractor:
        def __init__(self, model_complexity=1):
            pass

        def extract(self, video_path):
            # Wrist never moves -> segmenter finds no peak-to-trough cycle.
            frames = []
            for i in range(10):
                frames.append({
                    "frame_idx": i,
                    "timestamp_s": i / 30.0,
                    "landmarks": {
                        "right_wrist": {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0},
                    },
                    "angles": {},
                })
            return frames, 30.0

    monkeypatch.setattr(main_module, "PoseExtractor", _StaticWristPoseExtractor)

    frames, fps, reps, session_agg, args = main_module.main([
        "--input", str(video_path),
        "--output", str(output_dir),
    ])

    assert reps == []
    assert session_agg == {}
    assert not (output_dir / "session_report.json").exists()
    assert not (output_dir / "annotated_clip.mp4").exists()


def test_main_no_usable_frames_exits(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    _write_blank_video(video_path, n_frames=5)

    class _EmptyPoseExtractor:
        def __init__(self, model_complexity=1):
            pass

        def extract(self, video_path):
            return [None] * 5, 30.0

    monkeypatch.setattr(main_module, "PoseExtractor", _EmptyPoseExtractor)

    with pytest.raises(SystemExit) as exc_info:
        main_module.main(["--input", str(video_path), "--output", str(tmp_path / "out")])
    assert exc_info.value.code == 1
