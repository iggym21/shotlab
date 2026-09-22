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


def test_segment_reps_empty_frame_list_returns_empty():
    assert segment_reps([], 30.0) == []


def test_segment_reps_rejects_too_short_reps():
    # Single rep spanning only 3 frames at 30fps = 0.1s, well under 0.5s default minimum.
    y = np.array([0.8, 0.5, 0.2, 0.5, 0.8])
    frames = _make_frames(y)
    fps = 30.0
    reps = segment_reps(frames, fps, min_rep_duration_s=0.5, min_gap_s=0.1)
    assert len(reps) == 0
