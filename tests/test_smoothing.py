import numpy as np
import pytest

from analyzer.smoothing import smooth_landmarks


def _frame(x, y, visibility=1.0):
    return {"pt": {"x": x, "y": y, "z": 0.0, "visibility": visibility}}


def test_smooth_landmarks_preserves_length_and_none_positions():
    frames = [_frame(0.1, 0.1), None, _frame(0.3, 0.3), _frame(0.4, 0.4), None]
    result = smooth_landmarks(frames)
    assert len(result) == 5
    assert result[1] is None
    assert result[4] is None
    assert result[0] is not None
    assert result[2] is not None


def test_smooth_landmarks_reduces_noise_variance():
    rng = np.random.default_rng(0)
    n = 60
    t = np.linspace(0, 1, n)
    clean_x = t
    noisy_x = clean_x + rng.normal(0, 0.05, n)

    frames = [_frame(x, 0.5) for x in noisy_x]
    result = smooth_landmarks(frames)
    smoothed_x = np.array([f["pt"]["x"] for f in result])

    raw_error = np.sum((noisy_x - clean_x) ** 2)
    smoothed_error = np.sum((smoothed_x - clean_x) ** 2)
    assert smoothed_error < raw_error


def test_smooth_landmarks_keeps_z_and_visibility_untouched():
    frames = [_frame(0.1, 0.1, visibility=0.87), _frame(0.2, 0.2, visibility=0.42)]
    result = smooth_landmarks(frames)
    assert result[0]["pt"]["visibility"] == pytest.approx(0.87)
    assert result[1]["pt"]["visibility"] == pytest.approx(0.42)
    assert result[0]["pt"]["z"] == pytest.approx(0.0)


def test_smooth_landmarks_empty_list():
    assert smooth_landmarks([]) == []


def test_smooth_landmarks_all_none():
    frames = [None, None, None]
    result = smooth_landmarks(frames)
    assert result == [None, None, None]


def test_smooth_landmarks_short_sequence_does_not_crash():
    frames = [_frame(0.1, 0.1), _frame(0.2, 0.2)]
    result = smooth_landmarks(frames)
    assert len(result) == 2
    assert all(f is not None for f in result)
