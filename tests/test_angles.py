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
