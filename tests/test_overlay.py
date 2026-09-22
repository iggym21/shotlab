import numpy as np

from analyzer.overlay import draw_frame


def _blank_frame(w=320, h=240):
    return np.zeros((h, w, 3), dtype="uint8")


def _frame_data():
    return {
        "frame_idx": 0,
        "timestamp_s": 1.5,
        "landmarks": {
            "right_shoulder": {"x": 0.4, "y": 0.3, "z": 0.0, "visibility": 1.0},
            "right_elbow": {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0},
            "right_wrist": {"x": 0.6, "y": 0.3, "z": 0.0, "visibility": 1.0},
        },
        "angles": {"elbow_right": 95.0},
    }


def test_draw_frame_returns_same_shape_and_dtype():
    frame = _blank_frame()
    out = draw_frame(frame, _frame_data(), rep_id=2, timestamp_s=1.5, in_rep=True)
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype


def test_draw_frame_does_not_mutate_input():
    frame = _blank_frame()
    original = frame.copy()
    draw_frame(frame, _frame_data(), rep_id=2, timestamp_s=1.5, in_rep=True)
    assert np.array_equal(frame, original)


def test_draw_frame_draws_something_with_pose_data():
    frame = _blank_frame()
    out = draw_frame(frame, _frame_data(), rep_id=0, timestamp_s=0.0, in_rep=True)
    assert out.sum() > 0


def test_draw_frame_handles_none_frame_data():
    frame = _blank_frame()
    out = draw_frame(frame, None, rep_id=None, timestamp_s=0.0, in_rep=False)
    assert out.shape == frame.shape
    # header text still drawn even with no pose data
    assert out.sum() > 0


def test_draw_frame_skips_none_angles_without_crashing():
    frame = _blank_frame()
    frame_data = _frame_data()
    frame_data["angles"]["elbow_right"] = None
    out = draw_frame(frame, frame_data, rep_id=0, timestamp_s=0.0, in_rep=True)
    assert out.shape == frame.shape
