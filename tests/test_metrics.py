import pytest

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
    assert rep["elbow_flare"] == pytest.approx(0.0)  # no landmarks in fixture -> safe fallback
    assert rep["confidence"] == pytest.approx(0.0)
    assert rep["low_confidence"] is True
    assert rep["out_of_range"] == []


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
        {"elbow_at_release": 170, "knee_bend_at_setup": 140, "wrist_follow_through": 80, "arc_proxy": 90, "elbow_flare": 0.01},
        {"elbow_at_release": 180, "knee_bend_at_setup": 145, "wrist_follow_through": 85, "arc_proxy": 95, "elbow_flare": 0.02},
        {"elbow_at_release": 175, "knee_bend_at_setup": 142, "wrist_follow_through": 82, "arc_proxy": 92, "elbow_flare": 0.03},
    ]
    agg = aggregate_session(reps)

    assert set(agg.keys()) == {
        "elbow_at_release", "knee_bend_at_setup", "wrist_follow_through", "arc_proxy", "elbow_flare",
    }
    elbow = agg["elbow_at_release"]
    assert elbow["mean"] == pytest.approx((170 + 180 + 175) / 3)
    assert elbow["min"] == pytest.approx(170)
    assert elbow["max"] == pytest.approx(180)
    assert elbow["rep_to_rep_delta"] == pytest.approx([10, -5])


def test_aggregate_session_empty_reps_returns_empty_dict():
    assert aggregate_session([]) == {}


def _lm(x, y, visibility=1.0):
    return {"x": x, "y": y, "z": 0.0, "visibility": visibility}


def test_compute_rep_metrics_elbow_flare_matches_hand_calculation():
    # shoulder-wrist line is vertical (x=0.4); elbow sticks out to x=0.5 at the
    # same height as the shoulder-hip midline -> flare = (0.5-0.4)/shoulder_width.
    landmarks = {
        "left_shoulder": {"x": 0.2, "y": 0.3, "z": 0.0, "visibility": 1.0},  # shoulder_width = 0.2
        "right_shoulder": _lm(0.4, 0.3),
        "right_elbow": _lm(0.5, 0.5),
        "right_wrist": _lm(0.4, 0.7),
    }
    frames = [{
        "frame_idx": 0, "timestamp_s": 0.0, "landmarks": landmarks,
        "angles": {"elbow_right": 178, "knee_right": 140, "knee_left": 140, "wrist_right": 80, "shoulder_right": 90},
    }]
    reps = compute_rep_metrics([(0, 0, 0)], frames)
    # line at elbow's y (0.5) is still x=0.4 (shoulder-wrist line is vertical) -> offset 0.1 / width 0.2
    assert reps[0]["elbow_flare"] == pytest.approx(0.5, abs=1e-6)


def test_compute_rep_metrics_confidence_averages_visibility_in_window():
    def _frame_with_visibility(v):
        return {
            "frame_idx": 0, "timestamp_s": 0.0,
            "landmarks": {"right_wrist": _lm(0.5, 0.5, visibility=v)},
            "angles": {"elbow_right": 170, "knee_right": 140, "knee_left": 140, "wrist_right": 80, "shoulder_right": 90},
        }
    frames = [_frame_with_visibility(0.4), _frame_with_visibility(0.5), _frame_with_visibility(0.6)]
    reps = compute_rep_metrics([(0, 1, 2)], frames)
    assert reps[0]["confidence"] == pytest.approx((0.4 + 0.5 + 0.6) / 3)
    assert reps[0]["low_confidence"] is True  # mean 0.5 < LOW_CONFIDENCE_THRESHOLD (0.7)


def test_compute_rep_metrics_low_confidence_flag_uses_threshold():
    def _frame_with_visibility(v):
        return {
            "frame_idx": 0, "timestamp_s": 0.0,
            "landmarks": {"right_wrist": _lm(0.5, 0.5, visibility=v)},
            "angles": {},
        }
    high_conf_frames = [_frame_with_visibility(0.95)] * 3
    reps = compute_rep_metrics([(0, 1, 2)], high_conf_frames)
    assert reps[0]["low_confidence"] is False


def test_compute_rep_metrics_flags_out_of_range_metrics():
    frames = [_frame(elbow=90, knee_r=140, knee_l=140, wrist=90, shoulder=60)] * 3
    reps = compute_rep_metrics([(0, 1, 2)], frames)
    assert "elbow_at_release" in reps[0]["out_of_range"]


def test_compute_rep_metrics_shooting_side_left_reads_left_joints():
    frames = [{
        "frame_idx": 0, "timestamp_s": 0.0,
        "landmarks": {},
        "angles": {
            "elbow_left": 178, "knee_right": 140, "knee_left": 140,
            "wrist_left": 75, "shoulder_left": 88,
            "elbow_right": None, "wrist_right": None, "shoulder_right": None,
        },
    }] * 3
    reps = compute_rep_metrics([(0, 1, 2)], frames, shooting_side="left")
    assert reps[0]["elbow_at_release"] == pytest.approx(178)
    assert reps[0]["wrist_follow_through"] == pytest.approx(75)
    assert reps[0]["arc_proxy"] == pytest.approx(88)
