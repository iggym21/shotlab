import json

from analyzer.history import append_session, save_trend_chart


def _session_agg(elbow_mean=175.0):
    return {
        "elbow_at_release": {"mean": elbow_mean, "std": 2.0, "min": 170.0, "max": 180.0, "rep_to_rep_delta": [1.0]},
        "arc_proxy": {"mean": 90.0, "std": 1.0, "min": 88.0, "max": 92.0, "rep_to_rep_delta": [0.5]},
    }


def test_append_session_creates_new_history_file(tmp_path):
    history = append_session(str(tmp_path), "clip.mp4", _session_agg(), [{"rep_id": 0}], timestamp="2026-01-01T00:00:00+00:00")

    assert (tmp_path / "session_history.json").exists()
    assert len(history) == 1
    assert history[0]["input"] == "clip.mp4"
    assert history[0]["n_reps"] == 1
    assert history[0]["timestamp"] == "2026-01-01T00:00:00+00:00"
    assert history[0]["session_summary"]["elbow_at_release"]["mean"] == 175.0


def test_append_session_appends_to_existing_history(tmp_path):
    append_session(str(tmp_path), "clip1.mp4", _session_agg(170.0), [{"rep_id": 0}], timestamp="2026-01-01T00:00:00+00:00")
    history = append_session(str(tmp_path), "clip2.mp4", _session_agg(180.0), [{"rep_id": 0}], timestamp="2026-01-02T00:00:00+00:00")

    assert len(history) == 2
    assert history[0]["input"] == "clip1.mp4"
    assert history[1]["input"] == "clip2.mp4"

    with open(tmp_path / "session_history.json") as f:
        on_disk = json.load(f)
    assert len(on_disk) == 2


def test_append_session_uses_current_time_when_not_given(tmp_path):
    history = append_session(str(tmp_path), "clip.mp4", _session_agg(), [{"rep_id": 0}])
    assert history[0]["timestamp"]  # non-empty, ISO-ish string
    assert "T" in history[0]["timestamp"]


def test_save_trend_chart_with_no_history_does_not_crash(tmp_path):
    path = save_trend_chart([], str(tmp_path))
    with open(path, "rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n"


def test_save_trend_chart_with_multiple_sessions(tmp_path):
    history = append_session(str(tmp_path), "clip1.mp4", _session_agg(170.0), [{"rep_id": 0}], timestamp="2026-01-01T00:00:00+00:00")
    history = append_session(str(tmp_path), "clip2.mp4", _session_agg(180.0), [{"rep_id": 0}], timestamp="2026-01-02T00:00:00+00:00")

    path = save_trend_chart(history, str(tmp_path))
    with open(path, "rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n"


def test_save_trend_chart_handles_sessions_missing_a_metric(tmp_path):
    history = [
        {"timestamp": "t1", "input": "a.mp4", "n_reps": 1, "session_summary": {"elbow_at_release": {"mean": 170.0}}},
        {"timestamp": "t2", "input": "b.mp4", "n_reps": 1, "session_summary": {}},
    ]
    path = save_trend_chart(history, str(tmp_path))
    with open(path, "rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n"
