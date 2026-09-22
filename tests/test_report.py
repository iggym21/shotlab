import json

from analyzer.metrics import aggregate_session
from analyzer.report import save_json, save_chart


def _reps(n=4):
    return [
        {
            "rep_id": i,
            "start_frame": i * 10,
            "release_frame": i * 10 + 5,
            "end_frame": i * 10 + 9,
            "elbow_at_release": 170 + i,
            "knee_bend_at_setup": 140 + i,
            "wrist_follow_through": 80 + i,
            "arc_proxy": 90 + i,
            "elbow_flare": 0.01 * i,
            "confidence": 0.95,
            "low_confidence": False,
            "out_of_range": [],
            "label": "consistent",
        }
        for i in range(n)
    ]


def test_save_json_writes_expected_structure(tmp_path):
    reps = _reps()
    agg = aggregate_session(reps)
    path = save_json(agg, reps, str(tmp_path))

    assert path == str(tmp_path / "session_report.json")
    with open(path) as f:
        data = json.load(f)

    assert data["reps"] == reps
    assert data["session_summary"] == agg
    assert set(data["session_summary"].keys()) == {
        "elbow_at_release", "knee_bend_at_setup", "wrist_follow_through", "arc_proxy", "elbow_flare",
    }


def test_save_json_handles_empty_reps(tmp_path):
    path = save_json({}, [], str(tmp_path))
    with open(path) as f:
        data = json.load(f)
    assert data == {"reps": [], "session_summary": {}}


def test_save_chart_creates_png_file(tmp_path):
    reps = _reps()
    agg = aggregate_session(reps)
    path = save_chart(reps, agg, str(tmp_path))

    assert path == str(tmp_path / "session_summary.png")
    with open(path, "rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n"


def test_save_chart_handles_empty_reps_without_crashing(tmp_path):
    path = save_chart([], {}, str(tmp_path))
    with open(path, "rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n"


def test_save_json_creates_output_dir_if_missing(tmp_path):
    nested = tmp_path / "nested" / "output"
    path = save_json({}, [], str(nested))
    assert nested.exists()
    assert path == str(nested / "session_report.json")
