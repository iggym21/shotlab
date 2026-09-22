from analyzer.reference_ranges import flag_out_of_range, REFERENCE_RANGES


def test_flag_out_of_range_returns_empty_when_all_within_bounds():
    rep = {
        "elbow_at_release": 175.0,
        "knee_bend_at_setup": 120.0,
        "wrist_follow_through": 75.0,
        "arc_proxy": 90.0,
        "elbow_flare": 0.05,
    }
    assert flag_out_of_range(rep) == []


def test_flag_out_of_range_flags_values_below_and_above_band():
    rep = {
        "elbow_at_release": 90.0,    # below (160, 180)
        "knee_bend_at_setup": 120.0,
        "wrist_follow_through": 200.0,  # above (60, 90)
        "arc_proxy": 90.0,
        "elbow_flare": 0.05,
    }
    flags = flag_out_of_range(rep)
    assert "elbow_at_release" in flags
    assert "wrist_follow_through" in flags
    assert "knee_bend_at_setup" not in flags


def test_flag_out_of_range_skips_missing_metrics():
    rep = {"elbow_at_release": 175.0}
    assert flag_out_of_range(rep) == []


def test_flag_out_of_range_boundary_values_are_inclusive():
    low, high = REFERENCE_RANGES["arc_proxy"]
    rep = {"arc_proxy": low}
    assert "arc_proxy" not in flag_out_of_range(rep)
    rep = {"arc_proxy": high}
    assert "arc_proxy" not in flag_out_of_range(rep)
