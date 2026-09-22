"""Generic free-throw form reference ranges, used to flag reps worth a second look.

These are reasonable starting bands, not a coaching standard — retune per shooter
if they don't match what you're seeing.
"""

REFERENCE_RANGES = {
    "elbow_at_release": (160.0, 180.0),
    "knee_bend_at_setup": (100.0, 140.0),
    "wrist_follow_through": (60.0, 90.0),
    "arc_proxy": (75.0, 100.0),
    "elbow_flare": (-0.15, 0.15),
}


def flag_out_of_range(rep: dict) -> list:
    """Returns the metric names in rep whose value falls outside its reference band."""
    flags = []
    for metric, (low, high) in REFERENCE_RANGES.items():
        value = rep.get(metric)
        if value is None:
            continue
        if value < low or value > high:
            flags.append(metric)
    return flags
