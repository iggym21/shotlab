"""Per-rep metric extraction and session-level aggregation."""
import logging

import numpy as np

from analyzer.reference_ranges import flag_out_of_range

logger = logging.getLogger(__name__)

KNEE_BENT_THRESHOLD_DEG = 150.0
LOW_CONFIDENCE_THRESHOLD = 0.7

METRIC_FIELDS = (
    "elbow_at_release",
    "knee_bend_at_setup",
    "wrist_follow_through",
    "arc_proxy",
    "elbow_flare",
)


def _find_most_extended_elbow_frame(frames, start, end, elbow_joint):
    best_idx = None
    best_angle = -1.0
    for i in range(start, end + 1):
        frame = frames[i]
        if frame is None:
            continue
        angle = frame["angles"].get(elbow_joint)
        if angle is not None and angle > best_angle:
            best_angle = angle
            best_idx = i
    return best_idx


def _find_setup_frame(frames, start, end):
    for i in range(start, end + 1):
        frame = frames[i]
        if frame is None:
            continue
        knee_r = frame["angles"].get("knee_right")
        knee_l = frame["angles"].get("knee_left")
        if knee_r is not None and knee_l is not None:
            if knee_r <= KNEE_BENT_THRESHOLD_DEG and knee_l <= KNEE_BENT_THRESHOLD_DEG:
                return i
    return start


def _safe_angle(frames, idx, angle_name, rep_id):
    frame = frames[idx] if 0 <= idx < len(frames) else None
    if frame is not None:
        value = frame["angles"].get(angle_name)
        if value is not None:
            return value
    logger.warning("rep %d: missing %s at frame %d, defaulting to 0.0", rep_id, angle_name, idx)
    return 0.0


def _compute_elbow_flare(frames, idx, shooting_side, rep_id):
    """Signed horizontal offset of the elbow from the shoulder-wrist line at idx,
    normalized by shoulder width. ~0 = elbow tucked on the line; larger magnitude =
    more flare. Falls back to 0.0 (with a warning) if any required point is missing.
    """
    frame = frames[idx] if 0 <= idx < len(frames) else None
    if frame is None:
        logger.warning("rep %d: missing frame %d for elbow_flare, defaulting to 0.0", rep_id, idx)
        return 0.0

    landmarks = frame["landmarks"]
    shoulder = landmarks.get(f"{shooting_side}_shoulder")
    elbow = landmarks.get(f"{shooting_side}_elbow")
    wrist = landmarks.get(f"{shooting_side}_wrist")
    left_shoulder = landmarks.get("left_shoulder")
    right_shoulder = landmarks.get("right_shoulder")

    if any(pt is None for pt in (shoulder, elbow, wrist, left_shoulder, right_shoulder)):
        logger.warning("rep %d: missing landmarks for elbow_flare at frame %d, defaulting to 0.0", rep_id, idx)
        return 0.0

    shoulder_width = abs(right_shoulder["x"] - left_shoulder["x"])
    if shoulder_width < 1e-6:
        logger.warning("rep %d: degenerate shoulder width at frame %d, defaulting elbow_flare to 0.0", rep_id, idx)
        return 0.0

    dy = wrist["y"] - shoulder["y"]
    if abs(dy) < 1e-9:
        line_x = shoulder["x"]
    else:
        t = (elbow["y"] - shoulder["y"]) / dy
        line_x = shoulder["x"] + t * (wrist["x"] - shoulder["x"])

    return float((elbow["x"] - line_x) / shoulder_width)


def _compute_confidence(frames, start, end):
    visibilities = []
    for i in range(start, end + 1):
        frame = frames[i]
        if frame is None:
            continue
        visibilities.extend(lm["visibility"] for lm in frame["landmarks"].values())
    if not visibilities:
        return 0.0
    return float(np.mean(visibilities))


def compute_rep_metrics(rep_bounds, frames, shooting_side: str = "right"):
    reps = []
    for rep_id, (start, release, end) in enumerate(rep_bounds):
        elbow_joint = f"elbow_{shooting_side}"
        extended_idx = _find_most_extended_elbow_frame(frames, start, end, elbow_joint)
        if extended_idx is None:
            extended_idx = release

        setup_idx = _find_setup_frame(frames, start, end)

        rep = {
            "rep_id": rep_id,
            "start_frame": start,
            "release_frame": release,
            "end_frame": end,
            "elbow_at_release": _safe_angle(frames, extended_idx, elbow_joint, rep_id),
            "knee_bend_at_setup": _safe_angle(frames, setup_idx, f"knee_{shooting_side}", rep_id),
            "wrist_follow_through": _safe_angle(frames, end, f"wrist_{shooting_side}", rep_id),
            "arc_proxy": _safe_angle(frames, extended_idx, f"shoulder_{shooting_side}", rep_id),
            "elbow_flare": _compute_elbow_flare(frames, extended_idx, shooting_side, rep_id),
            "confidence": _compute_confidence(frames, start, end),
            "label": "unlabeled",
        }
        rep["low_confidence"] = rep["confidence"] < LOW_CONFIDENCE_THRESHOLD
        rep["out_of_range"] = flag_out_of_range(rep)
        reps.append(rep)
    return reps


def aggregate_session(reps):
    if not reps:
        return {}

    agg = {}
    for field in METRIC_FIELDS:
        values = np.array([rep[field] for rep in reps], dtype=float)
        agg[field] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "rep_to_rep_delta": np.diff(values).tolist(),
        }
    return agg
