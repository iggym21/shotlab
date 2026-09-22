"""Per-rep metric extraction and session-level aggregation."""
import logging

import numpy as np

logger = logging.getLogger(__name__)

KNEE_BENT_THRESHOLD_DEG = 150.0

METRIC_FIELDS = (
    "elbow_at_release",
    "knee_bend_at_setup",
    "wrist_follow_through",
    "arc_proxy",
)


def _find_most_extended_elbow_frame(frames, start, end):
    best_idx = None
    best_angle = -1.0
    for i in range(start, end + 1):
        frame = frames[i]
        if frame is None:
            continue
        angle = frame["angles"].get("elbow_right")
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


def compute_rep_metrics(rep_bounds, frames):
    reps = []
    for rep_id, (start, release, end) in enumerate(rep_bounds):
        extended_idx = _find_most_extended_elbow_frame(frames, start, end)
        if extended_idx is None:
            extended_idx = release

        setup_idx = _find_setup_frame(frames, start, end)

        reps.append({
            "rep_id": rep_id,
            "start_frame": start,
            "release_frame": release,
            "end_frame": end,
            "elbow_at_release": _safe_angle(frames, extended_idx, "elbow_right", rep_id),
            "knee_bend_at_setup": _safe_angle(frames, setup_idx, "knee_right", rep_id),
            "wrist_follow_through": _safe_angle(frames, end, "wrist_right", rep_id),
            "arc_proxy": _safe_angle(frames, extended_idx, "shoulder_right", rep_id),
            "label": "unlabeled",
        })
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
