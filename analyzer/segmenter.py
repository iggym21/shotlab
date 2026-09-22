"""Rep segmentation from shooting-wrist vertical trajectory."""
import numpy as np
from scipy.signal import find_peaks, savgol_filter


def _extract_wrist_y(frames):
    idx = np.arange(len(frames))
    y = np.full(len(frames), np.nan)
    for i, frame in enumerate(frames):
        if frame is not None:
            y[i] = frame["landmarks"]["right_wrist"]["y"]

    valid = ~np.isnan(y)
    if not valid.any():
        raise ValueError("No frames with visible right_wrist to segment reps from.")
    if not valid.all():
        y = np.interp(idx, idx[valid], y[valid])
    return y


def _smooth(y):
    window = min(9, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if window < 5:
        return y.copy()
    return savgol_filter(y, window_length=window, polyorder=2)


def _nearest_preceding_peak(peaks_idx, target, default):
    candidates = peaks_idx[peaks_idx < target]
    return int(candidates.max()) if len(candidates) else default


def _nearest_following_peak(peaks_idx, target, default):
    candidates = peaks_idx[peaks_idx > target]
    return int(candidates.min()) if len(candidates) else default


def segment_reps(frames, fps, min_rep_duration_s: float = 0.5, min_gap_s: float = 0.3):
    if len(frames) == 0:
        return []

    y = _extract_wrist_y(frames)
    y_smooth = _smooth(y)

    min_gap_frames = max(1, int(round(min_gap_s * fps)))

    troughs, _ = find_peaks(-y_smooth, distance=min_gap_frames)
    peaks, _ = find_peaks(y_smooth, distance=min_gap_frames)

    reps = []
    for release in troughs:
        start = _nearest_preceding_peak(peaks, release, default=0)
        end = _nearest_following_peak(peaks, release, default=len(y_smooth) - 1)

        duration_s = (end - start) / fps
        if duration_s < min_rep_duration_s:
            continue

        reps.append((int(start), int(release), int(end)))

    reps.sort(key=lambda r: r[0])
    return reps
