"""Cross-frame landmark coordinate smoothing (noise reduction before angle math)."""
import numpy as np
from scipy.signal import savgol_filter

SMOOTH_WINDOW = 9
SMOOTH_POLYORDER = 2


def _smooth_series(values: np.ndarray) -> np.ndarray:
    window = min(SMOOTH_WINDOW, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if window <= SMOOTH_POLYORDER:
        return values.copy()
    return savgol_filter(values, window_length=window, polyorder=SMOOTH_POLYORDER)


def smooth_landmarks(frames):
    """frames: list[dict[name -> {x,y,z,visibility}] | None].

    Smooths each landmark's x/y coordinate across the full frame sequence with a
    Savitzky-Golay filter. None frames (no pose detected) stay None and are never
    populated; gaps they create are linearly interpolated only so neighboring
    valid frames smooth correctly. z and visibility pass through unchanged.
    """
    n = len(frames)
    if n == 0:
        return []

    landmark_names = set()
    for frame in frames:
        if frame is not None:
            landmark_names.update(frame.keys())

    if not landmark_names:
        return list(frames)

    idx = np.arange(n)

    smoothed_coords = {}
    for name in landmark_names:
        x = np.full(n, np.nan)
        y = np.full(n, np.nan)
        for i, frame in enumerate(frames):
            if frame is not None and name in frame:
                x[i] = frame[name]["x"]
                y[i] = frame[name]["y"]

        present = ~np.isnan(x)
        if present.sum() < 2:
            smoothed_coords[name] = (x, y)
            continue

        x_filled = np.interp(idx, idx[present], x[present])
        y_filled = np.interp(idx, idx[present], y[present])

        smoothed_coords[name] = (_smooth_series(x_filled), _smooth_series(y_filled))

    result = []
    for i, frame in enumerate(frames):
        if frame is None:
            result.append(None)
            continue
        new_frame = {}
        for name, lm in frame.items():
            x_smooth, y_smooth = smoothed_coords[name]
            new_frame[name] = {
                "x": float(x_smooth[i]),
                "y": float(y_smooth[i]),
                "z": lm["z"],
                "visibility": lm["visibility"],
            }
        result.append(new_frame)
    return result
