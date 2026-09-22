"""Pure joint-angle geometry. No MediaPipe/OpenCV dependency."""
import numpy as np

VISIBILITY_THRESHOLD = 0.5

# joint_name -> (landmark_a, landmark_b, landmark_c); angle is measured at b.
JOINT_LANDMARKS = {
    "elbow_right": ("right_shoulder", "right_elbow", "right_wrist"),
    "elbow_left": ("left_shoulder", "left_elbow", "left_wrist"),
    "knee_right": ("right_hip", "right_knee", "right_ankle"),
    "knee_left": ("left_hip", "left_knee", "left_ankle"),
    "wrist_right": ("right_elbow", "right_wrist", "right_index"),
    "wrist_left": ("left_elbow", "left_wrist", "left_index"),
    "hip_right": ("right_shoulder", "right_hip", "right_knee"),
    "hip_left": ("left_shoulder", "left_hip", "left_knee"),
    "shoulder_right": ("right_elbow", "right_shoulder", "right_hip"),
    "shoulder_left": ("left_elbow", "left_shoulder", "left_hip"),
}

SHOOTING_SIDES = ("right", "left")


def side_joint_name(joint: str, shooting_side: str) -> str:
    """joint: one of 'elbow', 'knee', 'wrist', 'hip', 'shoulder'. Returns e.g. 'elbow_left'."""
    if shooting_side not in SHOOTING_SIDES:
        raise ValueError(f"shooting_side must be one of {SHOOTING_SIDES}, got {shooting_side!r}")
    return f"{joint}_{shooting_side}"


def angle_between(a, b, c) -> float:
    """Interior angle at b, in degrees, given 3 (x, y) points."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    ba = a - b
    bc = c - b

    angle_ba = np.arctan2(ba[1], ba[0])
    angle_bc = np.arctan2(bc[1], bc[0])

    diff = np.degrees(angle_ba - angle_bc)
    diff = diff % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    return float(diff)


def compute_angles(landmarks: dict) -> dict:
    """landmarks: {name: {x, y, z, visibility}}. Returns {joint_name: degrees|None}."""
    angles = {}
    for joint_name, (name_a, name_b, name_c) in JOINT_LANDMARKS.items():
        pts = []
        visible = True
        for name in (name_a, name_b, name_c):
            lm = landmarks.get(name)
            if lm is None or lm["visibility"] < VISIBILITY_THRESHOLD:
                visible = False
                break
            pts.append((lm["x"], lm["y"]))
        angles[joint_name] = angle_between(*pts) if visible else None
    return angles
