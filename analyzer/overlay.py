"""OpenCV drawing: skeleton overlay + angle callouts + HUD text."""
import cv2

BONES = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_index"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_index"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)

# joint_name in analyzer.angles.JOINT_LANDMARKS -> landmark to anchor the text label on
ANGLE_LABEL_ANCHOR = {
    "elbow_right": "right_elbow",
    "elbow_left": "left_elbow",
    "knee_right": "right_knee",
    "knee_left": "left_knee",
    "wrist_right": "right_wrist",
    "hip_right": "right_hip",
    "shoulder_right": "right_shoulder",
}

SKELETON_COLOR_IN_REP = (0, 200, 0)
SKELETON_COLOR_OUT_OF_REP = (160, 160, 160)
JOINT_RADIUS = 4


def _to_pixel(landmark, width, height):
    return int(landmark["x"] * width), int(landmark["y"] * height)


def _draw_text_with_outline(frame, text, org, scale=0.5, color=(255, 255, 255)):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_frame(frame, frame_data, rep_id, timestamp_s: float, in_rep: bool):
    out = frame.copy()
    height, width = out.shape[:2]

    header = f"t={timestamp_s:.2f}s"
    if rep_id is not None:
        header = f"rep {rep_id} | {header}"
    _draw_text_with_outline(out, header, (10, 24), scale=0.6)

    if frame_data is None:
        return out

    landmarks = frame_data["landmarks"]
    color = SKELETON_COLOR_IN_REP if in_rep else SKELETON_COLOR_OUT_OF_REP

    for name_a, name_b in BONES:
        lm_a = landmarks.get(name_a)
        lm_b = landmarks.get(name_b)
        if lm_a is None or lm_b is None:
            continue
        cv2.line(out, _to_pixel(lm_a, width, height), _to_pixel(lm_b, width, height), color, 2)

    for lm in landmarks.values():
        cv2.circle(out, _to_pixel(lm, width, height), JOINT_RADIUS, color, -1)

    for joint_name, angle in frame_data["angles"].items():
        if angle is None:
            continue
        anchor_name = ANGLE_LABEL_ANCHOR.get(joint_name)
        anchor = landmarks.get(anchor_name)
        if anchor is None:
            continue
        px, py = _to_pixel(anchor, width, height)
        _draw_text_with_outline(out, f"{angle:.0f}", (px + 6, py - 6))

    return out
