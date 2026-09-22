"""MediaPipe Pose wrapper: video -> list[FrameData | None]."""
import logging

import cv2
import mediapipe as mp

from analyzer.angles import compute_angles, VISIBILITY_THRESHOLD, SHOOTING_SIDES
from analyzer.smoothing import smooth_landmarks

logger = logging.getLogger(__name__)

_mp_pose = mp.solutions.pose

# MediaPipe enum member name -> our snake_case landmark name.
_ENUM_TO_NAME = {
    "LEFT_SHOULDER": "left_shoulder",
    "RIGHT_SHOULDER": "right_shoulder",
    "LEFT_ELBOW": "left_elbow",
    "RIGHT_ELBOW": "right_elbow",
    "LEFT_WRIST": "left_wrist",
    "RIGHT_WRIST": "right_wrist",
    "LEFT_INDEX": "left_index",
    "RIGHT_INDEX": "right_index",
    "LEFT_HIP": "left_hip",
    "RIGHT_HIP": "right_hip",
    "LEFT_KNEE": "left_knee",
    "RIGHT_KNEE": "right_knee",
    "LEFT_ANKLE": "left_ankle",
    "RIGHT_ANKLE": "right_ankle",
}

LANDMARK_NAMES = {
    _mp_pose.PoseLandmark[enum_name].value: name
    for enum_name, name in _ENUM_TO_NAME.items()
}

# Both knees/hips/ankles are always required (setup-stance detection checks both
# legs regardless of shooting side); both shoulders are required for the
# elbow-flare metric's shoulder-width normalization; only the shooting-side arm
# (elbow/wrist/index) needs to be visible, so the off-hand can be off-frame.
def required_landmarks(shooting_side: str) -> tuple:
    if shooting_side not in SHOOTING_SIDES:
        raise ValueError(f"shooting_side must be one of {SHOOTING_SIDES}, got {shooting_side!r}")
    names = {
        "right_hip", "right_knee", "right_ankle",
        "left_hip", "left_knee", "left_ankle",
        "right_shoulder", "left_shoulder",
        f"{shooting_side}_elbow", f"{shooting_side}_wrist", f"{shooting_side}_index",
    }
    return tuple(sorted(names))


class PoseExtractor:
    def __init__(self, model_complexity: int = 1, shooting_side: str = "right"):
        self.model_complexity = model_complexity
        self.shooting_side = shooting_side
        self.required_landmarks = required_landmarks(shooting_side)

    def extract(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        raw_landmarks = []

        with _mp_pose.Pose(
            model_complexity=self.model_complexity,
            static_image_mode=False,
        ) as pose:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                result = pose.process(frame_rgb)
                raw_landmarks.append(self._extract_raw_landmarks(result))

        cap.release()

        smoothed = smooth_landmarks(raw_landmarks)

        frames = [
            self._build_frame_data(landmarks, frame_idx, fps)
            for frame_idx, landmarks in enumerate(smoothed)
        ]

        logger.debug("Extracted %d frames at %.2f fps", len(frames), fps)
        return frames, fps

    def _extract_raw_landmarks(self, result):
        if not result.pose_landmarks:
            return None

        landmarks = {}
        for lm_idx, name in LANDMARK_NAMES.items():
            lm = result.pose_landmarks.landmark[lm_idx]
            landmarks[name] = {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
            }
        return landmarks

    def _build_frame_data(self, landmarks, frame_idx: int, fps: float):
        if landmarks is None:
            return None

        for required_name in self.required_landmarks:
            lm = landmarks.get(required_name)
            if lm is None or lm["visibility"] < VISIBILITY_THRESHOLD:
                return None

        angles = compute_angles(landmarks)

        return {
            "frame_idx": frame_idx,
            "timestamp_s": frame_idx / fps,
            "landmarks": landmarks,
            "angles": angles,
        }
