# camera.py
import cv2
import time
import mediapipe as mp

class Camera:
    def __init__(self, camera_index=0):
        # Use DirectShow on Windows to avoid long-open delay
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # MediaPipe Solutions (present in mediapipe==0.10.7)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        self.drawer = mp.solutions.drawing_utils
        self.connections = self.mp_hands.HAND_CONNECTIONS

        # FPS tracking
        self.prev_time = time.time()
        self.fps = 0

    def get_frame_and_hands(self):
        """
        Returns a tuple: (frame_bgr, left_hand_landmarks, right_hand_landmarks)
        - frame_bgr: np.ndarray or None
        - left/right: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList or None
        This function MUST always return a 3-tuple.
        """
        try:
            ok, frame = self.cap.read()
            if not ok:
                return None, None, None

            # Mirror view + to RGB
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            res = self.hands.process(rgb)

            left_hand = None
            right_hand = None

            if res.multi_hand_landmarks and res.multi_handedness:
                for lm, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label = handedness.classification[0].label  # "Left" or "Right"
                    # Draw landmarks for user feedback
                    self.drawer.draw_landmarks(frame, lm, self.connections)
                    if label == "Left":
                        left_hand = lm
                    else:
                        right_hand = lm

            # FPS overlay
            now = time.time()
            self.fps = 1 / (now - self.prev_time) if now != self.prev_time else 0
            self.prev_time = now
            cv2.putText(frame, f"FPS: {int(self.fps)}", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            return frame, left_hand, right_hand

        except Exception:
            # On any unexpected error, do NOT break the caller with None;
            # return a safe triple to keep the outer loop alive.
            return None, None, None

    def release(self):
        # Close MediaPipe resources and release the camera
        try:
            self.hands.close()
        except Exception:
            pass
        if self.cap:
            self.cap.release()