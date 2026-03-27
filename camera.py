# camera.py
import cv2
import mediapipe as mp

class Camera:
    def __init__(self, src=0, width=1280, height=720, mirror=True):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.mirror = mirror
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

    def get_frame_and_landmarks(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, None, None

        if self.mirror:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        left_lm, right_lm = None, None
        hands = results.multi_hand_landmarks or []

        if len(hands) == 1:
            # Single hand: choose side by x position in the (possibly mirrored) image
            h1 = hands[0]
            x1 = h1.landmark[0].x  # wrist x
            # x in [0..1] left->right. If mirrored, the image already reflects your view,
            # so "visually left" truly has smaller x.
            if x1 <= 0.5:
                left_lm = h1
            else:
                right_lm = h1

        elif len(hands) >= 2:
            # Assign by x ordering (visually leftmost is left hand)
            # Sort by wrist x
            sorted_hands = sorted(hands, key=lambda hlm: hlm.landmark[0].x)
            left_lm, right_lm = sorted_hands[0], sorted_hands[1]

        return rgb, left_lm, right_lm

    def release(self):
        try:
            self.hands.close()
        except Exception:
            pass
        if self.cap and self.cap.isOpened():
            self.cap.release()
