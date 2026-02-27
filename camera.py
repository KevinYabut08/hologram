import cv2
import mediapipe as mp
import time

class Camera:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        
        self.mp_draw = mp.solutions.drawing_utils
        
        # FPS tracking
        self.prev_time = time.time()
        self.fps = 0
    
    def get_frame_and_hands(self):  # This is the correct method name
        success, frame = self.cap.read()
        if not success:
            return None, None, None
        
        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands
        results = self.hands.process(rgb_frame)
        
        left_hand = None
        right_hand = None
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Assign to left or right
                if hand_label == "Left":
                    left_hand = hand_landmarks
                else:
                    right_hand = hand_landmarks
        
        # Calculate FPS
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time) if current_time != self.prev_time else 0
        self.prev_time = current_time
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {int(self.fps)}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, left_hand, right_hand
    
    def release(self):
        self.cap.release()