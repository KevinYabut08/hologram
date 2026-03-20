import math
import time
import numpy as np


class ImmersiveCubeController:
    def __init__(self):
        # Physical cube properties
        self.cube_rotation_x = 0.0
        self.cube_rotation_y = 0.0
        self.cube_rotation_z = 0.0

        # Hand tracking (MediaPipe landmarks passed in each frame)
        self.left_hand = None
        self.right_hand = None

        # Physical interaction states
        self.grabbed_face = None       # 'U','D','L','R','F','B' or None
        self.grab_strength = 0.0       # How hard are they grabbing?
        self.twist_angle = 0.0         # Current twist angle (degrees)
        self.snap_threshold = 30.0     # Degrees to snap rotation on release

        # Physical feedback
        self.last_twist_time = 0.0
        self.vibration_intensity = 0.0

        # Cube physics
        self.angular_velocity = [0.0, 0.0, 0.0]
        self.friction = 0.95

        # Zones for grabbing (face centers in cube space)
        self.grab_zones = {
            'U': {'center': (0.0,  1.0,  0.0), 'radius': 0.3, 'axis': (0, 1, 0)},
            'D': {'center': (0.0, -1.0,  0.0), 'radius': 0.3, 'axis': (0, 1, 0)},
            'L': {'center': (-1.0, 0.0,  0.0), 'radius': 0.3, 'axis': (1, 0, 0)},
            'R': {'center': ( 1.0, 0.0,  0.0), 'radius': 0.3, 'axis': (1, 0, 0)},
            'F': {'center': (0.0,  0.0,  1.0), 'radius': 0.3, 'axis': (0, 0, 1)},
            'B': {'center': (0.0,  0.0, -1.0), 'radius': 0.3, 'axis': (0, 0, 1)},
        }

        # ==== One-fist gesture mode (R/L via open-hand vertical swipe) ====
        self.pinch_as_fist = 0.85   # pinch >= this => "fist"
        self.open_as_open   = 0.35  # pinch <= this => "open"

        self.prev_open_y = None
        self.accum_open_dy = 0.0
        self.dy_deadzone = 0.01     # ignore tiny jitters
        self.dy_trigger  = 0.22     # vertical distance to trigger a 90° turn

        self.gesture_face = None    # 'R' or 'L' when posture is active
        self.gesture_cooldown_s = 0.35
        self.last_gesture_time = 0.0
        # =================================================================

    # ---------------- Basic utilities ----------------

    def calculate_distance(self, p1, p2):
        """3D distance between points"""
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

    def detect_hand_pinch_strength(self, hand_landmarks):
        """Pinch strength in [0..1] from thumb-index distance."""
        if hand_landmarks is None:
            return 0.0
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        distance = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 +
            (thumb_tip.y - index_tip.y)**2 +
            (thumb_tip.z - index_tip.z)**2
        )
        strength = 1.0 - min(1.0, distance * 10.0)
        return max(0.0, min(1.0, strength))

    def is_fist(self, pinch: float) -> bool:
        return pinch >= self.pinch_as_fist

    def is_open(self, pinch: float) -> bool:
        return pinch <= self.open_as_open

    def detect_hand_position(self, hand_landmarks):
        """Map wrist landmark to cube space (approx)."""
        if hand_landmarks is None:
            return None
        wrist = hand_landmarks.landmark[0]
        x = (wrist.x - 0.5) * 4.0   # scale to cube space
        y = (0.5 - wrist.y) * 4.0   # invert Y
        z = wrist.z * 3.0           # depth → Z
        return (x, y, z)

    # --------------- Grabbing & twisting ---------------

    def is_hand_near_face(self, hand_pos, face):
        if hand_pos is None or face not in self.grab_zones:
            return False
        zone = self.grab_zones[face]
        return self.calculate_distance(hand_pos, zone['center']) < zone['radius']

    def detect_grabbed_face(self, left_pos, right_pos, left_pinch, right_pinch):
        """Return face char if either hand is pinching near a face."""
        if left_pinch > 0.5:
            for face in self.grab_zones:
                if self.is_hand_near_face(left_pos, face):
                    return face
        if right_pinch > 0.5:
            for face in self.grab_zones:
                if self.is_hand_near_face(right_pos, face):
                    return face
        return None

    def calculate_twist_angle(self, left_pos, right_pos, grabbed_face):
        """Angle between hands projected per face axis (degrees)."""
        if left_pos is None or right_pos is None or grabbed_face is None:
            return 0.0
        axis = self.grab_zones[grabbed_face]['axis']
        dx = right_pos[0] - left_pos[0]
        dy = right_pos[1] - left_pos[1]
        dz = right_pos[2] - left_pos[2]
        if axis == (0, 1, 0):         # U/D around Y
            angle = math.degrees(math.atan2(dz, dx))
        elif axis == (1, 0, 0):       # L/R around X
            angle = math.degrees(math.atan2(dz, dy))
        else:                         # F/B around Z
            angle = math.degrees(math.atan2(dy, dx))
        return angle

    # ------------------ Physics ------------------

    def update_physics(self):
        self.cube_rotation_x += self.angular_velocity[0]
        self.cube_rotation_y += self.angular_velocity[1]
        self.cube_rotation_z += self.angular_velocity[2]
        self.angular_velocity[0] *= self.friction
        self.angular_velocity[1] *= self.friction
        self.angular_velocity[2] *= self.friction

    # ------------- One-fist gesture engine -------------

    def _maybe_start_or_update_gesture(self, left_pinch, right_pinch, left_pos, right_pos):
        """
        One-fist posture: Right fist + left open -> control 'R' via LEFT hand UP/DOWN.
                          Left fist  + right open -> control 'L' via RIGHT hand UP/DOWN.
        Emits discrete 90° moves when |Δy| passes dy_trigger. Returns action or None.
        """
        now = time.time()
        in_cooldown = (now - self.last_gesture_time) < self.gesture_cooldown_s

        left_fist  = self.is_fist(left_pinch)
        right_fist = self.is_fist(right_pinch)
        left_open  = self.is_open(left_pinch)
        right_open = self.is_open(right_pinch)

        # Need both hands tracked
        if left_pos is None or right_pos is None:
            self.gesture_face = None
            self.prev_open_y = None
            self.accum_open_dy = 0.0
            return None

        # Exactly one fist, other hand open => enter posture
        if right_fist and left_open and not left_fist:
            target_face = 'R'
            open_y = left_pos[1]      # control with open LEFT hand
        elif left_fist and right_open and not right_fist:
            target_face = 'L'
            open_y = right_pos[1]     # control with open RIGHT hand
        else:
            # Not in posture
            self.gesture_face = None
            self.prev_open_y = None
            self.accum_open_dy = 0.0
            return None

        # (Re)start for new face
        if self.gesture_face != target_face:
            self.gesture_face = target_face
            self.prev_open_y = open_y
            self.accum_open_dy = 0.0
            return None  # wait next frame for delta

        # Accumulate vertical movement of open hand
        if self.prev_open_y is None:
            self.prev_open_y = open_y
            return None

        dy = open_y - self.prev_open_y
        self.prev_open_y = open_y
        if abs(dy) > self.dy_deadzone:
            self.accum_open_dy += dy

        if in_cooldown:
            return None

        # Trigger UP (positive) -> R / L'
        if self.accum_open_dy >= self.dy_trigger:
            self.last_gesture_time = now
            self.accum_open_dy = 0.0
            if self.gesture_face == 'R':
                clockwise = True    # UP => R
            else:
                clockwise = False   # UP => L'
            suffix = "" if clockwise else "'"
            print(f"⬆️ One-fist gesture UP -> {self.gesture_face}{suffix}")
            return {
                "action": "rotate",
                "face": self.gesture_face,
                "clockwise": clockwise,
                "source": "one-fist-vertical"
            }

        # Trigger DOWN (negative) -> R' / L
        if self.accum_open_dy <= -self.dy_trigger:
            self.last_gesture_time = now
            self.accum_open_dy = 0.0
            if self.gesture_face == 'R':
                clockwise = False   # DOWN => R'
            else:
                clockwise = True    # DOWN => L
            suffix = "" if clockwise else "'"
            print(f"⬇️ One-fist gesture DOWN -> {self.gesture_face}{suffix}")
            return {
                "action": "rotate",
                "face": self.gesture_face,
                "clockwise": clockwise,
                "source": "one-fist-vertical"
            }

        return None

    # ------------------ Main update ------------------

    def update(self, left_hand_landmarks, right_hand_landmarks):
        """Update controller state from MediaPipe hands and produce actions if any."""
        # Positions & pinch strengths
        left_pos = self.detect_hand_position(left_hand_landmarks)
        right_pos = self.detect_hand_position(right_hand_landmarks)
        left_pinch = self.detect_hand_pinch_strength(left_hand_landmarks)
        right_pinch = self.detect_hand_pinch_strength(right_hand_landmarks)

        # 1) One-fist gesture has priority when not already grabbing
        if self.grabbed_face is None:
            action = self._maybe_start_or_update_gesture(left_pinch, right_pinch, left_pos, right_pos)
            if action:
                self.update_physics()
                return action

        # Check if we're currently in one-fist posture (even if no gesture fired yet)
        in_gesture_posture = self.gesture_face in ('R', 'L')

        # 2) Only allow pinch-near-face grabbing if NOT in one-fist posture
        new_grabbed_face = None
        if not in_gesture_posture:
            new_grabbed_face = self.detect_grabbed_face(left_pos, right_pos, left_pinch, right_pinch)

        # Handle grab begin/end
        if new_grabbed_face and not self.grabbed_face:
            print(f"🎯 GRABBED {new_grabbed_face} face!")
            self.grabbed_face = new_grabbed_face
            self.twist_angle = 0.0
            self.last_twist_time = time.time()

        elif not new_grabbed_face and self.grabbed_face:
            print(f"🖐️ RELEASED {self.grabbed_face} face")
            if abs(self.twist_angle) > self.snap_threshold:
                clockwise = self.twist_angle > 0.0
                action = {
                    "action": "rotate",
                    "face": self.grabbed_face,
                    "clockwise": clockwise,
                    "angle": self.twist_angle
                }
                self.twist_angle = 0.0
                self.grabbed_face = None
                self.update_physics()
                return action
            self.grabbed_face = None
            self.twist_angle = 0.0

        # 3) While grabbing, keep computing twist angle
        if self.grabbed_face:
            current_twist = self.calculate_twist_angle(left_pos, right_pos, self.grabbed_face)
            self.twist_angle = current_twist
            twist_strength = min(1.0, abs(self.twist_angle) / 90.0)
            print(f"🌀 Twisting {self.grabbed_face}: {self.twist_angle:.1f}°")

        # 4) If not grabbing and NOT in gesture posture, free-rotate the view with both hands
        else:
            if left_pos and right_pos and not in_gesture_posture:
                avg_x = (left_pos[0] + right_pos[0]) / 2.0
                avg_y = (left_pos[1] + right_pos[1]) / 2.0
                target_rot_y = avg_x * 90.0
                target_rot_x = avg_y * 90.0
                self.cube_rotation_x += (target_rot_x - self.cube_rotation_x) * 0.1
                self.cube_rotation_y += (target_rot_y - self.cube_rotation_y) * 0.1

        # Physics integration
        self.update_physics()
        return None

    # ---------------- Getters ----------------

    def get_rotation(self):
        return self.cube_rotation_x, self.cube_rotation_y

    def get_twist_angle(self):
        return self.twist_angle

    def get_grabbed_face(self):
        return self.grabbed_face

    def get_mode_text(self):
        if self.grabbed_face:
            direction = "clockwise" if self.twist_angle > 0 else "counter-clockwise"
            return f"✊ HOLDING {self.grabbed_face} - Twist {direction} to rotate ({abs(self.twist_angle):.0f}°)"
        elif self.gesture_face in ('R', 'L'):
            return f"✊ {self.gesture_face} gesture mode — move the other open hand UP/DOWN to turn"
        else:
            return "🖐️ Move hands to rotate cube, pinch near a face to grab"