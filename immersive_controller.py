import math
import time
import numpy as np

class ImmersiveCubeController:
    def __init__(self):
        # Physical cube properties
        self.cube_rotation_x = 0
        self.cube_rotation_y = 0
        self.cube_rotation_z = 0
        
        # Hand tracking
        self.left_hand = None
        self.right_hand = None
        
        # Physical interaction states
        self.grabbed_face = None
        self.grab_strength = 0  # How hard are they grabbing?
        self.twist_angle = 0    # Current twist angle
        self.snap_threshold = 30  # Degrees to snap rotation
        
        # Physical feedback
        self.last_twist_time = 0
        self.vibration_intensity = 0
        
        # Cube physics
        self.angular_velocity = [0, 0, 0]
        self.friction = 0.95
        
        # Zones for grabbing (like holding a real cube)
        self.grab_zones = {
            'U': {'center': (0, 1.0, 0),  'radius': 0.3, 'axis': (0, 1, 0)},
            'D': {'center': (0, -1.0, 0), 'radius': 0.3, 'axis': (0, 1, 0)},
            'L': {'center': (-1.0, 0, 0), 'radius': 0.3, 'axis': (1, 0, 0)},
            'R': {'center': (1.0, 0, 0),  'radius': 0.3, 'axis': (1, 0, 0)},
            'F': {'center': (0, 0, 1.0),  'radius': 0.3, 'axis': (0, 0, 1)},
            'B': {'center': (0, 0, -1.0), 'radius': 0.3, 'axis': (0, 0, 1)}
        }

        # ----------------- NEW: two-hand swipe state -----------------
        self.prev_left_x = None
        self.prev_right_x = None
        self.accum_left_dx = 0.0
        self.accum_right_dx = 0.0

        # Lateral thresholds (cube-space units; tune)
        self.dx_deadzone = 0.01     # ignore tiny jitters
        self.dx_trigger = 0.20      # how far you must move to fire a gesture

        # Fist detection thresholds
        self.pinch_as_fist = 0.85   # pinch strength >= this means "fist"
        self.open_as_open = 0.35    # pinch strength <= this means "open"

        # Cooldown to avoid multiple triggers from one swipe
        self.swipe_cooldown_s = 0.35
        self.last_swipe_time = 0.0
        # -------------------------------------------------------------

    def calculate_distance(self, p1, p2):
        """Calculate 3D distance between points"""
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
    
    def detect_hand_pinch_strength(self, hand_landmarks):
        """How hard is the hand pinching? (0=open, 1=tight pinch)"""
        if hand_landmarks is None:
            return 0.0
        
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        
        distance = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 +
            (thumb_tip.y - index_tip.y)**2 +
            (thumb_tip.z - index_tip.z)**2
        )
        strength = 1.0 - min(1.0, distance * 10)
        return max(0.0, min(1.0, strength))

    # ----------------- NEW: convenience predicates -----------------
    def is_fist(self, pinch_strength: float) -> bool:
        return pinch_strength >= self.pinch_as_fist

    def is_open(self, pinch_strength: float) -> bool:
        return pinch_strength <= self.open_as_open
    # ---------------------------------------------------------------

    def detect_hand_position(self, hand_landmarks):
        """Get approximate 3D position of hand relative to cube"""
        if hand_landmarks is None:
            return None
        
        wrist = hand_landmarks.landmark[0]
        # Simple camera→cube mapping (keep consistent with renderer/capture)
        x = (wrist.x - 0.5) * 4  # Scale to cube space
        y = (0.5 - wrist.y) * 4  # Invert Y
        z = wrist.z * 3          # Depth → Z
        return (x, y, z)
    
    def is_hand_near_face(self, hand_pos, face):
        """Check if hand is close enough to grab a face"""
        if hand_pos is None or face not in self.grab_zones:
            return False
        zone = self.grab_zones[face]
        distance = self.calculate_distance(hand_pos, zone['center'])
        return distance < zone['radius']
    
    def detect_grabbed_face(self, left_pos, right_pos, left_pinch, right_pinch):
        """Which face is being grabbed?"""
        # Prefer left if both trying to grab; your choice
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
        """Calculate how much the hands are twisting"""
        if left_pos is None or right_pos is None or grabbed_face is None:
            return 0.0
        
        zone = self.grab_zones[grabbed_face]
        axis = zone['axis']
        
        dx = right_pos[0] - left_pos[0]
        dy = right_pos[1] - left_pos[1]
        dz = right_pos[2] - left_pos[2]
        
        if axis == (0, 1, 0):       # U/D face - around Y
            angle = math.degrees(math.atan2(dz, dx))
        elif axis == (1, 0, 0):     # L/R face - around X
            angle = math.degrees(math.atan2(dz, dy))
        else:                        # F/B face - around Z
            angle = math.degrees(math.atan2(dy, dx))
        return angle
    
    def update_physics(self):
        """Update cube physics (spinning, etc.)"""
        self.cube_rotation_x += self.angular_velocity[0]
        self.cube_rotation_y += self.angular_velocity[1]
        self.cube_rotation_z += self.angular_velocity[2]
        self.angular_velocity[0] *= self.friction
        self.angular_velocity[1] *= self.friction
        self.angular_velocity[2] *= self.friction

    # ----------------- NEW: helper for two-hand swipe -----------------
    def _process_two_hand_swipe(self, left_pos, right_pos, left_pinch, right_pinch):
        """
        Returns an action dict if a two-hand swipe gesture should trigger a U/D move.
        Otherwise returns None.
        """
        now = time.time()
        if (now - self.last_swipe_time) < self.swipe_cooldown_s:
            # In cooldown, ignore gestures
            return None

        # Need both hands tracked
        if left_pos is None or right_pos is None:
            self.prev_left_x = None
            self.prev_right_x = None
            self.accum_left_dx = 0.0
            self.accum_right_dx = 0.0
            return None

        # Update accumulators for lateral movement
        # LEFT HAND
        if self.prev_left_x is None:
            self.prev_left_x = left_pos[0]
        dx_left = left_pos[0] - self.prev_left_x
        self.prev_left_x = left_pos[0]
        if abs(dx_left) > self.dx_deadzone:
            self.accum_left_dx += dx_left

        # RIGHT HAND
        if self.prev_right_x is None:
            self.prev_right_x = right_pos[0]
        dx_right = right_pos[0] - self.prev_right_x
        self.prev_right_x = right_pos[0]
        if abs(dx_right) > self.dx_deadzone:
            self.accum_right_dx += dx_right

        # Decide who is the "active" fist
        right_is_fist = self.is_fist(right_pinch)
        left_is_fist  = self.is_fist(left_pinch)
        right_is_open = self.is_open(right_pinch)
        left_is_open  = self.is_open(left_pinch)

        # Priority: exactly one fist (to avoid ambiguity)
        # Right fist + left open => control U layer by right-hand lateral move
        if right_is_fist and left_is_open and not left_is_fist:
            if self.accum_right_dx >= self.dx_trigger:
                # Right hand moved right -> U
                self.last_swipe_time = now
                self.accum_right_dx = 0.0
                print("➡️ Right-fist swipe RIGHT -> U")
                return {"action": "rotate", "face": "U", "clockwise": True, "source": "two-hand-swipe"}
            elif self.accum_right_dx <= -self.dx_trigger:
                # Right hand moved left -> U'
                self.last_swipe_time = now
                self.accum_right_dx = 0.0
                print("⬅️ Right-fist swipe LEFT -> U'")
                return {"action": "rotate", "face": "U", "clockwise": False, "source": "two-hand-swipe"}

        # Left fist + right open => control D layer by left-hand lateral move
        if left_is_fist and right_is_open and not right_is_fist:
            if self.accum_left_dx >= self.dx_trigger:
                # Left hand moved right -> D' (camera-right motion)
                self.last_swipe_time = now
                self.accum_left_dx = 0.0
                print("➡️ Left-fist swipe RIGHT -> D'")
                return {"action": "rotate", "face": "D", "clockwise": False, "source": "two-hand-swipe"}
            elif self.accum_left_dx <= -self.dx_trigger:
                # Left hand moved left -> D
                self.last_swipe_time = now
                self.accum_left_dx = 0.0
                print("⬅️ Left-fist swipe LEFT -> D")
                return {"action": "rotate", "face": "D", "clockwise": True, "source": "two-hand-swipe"}

        # If both fists or both open, don't trigger (reset accumulators slowly)
        # To keep responsiveness, we don't hard reset here; they continue accumulating.

        return None
    # -------------------------------------------------------------------

    def update(self, left_hand_landmarks, right_hand_landmarks):
        """Update controller with immersive physics"""
        # Get hand positions and pinch strength
        left_pos = self.detect_hand_position(left_hand_landmarks)
        right_pos = self.detect_hand_position(right_hand_landmarks)
        left_pinch = self.detect_hand_pinch_strength(left_hand_landmarks)
        right_pinch = self.detect_hand_pinch_strength(right_hand_landmarks)
        
        # ----------------- NEW: two-hand swipe shortcut -----------------
        # If we are NOT currently grabbing a face, allow the two-hand layer swipe
        if self.grabbed_face is None:
            swipe_action = self._process_two_hand_swipe(left_pos, right_pos, left_pinch, right_pinch)
            if swipe_action is not None:
                # When a swipe triggers, we DON'T enter grabbed-face state.
                # We just emit the action and return it for the renderer/engine to animate+commit.
                self.update_physics()
                return swipe_action
        # ----------------------------------------------------------------

        # Detect which face is being grabbed (single-hand grab mode)
        new_grabbed_face = self.detect_grabbed_face(left_pos, right_pos, left_pinch, right_pinch)
        
        # Handle grabbing/releasing
        if new_grabbed_face and not self.grabbed_face:
            # Just started grabbing
            print(f"🎯 GRABBED {new_grabbed_face} face!")
            self.grabbed_face = new_grabbed_face
            self.twist_angle = 0.0
            self.last_twist_time = time.time()
        
        elif not new_grabbed_face and self.grabbed_face:
            # Released
            print(f"🖐️ RELEASED {self.grabbed_face} face")
            if abs(self.twist_angle) > self.snap_threshold:
                clockwise = self.twist_angle > 0
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
        
        # Update twist if grabbing (single-hand mode)
        if self.grabbed_face:
            current_twist = self.calculate_twist_angle(left_pos, right_pos, self.grabbed_face)
            self.twist_angle = current_twist
            twist_strength = min(1.0, abs(self.twist_angle) / 90.0)
            print(f"🌀 Twisting {self.grabbed_face}: {self.twist_angle:.1f}°")
        
        # If not grabbing, free rotate view with both open hands
        else:
            if left_pos and right_pos:
                avg_x = (left_pos[0] + right_pos[0]) / 2
                avg_y = (left_pos[1] + right_pos[1]) / 2
                target_rot_y = avg_x * 90.0
                target_rot_x = avg_y * 90.0
                self.cube_rotation_x += (target_rot_x - self.cube_rotation_x) * 0.1
                self.cube_rotation_y += (target_rot_y - self.cube_rotation_y) * 0.1
        
        # Update physics
        self.update_physics()
        return None
    
    def get_rotation(self):
        """Get current cube rotation for rendering"""
        return self.cube_rotation_x, self.cube_rotation_y
    
    def get_twist_angle(self):
        """Get current twist angle for animation"""
        return self.twist_angle
    
    def get_grabbed_face(self):
        """Get currently grabbed face"""
        return self.grabbed_face
    
    def get_mode_text(self):
        """Get status text"""
        if self.grabbed_face:
            direction = "clockwise" if self.twist_angle > 0 else "counter-clockwise"
            return f"✊ HOLDING {self.grabbed_face} - Twist {direction} to rotate ({abs(self.twist_angle):.0f}°)"
        else:
            return "🖐️ Move hands to rotate cube, pinch near a face to grab"