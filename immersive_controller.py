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
            'U': {'center': (0, 1.0, 0), 'radius': 0.3, 'axis': (0, 1, 0)},
            'D': {'center': (0, -1.0, 0), 'radius': 0.3, 'axis': (0, 1, 0)},
            'L': {'center': (-1.0, 0, 0), 'radius': 0.3, 'axis': (1, 0, 0)},
            'R': {'center': (1.0, 0, 0), 'radius': 0.3, 'axis': (1, 0, 0)},
            'F': {'center': (0, 0, 1.0), 'radius': 0.3, 'axis': (0, 0, 1)},
            'B': {'center': (0, 0, -1.0), 'radius': 0.3, 'axis': (0, 0, 1)}
        }
    
    def calculate_distance(self, p1, p2):
        """Calculate 3D distance between points"""
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
    
    def detect_hand_pinch_strength(self, hand_landmarks):
        """How hard is the hand pinching?"""
        if hand_landmarks is None:
            return 0
        
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        
        # Distance between thumb and index
        distance = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 +
            (thumb_tip.y - index_tip.y)**2 +
            (thumb_tip.z - index_tip.z)**2
        )
        
        # Convert to pinch strength (0=open, 1=tight pinch)
        strength = 1.0 - min(1.0, distance * 10)
        return max(0, strength)
    
    def detect_hand_position(self, hand_landmarks):
        """Get approximate 3D position of hand relative to cube"""
        if hand_landmarks is None:
            return None
        
        # Use wrist and middle of hand for position
        wrist = hand_landmarks.landmark[0]
        middle_mcp = hand_landmarks.landmark[9]
        
        # Convert to 3D space around cube
        # Map from camera coordinates to cube coordinates
        x = (wrist.x - 0.5) * 4  # Scale to cube space
        y = (0.5 - wrist.y) * 4  # Invert Y
        z = wrist.z * 3  # Use depth for Z
        
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
        # Check left hand first
        if left_pinch > 0.5:
            for face in self.grab_zones:
                if self.is_hand_near_face(left_pos, face):
                    return face
        
        # Check right hand
        if right_pinch > 0.5:
            for face in self.grab_zones:
                if self.is_hand_near_face(right_pos, face):
                    return face
        
        return None
    
    def calculate_twist_angle(self, left_pos, right_pos, grabbed_face):
        """Calculate how much the hands are twisting"""
        if left_pos is None or right_pos is None or grabbed_face is None:
            return 0
        
        zone = self.grab_zones[grabbed_face]
        axis = zone['axis']
        
        # Project hand positions onto plane perpendicular to rotation axis
        # This simulates twisting motion
        
        # Simple approximation: use angle between hands
        dx = right_pos[0] - left_pos[0]
        dy = right_pos[1] - left_pos[1]
        dz = right_pos[2] - left_pos[2]
        
        # Different calculations based on which axis
        if axis == (0, 1, 0):  # U/D face - rotate around Y
            angle = math.atan2(dz, dx) * 180 / math.pi
        elif axis == (1, 0, 0):  # L/R face - rotate around X
            angle = math.atan2(dz, dy) * 180 / math.pi
        else:  # F/B face - rotate around Z
            angle = math.atan2(dy, dx) * 180 / math.pi
        
        return angle
    
    def update_physics(self):
        """Update cube physics (spinning, etc.)"""
        # Apply angular velocity
        self.cube_rotation_x += self.angular_velocity[0]
        self.cube_rotation_y += self.angular_velocity[1]
        self.cube_rotation_z += self.angular_velocity[2]
        
        # Apply friction
        self.angular_velocity[0] *= self.friction
        self.angular_velocity[1] *= self.friction
        self.angular_velocity[2] *= self.friction
    
    def update(self, left_hand_landmarks, right_hand_landmarks):
        """Update controller with immersive physics"""
        # Get hand positions and pinch strength
        left_pos = self.detect_hand_position(left_hand_landmarks)
        right_pos = self.detect_hand_position(right_hand_landmarks)
        left_pinch = self.detect_hand_pinch_strength(left_hand_landmarks)
        right_pinch = self.detect_hand_pinch_strength(right_hand_landmarks)
        
        # Detect which face is being grabbed
        new_grabbed_face = self.detect_grabbed_face(left_pos, right_pos, left_pinch, right_pinch)
        
        # Handle grabbing/releasing
        if new_grabbed_face and not self.grabbed_face:
            # Just started grabbing
            print(f"🎯 GRABBED {new_grabbed_face} face!")
            self.grabbed_face = new_grabbed_face
            self.twist_angle = 0
            self.last_twist_time = time.time()
        
        elif not new_grabbed_face and self.grabbed_face:
            # Released
            print(f"🖐️ RELEASED {self.grabbed_face} face")
            
            # Check if we should snap rotation
            if abs(self.twist_angle) > self.snap_threshold:
                # Determine direction (clockwise/counter-clockwise)
                clockwise = self.twist_angle > 0
                action = {
                    "action": "rotate",
                    "face": self.grabbed_face,
                    "clockwise": clockwise,
                    "angle": self.twist_angle
                }
                self.twist_angle = 0
                self.grabbed_face = None
                return action
            
            self.grabbed_face = None
            self.twist_angle = 0
        
        # Update twist if grabbing
        if self.grabbed_face:
            current_twist = self.calculate_twist_angle(left_pos, right_pos, self.grabbed_face)
            self.twist_angle = current_twist
            
            # Visual feedback based on twist strength
            twist_strength = min(1.0, abs(self.twist_angle) / 90)
            print(f"🌀 Twisting {self.grabbed_face}: {self.twist_angle:.1f}°")
        
        # If not grabbing, use open hands to rotate view
        else:
            if left_pos and right_pos:
                # Average hand position controls free rotation
                avg_x = (left_pos[0] + right_pos[0]) / 2
                avg_y = (left_pos[1] + right_pos[1]) / 2
                avg_z = (left_pos[2] + right_pos[2]) / 2
                
                # Smooth rotation with physics
                target_rot_y = avg_x * 90
                target_rot_x = avg_y * 90
                
                # Approach target with easing
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