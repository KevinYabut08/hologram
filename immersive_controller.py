# immersive_controller.py
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple


# ---------------- Mapping config (shared with hand_model) ----------------
@dataclass
class MappingConfig:
    # Mirror/scale/depth/offset must match HandSpaceMapper in hand_model.py
    mirror_x: bool = True
    sx: float = 4.2
    sy: float = 4.2
    sz: float = 2.2

    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0

    # depth policy: 'radial_clamp' | 'fixed_front'
    depth_policy: str = 'radial_clamp'
    r_shell_max: float = 1.65
    fixed_front_z: float = 1.12

    # keep mapped hand a bit in front of the cube front
    front_bias_z: float = 0.15
    min_z_from_cube: float = 1.02


class ImmersiveCubeController:
    """
    Maps MediaPipe hand landmarks to world space, detects direct contact with cube faces,
    supports pinch-to-grab and drag-twist on a face (snaps to 90° on release),
    and provides one-fist (R/L) vertical gestures when not in contact.
    """

    def __init__(self, mapping: MappingConfig = MappingConfig()):
        self.mapping = mapping

        # View / cube orientation (applied in renderer)
        self.cube_rotation_x = 0.0
        self.cube_rotation_y = 0.0
        self.cube_rotation_z = 0.0

        # Physics for free-orbit view
        self.angular_velocity = [0.0, 0.0, 0.0]
        self.friction = 0.95

        # Contact / grab state
        self.grabbed_face: Optional[str] = None   # 'U','D','L','R','F','B'
        self.twist_angle: float = 0.0             # accumulated (deg) while dragging
        self.snap_threshold: float = 25.0         # was 30.0 -> easier snap
        self._contact_active: bool = False
        self._contact_initial_angle: float = 0.0
        self._contact_last_angle: float = 0.0
        self._contact_tol: float = 0.25           # was 0.20 -> more tolerant plane distance
        self._contact_in_bounds: float = 1.06     # was 1.03 -> slightly more edge slack
        self._sticky_depth: float = 0.05          # how much to keep fingertip on plane while grabbing

        # Zones (kept for helper/highlight, though direct contact uses true face planes)
        self.grab_zones = {
            'U': {'center': (0.0,  1.0,  0.0), 'radius': 0.3, 'axis': (0, 1, 0)},
            'D': {'center': (0.0, -1.0,  0.0), 'radius': 0.3, 'axis': (0, 1, 0)},
            'L': {'center': (-1.0, 0.0,  0.0), 'radius': 0.3, 'axis': (1, 0, 0)},
            'R': {'center': ( 1.0, 0.0,  0.0), 'radius': 0.3, 'axis': (1, 0, 0)},
            'F': {'center': (0.0,  0.0,  1.0), 'radius': 0.3, 'axis': (0, 0, 1)},
            'B': {'center': (0.0,  0.0, -1.0), 'radius': 0.3, 'axis': (0, 0, 1)},
        }

        # One-fist vertical gesture (quick R/L)
        self.pinch_as_fist = 0.75      # was 0.85 -> easier pinch
        self.open_as_open = 0.35
        self.prev_open_y: Optional[float] = None
        self.accum_open_dy: float = 0.0
        self.dy_deadzone: float = 0.01
        self.dy_trigger: float = 0.22
        self.gesture_face: Optional[str] = None
        self.gesture_cooldown_s: float = 0.35
        self.last_gesture_time: float = 0.0

    # ----------------- Mapping helpers -----------------
    def _map_point(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Image-space (MediaPipe normalized) -> world space with depth policy + bias + offsets.
        This must mirror the logic in hand_model.HandSpaceMapper.map_point().
        """
        xw = (x - 0.5) * self.mapping.sx
        if self.mapping.mirror_x:
            xw = -xw
        yw = (0.5 - y) * self.mapping.sy
        zw = -z * self.mapping.sz
        p = [xw, yw, zw]

        # depth policy
        if self.mapping.depth_policy == 'fixed_front':
            p[2] = self.mapping.fixed_front_z
        else:
            r = math.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
            if r > self.mapping.r_shell_max and r > 1e-6:
                # in front half-space: scale XY primarily, cap Z softly
                if p[2] > 0.6:
                    scale_xy = self.mapping.r_shell_max / r
                    p[0] *= scale_xy
                    p[1] *= scale_xy
                    p[2] = min(p[2], self.mapping.r_shell_max)
                else:
                    scale = self.mapping.r_shell_max / r
                    p[0] *= scale; p[1] *= scale; p[2] *= scale

        # keep slightly in front of the front face (z≈+1)
        if p[2] > 0:
            p[2] = max(p[2] + self.mapping.front_bias_z, self.mapping.min_z_from_cube)

        # world offsets
        p[0] += self.mapping.offset_x
        p[1] += self.mapping.offset_y
        p[2] += self.mapping.offset_z
        return (p[0], p[1], p[2])

    def detect_index_tip_position(self, hand_landmarks) -> Optional[Tuple[float, float, float]]:
        if hand_landmarks is None:
            return None
        tip = hand_landmarks.landmark[8]  # index tip
        return self._map_point(tip.x, tip.y, tip.z)

    def detect_thumb_tip_position(self, hand_landmarks) -> Optional[Tuple[float, float, float]]:
        if hand_landmarks is None:
            return None
        tip = hand_landmarks.landmark[4]  # thumb tip
        return self._map_point(tip.x, tip.y, tip.z)

    def detect_hand_position(self, hand_landmarks) -> Optional[Tuple[float, float, float]]:
        if hand_landmarks is None:
            return None
        wrist = hand_landmarks.landmark[0]
        return self._map_point(wrist.x, wrist.y, wrist.z)

    # Prefer thumb OR index (whichever is closer to cube center) among available hands
    def _prefer_thumb_or_index(self, left_hand_landmarks, right_hand_landmarks) -> Optional[Tuple[float,float,float]]:
        tips = []
        for lm in (left_hand_landmarks, right_hand_landmarks):
            if lm is None:
                continue
            idx = self.detect_index_tip_position(lm)
            th  = self.detect_thumb_tip_position(lm)
            if idx and th:
                di = self._dist3(idx, (0,0,0))
                dt = self._dist3(th,  (0,0,0))
                tips.append(th if dt < di else idx)
            else:
                tips.append(idx or th)
        return tips[0] if tips else None

    # ----------------- Basic utils -----------------
    @staticmethod
    def _dist3(a, b) -> float:
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    def detect_hand_pinch_strength(self, hand_landmarks) -> float:
        if hand_landmarks is None:
            return 0.0
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        d = math.sqrt((thumb_tip.x - index_tip.x)**2 +
                      (thumb_tip.y - index_tip.y)**2 +
                      (thumb_tip.z - index_tip.z)**2)
        strength = 1.0 - min(1.0, d * 10.0)
        return max(0.0, min(1.0, strength))

    def is_fist(self, pinch: float) -> bool: return pinch >= self.pinch_as_fist
    def is_open(self, pinch: float) -> bool: return pinch <= self.open_as_open

    # ----------------- Rotation transforms -----------------
    @staticmethod
    def _deg2rad(d: float) -> float: return d * math.pi / 180.0

    def _world_to_cube_local(self, p: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Inverse-rotate world point into cube local space using current cube rotations.
        Renderer does:
            glRotatef(rot_x, 1,0,0); glRotatef(rot_y, 0,1,0)
        => world = R_y * R_x * local;  local = R_x^-1 * R_y^-1 * world.
        Apply Y^-1 then X^-1 to the world-space point.
        """
        x, y, z = p
        rx = -self.cube_rotation_x
        ry = -self.cube_rotation_y

        # Y^-1
        c, s = math.cos(self._deg2rad(ry)), math.sin(self._deg2rad(ry))
        x, z = c*x + s*z, -s*x + c*z
        # X^-1
        c, s = math.cos(self._deg2rad(rx)), math.sin(self._deg2rad(rx))
        y, z = c*y - s*z, s*y + c*z
        return (x, y, z)

    # ----------------- Face picking & twist math -----------------
    def _face_from_local_point(self, lp: Tuple[float,float,float]) -> Optional[str]:
        x, y, z = lp
        tol = self._contact_tol
        inb = self._contact_in_bounds
        if abs(z - 1.0) <= tol and abs(x) <= inb and abs(y) <= inb: return 'F'
        if abs(z + 1.0) <= tol and abs(x) <= inb and abs(y) <= inb: return 'B'
        if abs(x - 1.0) <= tol and abs(y) <= inb and abs(z) <= inb: return 'R'
        if abs(x + 1.0) <= tol and abs(y) <= inb and abs(z) <= inb: return 'L'
        if abs(y - 1.0) <= tol and abs(x) <= inb and abs(z) <= inb: return 'U'
        if abs(y + 1.0) <= tol and abs(x) <= inb and abs(z) <= inb: return 'D'
        return None

    def _angle_on_face(self, lp: Tuple[float,float,float], face: str) -> float:
        x, y, z = lp
        if face == 'F':   u, v = x,  y
        elif face == 'B': u, v = -x, y
        elif face == 'R': u, v = z,  y
        elif face == 'L': u, v = -z, y
        elif face == 'U': u, v = x,  z
        else:             u, v = x, -z   # 'D'
        return math.degrees(math.atan2(v, u))  # [-180, 180]

    # ----------------- Gesture engine (quick R/L) -----------------
    def _maybe_start_or_update_gesture(self, left_pinch, right_pinch, left_pos, right_pos):
        now = time.time()
        in_cd = (now - self.last_gesture_time) < self.gesture_cooldown_s
        left_fist  = self.is_fist(left_pinch)
        right_fist = self.is_fist(right_pinch)
        left_open  = self.is_open(left_pinch)
        right_open = self.is_open(right_pinch)

        if left_pos is None or right_pos is None:
            self.gesture_face = None
            self.prev_open_y = None
            self.accum_open_dy = 0.0
            return None

        if right_fist and left_open and not left_fist:
            target_face = 'R'; open_y = left_pos[1]
        elif left_fist and right_open and not right_fist:
            target_face = 'L'; open_y = right_pos[1]
        else:
            self.gesture_face = None
            self.prev_open_y = None
            self.accum_open_dy = 0.0
            return None

        if self.gesture_face != target_face:
            self.gesture_face = target_face
            self.prev_open_y = open_y
            self.accum_open_dy = 0.0
            return None

        if self.prev_open_y is None:
            self.prev_open_y = open_y
            return None

        dy = open_y - self.prev_open_y
        self.prev_open_y = open_y
        if abs(dy) > self.dy_deadzone:
            self.accum_open_dy += dy
        if in_cd:
            return None

        if self.accum_open_dy >= self.dy_trigger:
            self.last_gesture_time = now
            self.accum_open_dy = 0.0
            clockwise = (self.gesture_face == 'R')
            return {"action": "rotate", "face": self.gesture_face, "clockwise": clockwise, "source": "one-fist-vertical"}
        if self.accum_open_dy <= -self.dy_trigger:
            self.last_gesture_time = now
            self.accum_open_dy = 0.0
            clockwise = not (self.gesture_face == 'R')
            return {"action": "rotate", "face": self.gesture_face, "clockwise": clockwise, "source": "one-fist-vertical"}
        return None

    # ----------------- Physics -----------------
    def update_physics(self):
        # Integrate angular velocity into cube orientation and apply friction
        self.cube_rotation_x += self.angular_velocity[0]
        self.cube_rotation_y += self.angular_velocity[1]
        self.cube_rotation_z += self.angular_velocity[2]
        self.angular_velocity[0] *= self.friction
        self.angular_velocity[1] *= self.friction
        self.angular_velocity[2] *= self.friction

    # ----------------- Main update -----------------
    def update(self, left_hand_landmarks, right_hand_landmarks):
        # World-space points (already depth-managed by _map_point)
        left_wrist  = self.detect_hand_position(left_hand_landmarks)
        right_wrist = self.detect_hand_position(right_hand_landmarks)

        left_pinch = self.detect_hand_pinch_strength(left_hand_landmarks)
        right_pinch = self.detect_hand_pinch_strength(right_hand_landmarks)
        any_pinch = (left_pinch > 0.5) or (right_pinch > 0.5)

        # Choose a control fingertip: thumb OR index (per hand), whichever is closer to cube center
        tip_world = self._prefer_thumb_or_index(left_hand_landmarks, right_hand_landmarks)
        touched_face = None
        if tip_world is not None:
            tip_local = self._world_to_cube_local(tip_world)
            touched_face = self._face_from_local_point(tip_local)

        # ---- Direct contact & drag-twist when pinching on a face ----
        if any_pinch and touched_face:
            if not self._contact_active:
                # Begin contact
                self._contact_active = True
                self.grabbed_face = touched_face
                self._contact_initial_angle = self._angle_on_face(tip_local, touched_face)
                self._contact_last_angle = self._contact_initial_angle
                self.twist_angle = 0.0
            else:
                # Keep same face to avoid sudden jumps
                if self.grabbed_face != touched_face:
                    touched_face = self.grabbed_face

                # Push-out: keep fingertip just outside the grabbed face plane (cube-local)
                x, y, z = tip_local
                eps = 0.03
                if self.grabbed_face == 'F':   z = max(z,  1.0 + eps)
                elif self.grabbed_face == 'B': z = min(z, -1.0 - eps)
                elif self.grabbed_face == 'R': x = max(x,  1.0 + eps)
                elif self.grabbed_face == 'L': x = min(x, -1.0 - eps)
                elif self.grabbed_face == 'U': y = max(y,  1.0 + eps)
                else:                           y = min(y, -1.0 - eps)
                tip_local = (x, y, z)

                current_angle = self._angle_on_face(tip_local, self.grabbed_face)
                da = current_angle - self._contact_last_angle
                # unwrap angle delta to smallest step
                while da > 180.0: da -= 360.0
                while da < -180.0: da += 360.0
                self.twist_angle += da
                self._contact_last_angle = current_angle

            self.update_physics()
            return None  # no immediate action; we commit on release

        # End of pinch while in contact -> commit snap if over threshold
        if self._contact_active and not any_pinch:
            self._contact_active = False
            if self.grabbed_face and abs(self.twist_angle) > self.snap_threshold:
                clockwise = (self.twist_angle > 0)
                action = {
                    "action": "rotate",
                    "face": self.grabbed_face,
                    "clockwise": clockwise,
                    "angle": self.twist_angle,
                    "source": "direct-contact"
                }
                # Reset state
                self.twist_angle = 0.0
                self.grabbed_face = None
                self.update_physics()
                return action
            # No snap -> clear
            self.twist_angle = 0.0
            self.grabbed_face = None

        # ---- Not in contact: gesture or free orbit ----
        if self.grabbed_face is None:
            # One-fist quick R/L when not grabbing
            action = self._maybe_start_or_update_gesture(left_pinch, right_pinch, left_wrist, right_wrist)
            if action:
                self.update_physics()
                return action

            # Free-orbit view controlled by both wrists if not in gesture posture
            if left_wrist and right_wrist and self.gesture_face not in ('R', 'L'):
                avg_x = (left_wrist[0] + right_wrist[0]) / 2.0
                avg_y = (left_wrist[1] + right_wrist[1]) / 2.0
                target_rot_y = avg_x * 90.0
                target_rot_x = avg_y * 90.0
                self.cube_rotation_x += (target_rot_x - self.cube_rotation_x) * 0.1
                self.cube_rotation_y += (target_rot_y - self.cube_rotation_y) * 0.1

        self.update_physics()
        return None

    # ----------------- Helpers for UI/renderer -----------------
    def get_rotation(self) -> Tuple[float, float]:
        return self.cube_rotation_x, self.cube_rotation_y

    def get_twist_angle(self) -> float:
        return self.twist_angle

    def get_grabbed_face(self) -> Optional[str]:
        return self.grabbed_face

    def get_hover_face_from_tip_world(self, tip_world) -> Optional[str]:
        if tip_world is None:
            return None
        lp = self._world_to_cube_local(tip_world)
        return self._face_from_local_point(lp)

    def get_mode_text(self) -> str:
        if self._contact_active and self.grabbed_face:
            direction = "clockwise" if self.twist_angle > 0 else "counter-clockwise"
            return f"✊ GRABBED {self.grabbed_face} — drag to twist {direction} ({abs(self.twist_angle):.0f}°)"
        elif self.gesture_face in ('R', 'L'):
            return f"✊ {self.gesture_face} gesture — move the other open hand UP/DOWN"
        else:
            return "🖐️ Touch a face and pinch to grab; drag around to twist • (R/L quick-turn via one-fist)"