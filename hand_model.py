# hand_model.py  (replace HandSpaceMapper; add JointSmoother; minor edits in ArticulatedHandModelGL)
from dataclasses import dataclass
from typing import Tuple
import math
import numpy as np

from OpenGL.GL import (
    glColor3f, glPushMatrix, glPopMatrix, glTranslatef, glRotatef,
    glBegin, glEnd, glVertex3f, GL_QUADS
)
from OpenGL.GLU import gluNewQuadric, gluSphere, gluCylinder, gluDeleteQuadric

HAND_BONES = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12),
    (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20)
]

@dataclass
class HandSpaceMapper:
    mirror_x: bool = True
    sx: float = 4.2
    sy: float = 4.2
    sz: float = 2.2
    # world offsets (nudge mapped hand in world space)
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0
    # depth policy: 'radial_clamp' | 'fixed_front'
    depth_policy: str = 'radial_clamp'
    r_shell_max: float = 1.65
    fixed_front_z: float = 1.12
    # keep hand out of the cube front
    front_bias_z: float = 0.15
    min_z_from_cube: float = 1.02

    def map_point(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        # 1) base mapping from MediaPipe normalized coords
        xw = (x - 0.5) * self.sx
        if self.mirror_x:
            xw = -xw
        yw = (0.5 - y) * self.sy
        zw = -z * self.sz
        p = np.array([xw, yw, zw], dtype=np.float32)

        # 2) depth policy
        if self.depth_policy == 'fixed_front':
            p[2] = self.fixed_front_z
        else:
            # One‑sided / gentle radial clamp so we don’t get pulled through the cube
            r = float(np.linalg.norm(p))
            if r > self.r_shell_max and r > 1e-6:
                if p[2] > 0.6:  # in front half-space -> keep Z, scale XY
                    scale_xy = self.r_shell_max / r
                    p[0] *= scale_xy
                    p[1] *= scale_xy
                    p[2] = min(p[2], self.r_shell_max)
                else:
                    p = p * (self.r_shell_max / r)

        # 3) keep hand slightly in front of front face (z≈+1 plane)
        if p[2] > 0:
            p[2] = max(p[2] + self.front_bias_z, self.min_z_from_cube)

        # 4) world offsets (fine alignment)
        p[0] += self.offset_x
        p[1] += self.offset_y
        p[2] += self.offset_z
        return float(p[0]), float(p[1]), float(p[2])


# --- Optional smoothing to reduce jitter ---
class JointSmoother:
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.prev = None  # np.array shape (21,3)

    def apply(self, joints_np: np.ndarray) -> np.ndarray:
        if self.prev is None or self.prev.shape != joints_np.shape:
            self.prev = joints_np.copy()
            return joints_np
        self.prev = self.alpha * joints_np + (1.0 - self.alpha) * self.prev
        return self.prev


class ArticulatedHandModelGL:
    def __init__(self, mapper: HandSpaceMapper, joint_radius=0.055, bone_radius=0.045, color=(0.95, 0.72, 0.25)):
        self.mapper = mapper
        self.joint_radius = joint_radius
        self.bone_radius = bone_radius
        self.color = color
        self._quadric = gluNewQuadric()
        self._smoother = JointSmoother(alpha=0.35)

    def _mp_landmarks_to_world(self, hand_landmarks):
        pts = []
        for lm in hand_landmarks.landmark:
            pts.append(self.mapper.map_point(lm.x, lm.y, lm.z))
        joints = np.asarray(pts, dtype=np.float32)
        joints = self._smoother.apply(joints)
        return joints

    def _draw_sphere(self, radius: float):
        gluSphere(self._quadric, radius, 12, 10)

    def _axis_angle_from_z(self, vx, vy, vz):
        v_len = math.sqrt(vx*vx + vy*vy + vz*vz)
        if v_len < 1e-6: return 0.0, 0.0, 0.0, 1.0
        vx /= v_len; vy /= v_len; vz /= v_len
        dot = max(-1.0, min(1.0, vz))
        angle = math.degrees(math.acos(dot))
        ax, ay, az = -vy, vx, 0.0
        n = math.sqrt(ax*ax + ay*ay + az*az)
        if n < 1e-6:
            return (0.0, 1.0, 0.0, 0.0) if vz > 0.0 else (180.0, 1.0, 0.0, 0.0)
        ax /= n; ay /= n; az /= n
        return angle, ax, ay, az

    def _draw_capsule(self, p1, p2, radius: float):
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        vx, vy, vz = (x2-x1, y2-y1, z2-z1)
        length = math.sqrt(vx*vx + vy*vy + vz*vz)
        if length < 1e-5:
            glPushMatrix(); glTranslatef(x1, y1, z1); self._draw_sphere(radius); glPopMatrix(); return
        angle, ax, ay, az = self._axis_angle_from_z(vx, vy, vz)
        glPushMatrix()
        glTranslatef(x1, y1, z1)
        glRotatef(angle, ax, ay, az)
        gluCylinder(self._quadric, radius, radius, length, 10, 1)
        glPopMatrix()
        glPushMatrix(); glTranslatef(x1, y1, z1); self._draw_sphere(radius); glPopMatrix()
        glPushMatrix(); glTranslatef(x2, y2, z2); self._draw_sphere(radius); glPopMatrix()

    def draw(self, hand_landmarks):
        if hand_landmarks is None:
            return
        joints = self._mp_landmarks_to_world(hand_landmarks)
        glColor3f(*self.color)
        for a, b in HAND_BONES:
            self._draw_capsule(joints[a], joints[b], self.bone_radius)
        glColor3f(min(1.0, self.color[0]+0.05), min(1.0, self.color[1]+0.05), min(1.0, self.color[2]+0.05))
        for (jx, jy, jz) in joints:
            glPushMatrix(); glTranslatef(jx, jy, jz); self._draw_sphere(self.joint_radius * 0.9); glPopMatrix()

    def __del__(self):
        try:
            if self._quadric: gluDeleteQuadric(self._quadric)
        except Exception:
            pass
