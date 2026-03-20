# renderer.py
import math
import time
import random

import pygame
from pygame.locals import DOUBLEBUF, OPENGL

from OpenGL.GL import *
from OpenGL.GLU import *

# Read engine state
from rubiks import Face as F  # <-- adjust this import if your engine's file/module name differs


class ImmersiveCubeRenderer:
    """
    Windows-focused Pygame + PyOpenGL renderer for a 3x3 Rubik's Cube
    - 27 beveled cubies
    - Animated face turns (90°) with a move queue
    - Reads sticker colors from your RubiksCube engine instance
    """

    def __init__(self):
        pygame.init()
        self.display = (1600, 1000)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("🎮 PHYSICAL RUBIK'S CUBE - Grab & Twist!")

        # 60 FPS cap for smoother animation
        self.clock = pygame.time.Clock()

        # --- GL setup ---
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 80.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -12.0)

        # Lights
        glLightfv(GL_LIGHT0, GL_POSITION, (6, 8, 10, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.25, 0.25, 0.30, 1))

        glLightfv(GL_LIGHT1, GL_POSITION, (-7, -5, 8, 1))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, (0.5, 0.5, 0.6, 1))

        # Cube visual properties
        self.cube_size = 2.7
        self.cubie_size = self.cube_size / 3.0
        self.cubie_gap = 0.06     # gap between cubies
        self.bevel_size = 0.08    # rounded face inset

        # Materials
        self._setup_materials()

        # Text (2D overlay)
        pygame.font.init()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Starfield (precomputed for performance)
        self.stars = [(random.uniform(-24, 24), random.uniform(-18, 18), random.uniform(-20, -8)) for _ in range(280)]

        # Engine hookup
        self.cube = None  # set via set_cube()
        self._color_name = {
            0: "white",
            1: "yellow",
            2: "red",
            3: "orange",
            4: "green",
            5: "blue",
        }

        # Animation state & queue
        self.animating = False
        self.anim_face = None       # 'F','B','U','D','L','R'
        self.anim_dir = +1          # +1 (clockwise from face perspective), -1 (prime)
        self.anim_t0 = 0.0
        self.anim_duration = 0.22   # seconds
        self.queue = []             # list[(face, clockwise:bool)]

        # Visual feedback
        self.grab_highlight = 0.0   # 0..1
        self.snap_time = 0.0

        # Face rotation sign so that positive 'clockwise' matches intuitive face view
        self.FACE_SIGN = {
            'F': +1,
            'B': -1,
            'U': +1,
            'D': -1,
            'R': +1,
            'L': -1,
        }

    # ---------- Public API ----------

    def set_cube(self, cube):
        """Attach your RubiksCube engine instance so we can read and commit moves."""
        self.cube = cube

    def rotate_face(self, face: str, clockwise: bool):
        """
        Request a 90° face turn with animation.
        If an animation is already running, the move is queued.
        """
        face = face.upper()
        if self.animating:
            self.queue.append((face, clockwise))
            return
        self._start_animation(face, clockwise)

    def is_busy(self) -> bool:
        return self.animating or bool(self.queue)

    # ---------- Core rendering ----------

    def render(self, rotation_x, rotation_y, grabbed_face=None, twist_angle=0,
               mode_text="", move_count=0, timer=0.0) -> bool:
        """Main render call. Returns False if window was closed."""
        # Process basic quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        # Update grab highlight
        if grabbed_face:
            self.grab_highlight = min(1.0, self.grab_highlight + 0.12)
        else:
            self.grab_highlight = max(0.0, self.grab_highlight - 0.06)

        # Clear
        glClearColor(0.08, 0.08, 0.12, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Background
        self._draw_starfield()

        # Camera transform
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -12.0)
        glRotatef(rotation_x, 1, 0, 0)
        glRotatef(rotation_y, 0, 1, 0)

        # Grab overlays
        if grabbed_face:
            self._draw_grabbed_face_highlight(grabbed_face, twist_angle)
            self._draw_twist_indicator(grabbed_face, twist_angle)

        # Cube (animated or static)
        self._draw_cube_animated_or_static()

        # UI overlay
        self._draw_ui(mode_text, move_count, timer, grabbed_face, twist_angle)

        pygame.display.flip()
        self.clock.tick(60)
        return True

    def cleanup(self):
        pygame.quit()

    # ---------- Animation helpers ----------

    def _start_animation(self, face: str, clockwise: bool):
        self.animating = True
        self.anim_face = face
        self.anim_dir = +1 if clockwise else -1
        self.anim_t0 = time.time()

    def _finish_animation_and_commit(self):
        """Commit logical move to engine and start next queued move if any."""
        if self.cube and self.anim_face:
            move = self.anim_face if self.anim_dir > 0 else f"{self.anim_face}'"
            self.cube.perform_move(move)

        self.animating = False
        self.anim_face = None
        self.anim_dir = +1
        self.anim_t0 = 0.0

        # Next queued move, if any
        if self.queue:
            face, cw = self.queue.pop(0)
            self._start_animation(face, cw)

    def _easing(self, t: float) -> float:
        # Ease-out cubic for a nice click feel
        t = max(0.0, min(1.0, t))
        return 1.0 - (1.0 - t) ** 3

    # ---------- Drawing: cube ----------

    def _draw_cube_animated_or_static(self):
        """
        Draw all 27 cubies. If animating, rotate the target layer about its axis.
        """
        if self.cube is None:
            return

        # Precompute which cubies belong to the rotating layer
        rotating_indices = set()
        angle = 0.0
        axis = (0, 0, 0)

        if self.animating and self.anim_face:
            t = (time.time() - self.anim_t0) / self.anim_duration
            if t >= 1.0:
                # finalize
                self._finish_animation_and_commit()
            else:
                eased = self._easing(t)
                signed = self.anim_dir * self.FACE_SIGN[self.anim_face]
                angle = 90.0 * signed * eased
                axis = self._axis_for_face(self.anim_face)
                rotating_indices = self._rotating_layer_indices(self.anim_face)

        # Draw non-rotating cubies first
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                for z in (-1, 0, 1):
                    idx = (x, y, z)
                    if self.animating and idx in rotating_indices:
                        continue
                    self._draw_one_cubie(x, y, z)

        # Draw rotating cubies with extra transform
        if self.animating and rotating_indices:
            glPushMatrix()
            glRotatef(angle, *axis)
            for (x, y, z) in rotating_indices:
                self._draw_one_cubie(x, y, z)
            glPopMatrix()

    def _axis_for_face(self, face: str):
        face = face.upper()
        if face in ('F', 'B'):
            return (0, 0, 1)  # rotate around Z
        if face in ('U', 'D'):
            return (0, 1, 0)  # around Y
        return (1, 0, 0)      # L/R around X

    def _rotating_layer_indices(self, face: str):
        """
        Return set of (x,y,z) indices for the layer being turned.
        Coordinates are in {-1,0,1}.
        """
        s = set()
        if face == 'F':
            for x in (-1, 0, 1):
                for y in (-1, 0, 1):
                    s.add((x, y, +1))
        elif face == 'B':
            for x in (-1, 0, 1):
                for y in (-1, 0, 1):
                    s.add((x, y, -1))
        elif face == 'U':
            for x in (-1, 0, 1):
                for z in (-1, 0, 1):
                    s.add((x, +1, z))
        elif face == 'D':
            for x in (-1, 0, 1):
                for z in (-1, 0, 1):
                    s.add((x, -1, z))
        elif face == 'R':
            for y in (-1, 0, 1):
                for z in (-1, 0, 1):
                    s.add((+1, y, z))
        elif face == 'L':
            for y in (-1, 0, 1):
                for z in (-1, 0, 1):
                    s.add((-1, y, z))
        return s

    def _draw_one_cubie(self, x, y, z):
        """Draw a single cubie (beveled) at (x,y,z) with sticker colors from engine state."""
        # Skip center internal cubie
        if x == 0 and y == 0 and z == 0:
            return

        # Convert index to world position
        spacing = self.cubie_size + self.cubie_gap
        px = x * spacing
        py = y * spacing
        pz = z * spacing

        # Colors in order: [front, back, up, down, left, right] or None if not visible
        colors = self._colors_for_cubie(x, y, z)
        self._draw_beveled_cubie(px, py, pz, colors)

    # ---------- Mapping: engine state -> cubie sticker colors ----------

    def _colors_for_cubie(self, x, y, z):
        """
        Return color-name list for [Front, Back, Up, Down, Left, Right].
        Any internal/non-visible face returns None.
        """
        get = self._get_face_color_name  # shorthand
        colors = [None, None, None, None, None, None]

        # FRONT (z=+1)
        if z == +1:
            r = 1 - y
            c = x + 1
            colors[0] = get(F.FRONT, r, c)

        # BACK (z=-1)
        if z == -1:
            r = 1 - y
            c = 1 - x
            colors[1] = get(F.BACK, r, c)

        # UP (y=+1)
        if y == +1:
            r = 1 - z
            c = x + 1
            colors[2] = get(F.UP, r, c)

        # DOWN (y=-1)
        if y == -1:
            r = z + 1
            c = x + 1  # if mirrored, flip to c = 1 - x
            colors[3] = get(F.DOWN, r, c)

        # LEFT (x=-1)
        if x == -1:
            r = 1 - y
            c = 1 - z
            colors[4] = get(F.LEFT, r, c)

        # RIGHT (x=+1)
        if x == +1:
            r = 1 - y
            c = z + 1
            colors[5] = get(F.RIGHT, r, c)

        return colors

    def _get_face_color_name(self, face_enum, r, c):
        """
        Translate engine Color enum to material key string (e.g., 'green').
        """
        enum_val = int(self.cube.cube[face_enum][r, c].value)
        return self._color_name.get(enum_val, "black")

    # ---------- Drawing: beveled cubie ----------

    def _setup_materials(self):
        # Shiny plastic materials
        self.materials = {
            'white':    {'diffuse': (1.0, 1.0, 1.0), 'specular': (0.7, 0.7, 0.7), 'shininess': 100},
            'yellow':   {'diffuse': (1.0, 1.0, 0.0), 'specular': (0.7, 0.7, 0.0), 'shininess': 100},
            'red':      {'diffuse': (1.0, 0.0, 0.0), 'specular': (0.7, 0.0, 0.0), 'shininess': 100},
            'orange':   {'diffuse': (1.0, 0.5, 0.0), 'specular': (0.7, 0.35, 0.0), 'shininess': 100},
            'blue':     {'diffuse': (0.0, 0.0, 1.0), 'specular': (0.0, 0.0, 0.7), 'shininess': 100},
            'green':    {'diffuse': (0.0, 1.0, 0.0), 'specular': (0.0, 0.7, 0.0), 'shininess': 100},
            'black':    {'diffuse': (0.1, 0.1, 0.1), 'specular': (0.3, 0.3, 0.3), 'shininess': 50},
            'highlight':{'diffuse': (1.0, 1.0, 0.5), 'specular': (1.0, 1.0, 0.5), 'shininess': 120},
        }

    def _apply_material(self, key: str):
        """
        Lit material path (kept for future use or if you switch back).
        Not used for stickers in Option 1, but used elsewhere if needed.
        """
        m = self.materials.get(key, self.materials['black'])
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,  m['diffuse'])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, m['specular'])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, m['shininess'])

    # --- NEW: unlit color for stickers ---
    def _apply_unlit_color(self, key: str):
        """
        Applies a pure, unlit color (lighting disabled) so the color is always visible.
        """
        m = self.materials.get(key, self.materials['black'])
        r, g, b = m['diffuse']
        glDisable(GL_LIGHTING)   # <- critical: ignore lights, use pure color
        glColor3f(r, g, b)

    def _draw_beveled_cubie(self, x, y, z, colors):
        """
        Draw a single cubie at world position (x,y,z).
        colors: [front, back, up, down, left, right] with material keys or None.
        """
        half = self.cubie_size / 2.0 - self.cubie_gap
        bevel = self.bevel_size

        glPushMatrix()
        glTranslatef(x, y, z)

        # Define each face (normal, 4 vertices in local cubie space)
        faces = [
            # index, normal, quad vertices, color-key
            (0, (0, 0, 1),  [(-half+bevel, -half+bevel,  half), ( half-bevel, -half+bevel,  half),
                             ( half-bevel,  half-bevel,  half), (-half+bevel,  half-bevel,  half)]),
            (1, (0, 0, -1), [(-half+bevel, -half+bevel, -half), (-half+bevel,  half-bevel, -half),
                             ( half-bevel,  half-bevel, -half), ( half-bevel, -half+bevel, -half)]),
            (2, (0, 1, 0),  [(-half+bevel,  half, -half+bevel), (-half+bevel,  half,  half-bevel),
                             ( half-bevel,  half,  half-bevel), ( half-bevel,  half, -half+bevel)]),
            (3, (0, -1, 0), [(-half+bevel, -half, -half+bevel), ( half-bevel, -half, -half+bevel),
                             ( half-bevel, -half,  half-bevel), (-half+bevel, -half,  half-bevel)]),
            (4, (-1, 0, 0), [(-half, -half+bevel, -half+bevel), (-half, -half+bevel,  half-bevel),
                             (-half,  half-bevel,  half-bevel), (-half,  half-bevel, -half+bevel)]),
            (5, (1, 0, 0),  [( half, -half+bevel, -half+bevel), ( half,  half-bevel, -half+bevel),
                             ( half,  half-bevel,  half-bevel), ( half, -half+bevel,  half-bevel)]),
        ]

        # Draw faces with unlit colors (always visible)
        for idx, normal, verts in faces:
            col_key = colors[idx]
            if col_key:
                # Apply flat (unlit) color for the sticker quad
                self._apply_unlit_color(col_key)
                glBegin(GL_QUADS)
                glNormal3fv(normal)  # harmless here; normals are fine even with lighting off
                for vx, vy, vz in verts:
                    glVertex3f(vx, vy, vz)
                glEnd()

        # Edge lines for tactile look (already unlit)
        glDisable(GL_LIGHTING)
        glColor3f(0.06, 0.06, 0.06)
        glLineWidth(1.4)

        def _edge(a, b):
            glVertex3f(*a); glVertex3f(*b)

        glBegin(GL_LINES)
        # 12 edges of the bounding cube (slightly inset by bevel)
        a = half - bevel
        # bottom square
        _edge((-a, -a, -a), ( a, -a, -a))
        _edge(( a, -a, -a), ( a, -a,  a))
        _edge(( a, -a,  a), (-a, -a,  a))
        _edge((-a, -a,  a), (-a, -a, -a))
        # top square
        _edge((-a,  a, -a), ( a,  a, -a))
        _edge(( a,  a, -a), ( a,  a,  a))
        _edge(( a,  a,  a), (-a,  a,  a))
        _edge((-a,  a,  a), (-a,  a, -a))
        # vertical edges
        _edge((-a, -a, -a), (-a,  a, -a))
        _edge(( a, -a, -a), ( a,  a, -a))
        _edge(( a, -a,  a), ( a,  a,  a))
        _edge((-a, -a,  a), (-a,  a,  a))
        glEnd()

        # Restore lighting for the rest of the pipeline
        glEnable(GL_LIGHTING)

        glPopMatrix()

    # ---------- Overlays ----------

    def _draw_grabbed_face_highlight(self, face, twist_angle):
        """Semi-transparent highlight plane on the grabbed face."""
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        pulse = (math.sin(time.time() * 8.0) + 1.0) * 0.5
        alpha = 0.3 + 0.2 * pulse

        if abs(twist_angle) > 10:
            if twist_angle > 0:
                glColor4f(0.0, 1.0, 0.0, alpha)  # CW
            else:
                glColor4f(1.0, 0.0, 0.0, alpha)  # CCW
        else:
            glColor4f(1.0, 1.0, 0.0, alpha)

        glPushMatrix()
        size = self.cube_size / 2 + 0.08
        if face == 'U':
            glTranslatef(0,  self.cube_size/2 + 0.02, 0); glRotatef(90, 1, 0, 0)
        elif face == 'D':
            glTranslatef(0, -self.cube_size/2 - 0.02, 0); glRotatef(-90, 1, 0, 0)
        elif face == 'L':
            glTranslatef(-self.cube_size/2 - 0.02, 0, 0); glRotatef(90, 0, 1, 0)
        elif face == 'R':
            glTranslatef( self.cube_size/2 + 0.02, 0, 0); glRotatef(-90, 0, 1, 0)
        elif face == 'F':
            glTranslatef(0, 0,  self.cube_size/2 + 0.02)
        elif face == 'B':
            glTranslatef(0, 0, -self.cube_size/2 - 0.02)

        glBegin(GL_QUADS)
        glVertex3f(-size, -size, 0)
        glVertex3f( size, -size, 0)
        glVertex3f( size,  size, 0)
        glVertex3f(-size,  size, 0)
        glEnd()
        glPopMatrix()

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

    def _draw_twist_indicator(self, face, angle):
        if not face or abs(angle) < 5:
            return
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        if angle > 0:
            glColor4f(0.0, 1.0, 0.0, 0.75)
            start_angle, end_angle = 0, min(90, angle)
        else:
            glColor4f(1.0, 0.0, 0.0, 0.75)
            start_angle, end_angle = max(-90, angle), 0

        glPushMatrix()
        if face in ('U', 'D'):
            glTranslatef(0,  self.cube_size/2 + 0.25 if face == 'U' else -self.cube_size/2 - 0.25, 0)
            glRotatef(90, 1, 0, 0)
        elif face in ('L', 'R'):
            glTranslatef(self.cube_size/2 + 0.25 if face == 'R' else -self.cube_size/2 - 0.25, 0, 0)
            glRotatef(90, 0, 1, 0)
        elif face in ('F', 'B'):
            glTranslatef(0, 0,  self.cube_size/2 + 0.25 if face == 'F' else -self.cube_size/2 - 0.25)

        radius = self.cube_size/2 + 0.35
        glLineWidth(4.0)
        glBegin(GL_LINE_STRIP)
        steps = 30
        for i in range(steps + 1):
            t = start_angle + (end_angle - start_angle) * (i / steps)
            rad = math.radians(t)
            glVertex3f(radius * math.cos(rad), radius * math.sin(rad), 0)
        glEnd()
        glPopMatrix()

        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    def _draw_starfield(self):
        glDisable(GL_LIGHTING)
        glPointSize(2.0)
        glBegin(GL_POINTS)
        glColor3f(1, 1, 1)
        for x, y, z in self.stars:
            glVertex3f(x, y, z)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_ui(self, mode_text, move_count, timer, grabbed_face, twist_angle):
        """Simple 2D UI overlay using glDrawPixels from Pygame surfaces."""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.display[0], self.display[1], 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        # Mode text
        mode_surface = self.font.render(mode_text, True, (255, 255, 0))
        mode_data = pygame.image.tostring(mode_surface, "RGBA", True)
        glRasterPos2f(20, 40)
        glDrawPixels(mode_surface.get_width(), mode_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, mode_data)

        # Twist
        if grabbed_face and abs(twist_angle) > 5:
            twist_text = f"Twist: {abs(twist_angle):.0f}° ({'CW' if twist_angle > 0 else 'CCW'})"
            twist_surface = self.font.render(twist_text, True, (0, 255, 0) if twist_angle > 0 else (255, 0, 0))
            twist_data = pygame.image.tostring(twist_surface, "RGBA", True)
            glRasterPos2f(self.display[0] - 320, 40)
            glDrawPixels(twist_surface.get_width(), twist_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, twist_data)

        # Stats
        stats_y = 80
        stats = [
            f"Moves: {move_count}",
            f"Time: {timer:.1f}s",
            "Press Q to quit",
            "Grab near a face and twist!"
        ]
        for i, text in enumerate(stats):
            surf = self.small_font.render(text, True, (200, 200, 255))
            data = pygame.image.tostring(surf, "RGBA", True)
            glRasterPos2f(20, stats_y + i * 28)
            glDrawPixels(surf.get_width(), surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, data)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()