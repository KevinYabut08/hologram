# main.py
import sys
import math
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_ESCAPE
from pygame import K_LEFT, K_RIGHT, K_UP, K_DOWN, K_PAGEUP, K_PAGEDOWN
from pygame import K_LEFTBRACKET, K_RIGHTBRACKET, K_SEMICOLON, K_QUOTE, K_m, K_f, K_c

from OpenGL.GL import *
from OpenGL.GLU import *

from camera import Camera
from rubiks import RubiksCube, Face
from hand_model import HandSpaceMapper, ArticulatedHandModelGL
from immersive_controller import ImmersiveCubeController, MappingConfig

# --- Rubik's cube rendering (stickers) ---
FACE_TRANSFORMS = {
    'F': (0, 0, 1,   0,   0,   0),
    'B': (0, 0, -1,  0, 180,   0),
    'U': (0, 1, 0, -90,   0,   0),
    'D': (0,-1, 0,  90,   0,   0),
    'L': (-1,0, 0,   0,  90,   0),
    'R': (1, 0, 0,   0, -90,   0),
}
FACE_ENUM_MAP = {
    'F': Face.FRONT, 'B': Face.BACK, 'U': Face.UP,
    'D': Face.DOWN,  'L': Face.LEFT, 'R': Face.RIGHT,
}

def draw_face_3x3(colors_3x3, highlight=False):
    size = 2.0
    tile = size / 3.0
    z = 0.0
    border = 0.02
    for i in range(3):
        for j in range(3):
            r, g, b = colors_3x3[i][j]
            glColor3f(r, g, b)
            cx = -size/2 + (j + 0.5) * tile
            cy =  size/2 - (i + 0.5) * tile
            half = tile/2 - border
            glBegin(GL_QUADS)
            glVertex3f(cx - half, cy - half, z)
            glVertex3f(cx + half, cy - half, z)
            glVertex3f(cx + half, cy + half, z)
            glVertex3f(cx - half, cy + half, z)
            glEnd()
    # outline
    glColor3f(0.1, 0.1, 0.1 if not highlight else 0.0)
    glLineWidth(2.0 if not highlight else 4.0)
    glBegin(GL_LINE_LOOP)
    glVertex3f(-1, -1, z)
    glVertex3f( 1, -1, z)
    glVertex3f( 1,  1, z)
    glVertex3f(-1,  1, z)
    glEnd()

def draw_rubiks_cube(cube: RubiksCube, rot_x=0.0, rot_y=0.0, highlight_face=None):
    glPushMatrix()
    glRotatef(rot_x, 1, 0, 0)
    glRotatef(rot_y, 0, 1, 0)
    for face_char, (tx, ty, tz, rx, ry, rz) in FACE_TRANSFORMS.items():
        glPushMatrix()
        glTranslatef(tx, ty, tz)
        glRotatef(rx, 1, 0, 0)
        glRotatef(ry, 0, 1, 0)
        glRotatef(rz, 0, 0, 1)
        colors = cube.get_face_colors(FACE_ENUM_MAP[face_char])
        draw_face_3x3(colors, highlight=(highlight_face == face_char))
        glPopMatrix()
    glPopMatrix()

def init_gl(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, width/float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.05, 0.06, 0.08, 1.0)
    # optional nicer lines
    try:
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    except Exception:
        pass

# --- Share mapper between renderer & controller ---
def mapper_to_config(mapper: HandSpaceMapper) -> MappingConfig:
    return MappingConfig(
        mirror_x=mapper.mirror_x,
        sx=mapper.sx, sy=mapper.sy, sz=mapper.sz,
        offset_x=mapper.offset_x, offset_y=mapper.offset_y, offset_z=mapper.offset_z,
        depth_policy=mapper.depth_policy,
        r_shell_max=mapper.r_shell_max,
        fixed_front_z=mapper.fixed_front_z,
        front_bias_z=mapper.front_bias_z,
        min_z_from_cube=mapper.min_z_from_cube
    )

def reapply_mapping(controller: ImmersiveCubeController, mapper: HandSpaceMapper):
    controller.mapping = mapper_to_config(mapper)

def calibrate_center_with_tip(mapper: HandSpaceMapper, controller: ImmersiveCubeController, current_tip_world):
    """
    Nudge offsets so the current tip ends up near the cube center (0,0,z_target)
    while keeping z slightly in front of the cube (z_target ~ fixed_front or ~1.12).
    """
    if not current_tip_world:
        return
    tx, ty, tz = current_tip_world
    z_target = mapper.fixed_front_z if mapper.depth_policy == 'fixed_front' else 1.12
    mapper.offset_x += -tx
    mapper.offset_y += -ty
    mapper.offset_z += (z_target - tz)
    reapply_mapping(controller, mapper)

def main():
    pygame.init()
    win_w, win_h = 1280, 720
    pygame.display.set_mode((win_w, win_h), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Rubik's Cube + Solid 3D Hand (Contact)")
    init_gl(win_w, win_h)

    # Camera
    cam = Camera(src=0, width=1280, height=720, mirror=True)

    # Shared mapping & depth policy (keeps hand in front of the cube, not inside)

    mapper = HandSpaceMapper(
        mirror_x=False, sx=4.2, sy=4.2, sz=2.2,
        r_shell_max=1.65, fixed_front_z=1.12,
        front_bias_z=0.15, min_z_from_cube=1.02
    )

    controller = ImmersiveCubeController(mapping=mapper_to_config(mapper))
    cube = RubiksCube()

    hand_model_left  = ArticulatedHandModelGL(mapper=mapper, joint_radius=0.055, bone_radius=0.045, color=(0.25, 0.80, 1.00))
    hand_model_right = ArticulatedHandModelGL(mapper=mapper, joint_radius=0.055, bone_radius=0.045, color=(0.95, 0.72, 0.25))


    # Scene camera transform
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, -7.5)  # a bit farther to give Z space

    clock = pygame.time.Clock()
    running = True

    # text overlay (optional) - quick debug
    font = None
    try:
        font = pygame.font.SysFont("consolas", 16)
    except Exception:
        pass

    left_lm = right_lm = None

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                running = False
            elif event.type == KEYDOWN:
                # Mirror toggle
                if event.key == K_m:
                    mapper.mirror_x = not mapper.mirror_x
                    reapply_mapping(controller, mapper)
                # Fine offsets
                elif event.key == K_LEFT:
                    mapper.offset_x -= 0.05; reapply_mapping(controller, mapper)
                elif event.key == K_RIGHT:
                    mapper.offset_x += 0.05; reapply_mapping(controller, mapper)
                elif event.key == K_UP:
                    mapper.offset_y += 0.05; reapply_mapping(controller, mapper)
                elif event.key == K_DOWN:
                    mapper.offset_y -= 0.05; reapply_mapping(controller, mapper)
                elif event.key == K_PAGEUP:
                    mapper.offset_z += 0.05; reapply_mapping(controller, mapper)
                elif event.key == K_PAGEDOWN:
                    mapper.offset_z -= 0.05; reapply_mapping(controller, mapper)
                # Scale tweaks ([/], ;/' for z)
                elif event.key == K_LEFTBRACKET:
                    mapper.sx -= 0.1; mapper.sy -= 0.1; reapply_mapping(controller, mapper)
                elif event.key == K_RIGHTBRACKET:
                    mapper.sx += 0.1; mapper.sy += 0.1; reapply_mapping(controller, mapper)
                elif event.key == K_SEMICOLON:
                    mapper.sz = max(1.6, mapper.sz - 0.1); reapply_mapping(controller, mapper)
                elif event.key == K_QUOTE:
                    mapper.sz = min(3.6, mapper.sz + 0.1); reapply_mapping(controller, mapper)
                # Depth policy toggle
                elif event.key == K_f:
                    mapper.depth_policy = 'fixed_front' if mapper.depth_policy == 'radial_clamp' else 'radial_clamp'
                    reapply_mapping(controller, mapper)
                # Quick center calibration
                elif event.key == K_c:
                    # use current (mapped) tip
                    lt = controller.detect_index_tip_position(left_lm) if left_lm else None
                    rt = controller.detect_index_tip_position(right_lm) if right_lm else None
                    calibrate_center_with_tip(mapper, controller, lt or rt)

        # Camera + MediaPipe
        _, left_lm, right_lm = cam.get_frame_and_landmarks()

        # Controller -> may return a rotate action when releasing a twist
        action = controller.update(left_lm, right_lm)
        if action and action.get('action') == 'rotate':
            face = action['face']   # 'R','L','U','D','F','B'
            cw = action['clockwise']
            move = f"{face}" + ("" if cw else "'")
            cube.perform_move(move)

        # Hover (for highlight) from mapped fingertip
        left_tip = controller.detect_index_tip_position(left_lm) if left_lm else None
        right_tip = controller.detect_index_tip_position(right_lm) if right_lm else None
        probe = left_tip or right_tip
        hover = controller.get_hover_face_from_tip_world(probe)

        # Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glPushMatrix()
        rot_x, rot_y = controller.get_rotation()
        draw_rubiks_cube(cube, rot_x=rot_x, rot_y=rot_y, highlight_face=hover or controller.get_grabbed_face())
        glPopMatrix()

        # Draw 3D solid hands (kept near front by mapper)

        hand_model_left.draw(left_lm)
        hand_model_right.draw(right_lm)


        # optional overlay (mode text)
        if font:
            txt = controller.get_mode_text()
            surf = font.render(txt, True, (230, 230, 230))
            # blit over a Pygame Surface then to screen: need to disable depth test for 2D overlay
            glDisable(GL_DEPTH_TEST)
            data = pygame.image.tostring(surf, "RGBA", True)
            w, h = surf.get_size()
            glWindowPos2i(10, 10)
            glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, data)
            glEnable(GL_DEPTH_TEST)

        pygame.display.flip()
        clock.tick(60)

    cam.release()
    pygame.quit()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Fatal error:', e)
        pygame.quit()
        sys.exit(1)
