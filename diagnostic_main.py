# diagnostic_main.py
import time
from renderer import ImmersiveCubeRenderer
from rubiks import RubiksCube
from controller import ImmersiveCubeController  # your updated controller

def diagnostic_main():
    cube = RubiksCube()
    renderer = ImmersiveCubeRenderer()
    renderer.set_cube(cube)
    ctrl = ImmersiveCubeController()

    t0 = time.time()
    rot_x, rot_y = 20.0, -30.0
    try:
        while True:
            # Slowly spin to prove the loop is alive
            rot_y += 0.35
            ok = renderer.render(
                rotation_x=rot_x,
                rotation_y=rot_y,
                grabbed_face=None,
                twist_angle=0,
                mode_text="DIAG: no camera – press window X to quit",
                move_count=0,
                timer=time.time() - t0,
            )
            if not ok:
                break
            time.sleep(1/60)
    finally:
        renderer.cleanup()

if __name__ == "__main__":
    diagnostic_main()