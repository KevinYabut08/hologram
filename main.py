# main.py
import cv2
import time
import numpy as np  # NEW: for blank frame fallback

from camera import Camera
from immersive_controller import ImmersiveCubeController
from renderer import ImmersiveCubeRenderer
from rubiks import RubiksCube


def main():
    print("=" * 80)
    print("🎮 PHYSICAL RUBIK'S CUBE SIMULATOR")
    print("=" * 80)
    print("\nInitializing immersive experience...")

    camera = Camera()
    controller = ImmersiveCubeController()
    renderer = ImmersiveCubeRenderer()

    # Attach cube engine to renderer
    cube = RubiksCube()
    renderer.set_cube(cube)

    print("\n✅ Ready to feel the cube!")
    print("\n" + "=" * 80)
    print("IMMERSIVE CONTROLS:")
    print("=" * 80)
    print("🖐️  OPEN HANDS: Rotate cube view")
    print("🤏 PINCH NEAR A FACE: Grab that face")
    print("🌀 TWIST HANDS: Rotate grabbed layer")
    print("🖐️  RELEASE: Snap rotation into place")
    print("=" * 80)
    print("\n⚠️  Feel the resistance, hear the clicks!")
    print("=" * 80)

    move_count = 0
    start_time = time.time()

    # NEW: fallback “no camera” frame so imshow keeps a window visible
    blank_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(blank_frame, "NO CAMERA - demo mode (press Q to quit)", (24, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # NEW: soft warm-up window—don’t exit if first frames are missing
    consecutive_none_frames = 0
    max_none_frames_before_demo_msg = 30  # ~0.5s at 60 FPS
    demo_mode_announced = False

    try:
        while True:
            # ---- Camera & hands ----
            frame, left_hand, right_hand = camera.get_frame_and_hands()

            if frame is None:
                consecutive_none_frames += 1
                # Keep running: hands become None in demo mode
                left_hand = None
                right_hand = None
                frame_to_show = blank_frame.copy()

                # Print demo-mode info once if camera keeps failing
                if (consecutive_none_frames >= max_none_frames_before_demo_msg) and not demo_mode_announced:
                    print("⚠️  No camera frames detected. Running in NO-CAMERA demo mode (hands=None).")
                    demo_mode_announced = True
            else:
                consecutive_none_frames = 0
                frame_to_show = frame

            # ---- Controller update ----
            action = controller.update(left_hand, right_hand)

            # On snapped gesture or gesture shortcut: queue a move
            if action and action.get("action") == "rotate":
                face = action["face"]          # 'F','B','U','D','L','R'
                clockwise = action["clockwise"]
                renderer.rotate_face(face, clockwise)  # starts animation & commits on finish
                move_count += 1  # NEW: count moves for HUD stats

            # ---- HUD / overlay info ----
            rotation_x, rotation_y = controller.get_rotation()
            twist_angle = controller.get_twist_angle()
            grabbed_face = controller.get_grabbed_face()
            mode_text = controller.get_mode_text()
            elapsed_time = time.time() - start_time

            # ---- Render 3D scene ----
            ok = renderer.render(
                rotation_x, rotation_y, grabbed_face, twist_angle,
                mode_text, move_count, elapsed_time
            )
            if not ok:
                print("Window requested close (pygame.QUIT).")
                break

            # ---- Camera overlay window (optional) ----
            # Only call imshow if we have a (real or blank) frame to present
            h, w = frame_to_show.shape[:2]

            cv2.putText(frame_to_show, "PHYSICAL RUBIK'S CUBE", (max(10, w // 2 - 200), 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if left_hand is not None:
                cv2.putText(frame_to_show, "LEFT: READY", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            if right_hand is not None:
                cv2.putText(frame_to_show, "RIGHT: READY", (w - 250, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if grabbed_face:
                cv2.putText(frame_to_show, f"GRABBED: {grabbed_face}", (max(10, w // 2 - 100), 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.putText(frame_to_show, f"TWIST: {twist_angle:.0f}°", (max(10, w // 2 - 80), 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.putText(frame_to_show, f"Moves: {move_count}", (w - 200, h - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame_to_show, f"Time: {elapsed_time:.1f}s", (w - 200, h - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow('Camera - Feel the Cube! (Q to quit)', frame_to_show)

            # Keyboard (cv2 window)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Key 'q' pressed. Exiting.")
                break

            # (Optional) tiny sleep to reduce CPU a bit
            # time.sleep(1/120)

    except Exception as e:
        # Surface any hidden exceptions instead of silently finishing
        import traceback
        print("❌ Unhandled exception in main loop:")
        traceback.print_exc()

    finally:
        print("\n" + "=" * 80)
        print("🎮 SESSION STATS:")
        print("=" * 80)
        print(f"Total moves: {move_count}")
        print(f"Session time: {time.time() - start_time:.1f}s")
        print("=" * 80)

        print("\n🛑 Closing immersive experience...")
        try:
            camera.release()
        except Exception:
            pass
        try:
            renderer.cleanup()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("✅ Thanks for playing with feeling! 🎮✨")


if __name__ == "__main__":
    main()