# main.py
import cv2
import time

from camera import Camera
from immersive_controller import ImmersiveCubeController
from renderer import ImmersiveCubeRenderer

# NEW: import your Python cube engine
from rubiks import RubiksCube

def main():
    print("=" * 80)
    print("🎮 PHYSICAL RUBIK'S CUBE SIMULATOR")
    print("=" * 80)
    print("\nInitializing immersive experience...")

    camera = Camera()
    controller = ImmersiveCubeController()
    renderer = ImmersiveCubeRenderer()

    # NEW: create cube and attach to renderer
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

    try:
        while True:
            frame, left_hand, right_hand = camera.get_frame_and_hands()
            if frame is None:
                break

            # Update controller with MediaPipe landmarks
            action = controller.update(left_hand, right_hand)

            # On snapped gesture:
            if action and action["action"] == "rotate":
                face = action["face"]          # 'F','B','U','D','L','R'
                clockwise = action["clockwise"]
                renderer.rotate_face(face, clockwise)  # starts animation & commits on finish

            # HUD / overlay info
            rotation_x, rotation_y = controller.get_rotation()
            twist_angle = controller.get_twist_angle()
            grabbed_face = controller.get_grabbed_face()
            mode_text = controller.get_mode_text()
            elapsed_time = time.time() - start_time

            # Draw HUD (2D net etc.)
            if not renderer.render(rotation_x, rotation_y, grabbed_face, twist_angle,
                                   mode_text, move_count, elapsed_time):
                break

            # Overlay camera text
            h, w = frame.shape[:2]
            cv2.putText(frame, "PHYSICAL RUBIK'S CUBE", (max(10, w//2 - 200), 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if left_hand:
                cv2.putText(frame, "LEFT: READY", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            if right_hand:
                cv2.putText(frame, "RIGHT: READY", (w - 250, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if grabbed_face:
                cv2.putText(frame, f"GRABBED: {grabbed_face}", (max(10, w//2 - 100), 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.putText(frame, f"TWIST: {twist_angle:.0f}°", (max(10, w//2 - 80), 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.putText(frame, f"Moves: {move_count}", (w - 200, h - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Time: {elapsed_time:.1f}s", (w - 200, h - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow('Camera - Feel the Cube! (Q to quit)', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        print("\n" + "=" * 80)
        print("🎮 SESSION STATS:")
        print("=" * 80)
        print(f"Total moves: {move_count}")
        print(f"Session time: {time.time() - start_time:.1f}s")
        print("=" * 80)

        print("\n🛑 Closing immersive experience...")
        camera.release()
        renderer.cleanup()
        cv2.destroyAllWindows()
        print("✅ Thanks for playing with feeling! 🎮✨")

if __name__ == "__main__":
    main()