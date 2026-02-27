import cv2
import time
from camera import Camera
from immersive_controller import ImmersiveCubeController
from renderer import ImmersiveCubeRenderer  # Use new renderer

def main():
    print("=" * 80)
    print("🎮 PHYSICAL RUBIK'S CUBE SIMULATOR")
    print("=" * 80)
    print("\nInitializing immersive experience...")
    
    # Initialize components
    camera = Camera()
    controller = ImmersiveCubeController()
    renderer = ImmersiveCubeRenderer()
    
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
            # Get camera frame
            frame, left_hand, right_hand = camera.get_frame_and_hands()
            
            if frame is None:
                break
            
            # Update immersive controller
            action = controller.update(left_hand, right_hand)
            
            # Handle rotation actions
            if action and action["action"] == "rotate":
                move_count += 1
                prime = "" if action["clockwise"] else "'"
                print(f"🔧 SNAP! Move {move_count}: {action['face']}{prime} ({action['angle']:.0f}°)")
                # Here you would trigger the cube rotation
                # renderer.rotate_face(action["face"], action["clockwise"])
            
            # Get current state
            rotation_x, rotation_y = controller.get_rotation()
            twist_angle = controller.get_twist_angle()
            grabbed_face = controller.get_grabbed_face()
            mode_text = controller.get_mode_text()
            elapsed_time = time.time() - start_time
            
            # Render with physical feedback
            if not renderer.render(rotation_x, rotation_y, grabbed_face, twist_angle,
                                 mode_text, move_count, elapsed_time):
                break
            
            # Draw hand feedback on camera
            h, w = frame.shape[:2]
            
            # Draw simple instructions
            cv2.putText(frame, "PHYSICAL RUBIK'S CUBE", (w//2 - 200, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Draw hand status
            if left_hand:
                cv2.putText(frame, "LEFT: READY", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            if right_hand:
                cv2.putText(frame, "RIGHT: READY", (w - 250, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw grabbed face info
            if grabbed_face:
                cv2.putText(frame, f"GRABBED: {grabbed_face}", (w//2 - 100, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.putText(frame, f"TWIST: {twist_angle:.0f}°", (w//2 - 80, 190),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Draw stats
            cv2.putText(frame, f"Moves: {move_count}", (w - 200, h - 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Time: {elapsed_time:.1f}s", (w - 200, h - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show camera feed
            cv2.imshow('Camera - Feel the Cube! (Q to quit)', frame)
            
            # Check for quit
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