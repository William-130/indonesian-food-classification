"""
webcam_app.py
Desktop webcam app using OpenCV for real-time food prediction.
Uses FoodClassifier from food_inference.py to load the EfficientNet model
and run inference on frames captured from the webcam.

Usage:
    python webcam_app.py

Press 'q' to quit.
"""

import os
import sys
import time
import cv2

# Ensure current directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from food_inference import FoodClassifier


def main(camera_id=0, model_path='food_model_efficientnet.pth'):
    """Main webcam loop with real-time prediction"""
    print("="*60)
    print("WEBCAM FOOD CLASSIFIER - Real-time Prediction")
    print("="*60)
    print("\nControls:")
    print("  Press 'q' to quit")
    print("  Press 's' to save screenshot")
    print("\nStarting webcam...")
    
    # Load classifier
    classifier = FoodClassifier(model_path=model_path)
    
    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("❌ Error: Cannot open webcam")
        print("Try different camera_id: python -c \"import cv2; print(cv2.VideoCapture(0).isOpened())\"")
        return
    
    print("✓ Webcam opened successfully")
    print("\nPredicting...")
    
    fps_time = time.time()
    frame_count = 0
    temp_path = 'temp_webcam.jpg'
    top_pred = 'Waiting...'
    top_conf = '0%'
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            # Make a copy for display
            display = frame.copy()
            
            # Resize for faster inference (every 10 frames to reduce load)
            if frame_count % 10 == 0:
                small = cv2.resize(frame, (224, 224))
                cv2.imwrite(temp_path, small)
                
                # Predict
                try:
                    result = classifier.predict(temp_path, top_k=1)
                    top_pred = result['top_prediction']
                    top_conf = result['top_confidence']
                except Exception as e:
                    top_pred = 'Error'
                    top_conf = '0%'
                    print(f"Prediction error: {e}")
            
            # Overlay prediction text
            cv2.rectangle(display, (5, 5), (450, 80), (0, 0, 0), -1)
            cv2.rectangle(display, (5, 5), (450, 80), (0, 255, 0), 2)
            
            text = f"{top_pred.replace('-', ' ').title()}"
            cv2.putText(display, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            conf_text = f"Confidence: {top_conf}"
            cv2.putText(display, conf_text, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                current_time = time.time()
                fps = 30 / (current_time - fps_time)
                fps_time = current_time
            else:
                fps = 0.0
            
            # FPS counter
            if fps > 0:
                cv2.putText(display, f"FPS: {fps:.1f}", (display.shape[1] - 120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
            
            # Show frame
            cv2.imshow('Indonesian Food Classifier - Webcam', display)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n✓ Quitting...")
                break
            elif key == ord('s'):
                screenshot = f'screenshot_{int(time.time())}.jpg'
                cv2.imwrite(screenshot, display)
                print(f"✓ Screenshot saved: {screenshot}")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Remove temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        
        print("✓ Webcam closed")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n✓ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
