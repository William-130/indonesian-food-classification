"""
YOLOv8 Food Detection + Classification + Nutrition Info
Real-time webcam application with object detection and nutrition display
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from food_inference import FoodClassifier
from nutrition_matcher import NutritionMatcher
import time


class YOLOFoodDetector:
    def __init__(
        self,
        yolo_model='yolov8n.pt',  # YOLOv8 nano (fastest)
        classifier_model='food_model_efficientnet.pth',
        nutrition_csv='Dataset ML/nutrition.csv'
    ):
        """
        Initialize YOLOv8 detector + EfficientNet classifier + Nutrition matcher
        
        Args:
            yolo_model: YOLOv8 model path (yolov8n/s/m/l/x)
            classifier_model: EfficientNet model path
            nutrition_csv: Nutrition database CSV path
        """
        print("🚀 Loading YOLOv8 Food Detection System...")
        
        # Load YOLOv8 for object detection
        print("📦 Loading YOLOv8 model...")
        self.yolo = YOLO(yolo_model)
        
        # Load EfficientNet for food classification
        print("🍱 Loading EfficientNet classifier...")
        self.classifier = FoodClassifier(classifier_model)
        
        # Load nutrition matcher
        print("🥗 Loading nutrition database...")
        self.nutrition_matcher = NutritionMatcher(nutrition_csv)
        
        # Detection settings
        self.conf_threshold = 0.5  # YOLOv8 confidence threshold
        self.iou_threshold = 0.45  # NMS IOU threshold
        
        print("✅ All systems ready!\n")
    
    def detect_and_classify(self, frame):
        """
        Detect objects with YOLOv8, classify food, get nutrition info
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            list: Detections with [x1, y1, x2, y2, class_name, confidence, nutrition]
        """
        results = []
        
        # YOLOv8 detection (looking for food items)
        yolo_results = self.yolo.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        # Process each detection
        for box in yolo_results.boxes:
            # Get bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            yolo_class = int(box.cls[0])
            yolo_name = yolo_results.names[yolo_class]
            
            # Only process if detected as food-related object
            # YOLOv8 COCO classes: bowl, cup, fork, knife, spoon, banana, apple, etc.
            # We'll classify any object for now (can filter later)
            
            # Crop detected region
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            # Classify the cropped food image
            try:
                predictions = self.classifier.predict(crop, top_k=1)
                
                if predictions:
                    food_class = predictions[0]['class']
                    food_conf = predictions[0]['confidence']
                    
                    # Get nutrition information
                    nutrition = self.nutrition_matcher.get_nutrition(food_class)
                    
                    results.append({
                        'bbox': [x1, y1, x2, y2],
                        'yolo_class': yolo_name,
                        'yolo_conf': conf,
                        'food_class': food_class,
                        'food_conf': food_conf,
                        'nutrition': nutrition
                    })
            except Exception as e:
                print(f"⚠️  Classification error: {e}")
                continue
        
        return results
    
    def draw_results(self, frame, detections):
        """
        Draw bounding boxes and nutrition info on frame
        
        Args:
            frame: Input frame
            detections: Detection results
            
        Returns:
            frame: Annotated frame
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            food_class = det['food_class']
            food_conf = det['food_conf']
            nutrition = det['nutrition']
            
            # Choose color based on confidence
            if food_conf > 0.7:
                color = (0, 255, 0)  # Green
            elif food_conf > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare text
            food_name = food_class.replace('-', ' ').title()
            
            if nutrition:
                # Main label
                label = f"{food_name} ({food_conf*100:.1f}%)"
                
                # Nutrition info
                nutri_text = [
                    f"{nutrition['matched_name']}",
                    f"Cal: {nutrition['calories']:.0f} kcal",
                    f"Protein: {nutrition['proteins']:.1f}g",
                    f"Fat: {nutrition['fat']:.1f}g",
                    f"Carbs: {nutrition['carbohydrate']:.1f}g"
                ]
            else:
                label = f"{food_name} ({food_conf*100:.1f}%)"
                nutri_text = ["No nutrition data"]
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw nutrition info box (bottom of bbox)
            nutri_y = y2 + 20
            for i, text in enumerate(nutri_text):
                y_pos = nutri_y + (i * 25)
                
                # Background
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y_pos - text_h - 5), (x1 + text_w + 10, y_pos + 5), (0, 0, 0), -1)
                
                # Text
                cv2.putText(frame, text, (x1 + 5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self, camera_id=0):
        """
        Run real-time detection on webcam
        
        Args:
            camera_id: Camera device ID (0 for default webcam)
        """
        print("🎥 Starting webcam...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Error: Cannot open camera")
            return
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "=" * 60)
        print("🍽️  YOLOv8 FOOD DETECTION + NUTRITION INFO")
        print("=" * 60)
        print("Controls:")
        print("  [SPACE] - Pause/Resume detection")
        print("  [S]     - Save screenshot")
        print("  [Q]     - Quit")
        print("=" * 60 + "\n")
        
        fps_time = time.time()
        frame_count = 0
        fps = 0
        paused = False
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    fps = 10 / (time.time() - fps_time)
                    fps_time = time.time()
                
                # Process frame (only if not paused)
                if not paused:
                    detections = self.detect_and_classify(frame)
                    frame = self.draw_results(frame, detections)
                else:
                    # Show pause indicator
                    cv2.putText(frame, "PAUSED", (frame.shape[1] - 150, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Draw FPS
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('YOLOv8 Food Detection + Nutrition', frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Exiting...")
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
                elif key == ord('s'):
                    filename = f'food_detection_{int(time.time())}.jpg'
                    cv2.imwrite(filename, frame)
                    print(f"📸 Screenshot saved: {filename}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\n✅ Webcam closed")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8 Food Detection + Nutrition')
    parser.add_argument('--yolo', type=str, default='yolov8n.pt',
                       help='YOLOv8 model (n/s/m/l/x)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = YOLOFoodDetector(yolo_model=args.yolo)
    
    # Run detection
    detector.run(camera_id=args.camera)


if __name__ == "__main__":
    main()
