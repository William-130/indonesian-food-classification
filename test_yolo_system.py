"""
Quick test script for YOLOv8 Food Detection System
Tests all components without requiring webcam
"""

import sys
import cv2
import numpy as np

print("=" * 70)
print(" YOLOv8 FOOD DETECTION SYSTEM - COMPONENT TEST")
print("=" * 70)
print()

# Test 1: Import checks
print("TEST 1: Checking imports...")
print("-" * 70)

try:
    from ultralytics import YOLO
    print("✓ ultralytics (YOLOv8)")
except ImportError as e:
    print(f"✗ ultralytics: {e}")
    sys.exit(1)

try:
    from thefuzz import fuzz, process
    print("✓ thefuzz (fuzzy matching)")
except ImportError as e:
    print(f"✗ thefuzz: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print("✓ pandas")
except ImportError as e:
    print(f"✗ pandas: {e}")
    sys.exit(1)

try:
    from food_inference import FoodClassifier
    print("✓ food_inference")
except ImportError as e:
    print(f"✗ food_inference: {e}")
    sys.exit(1)

print()

# Test 2: Nutrition Matcher
print("TEST 2: Testing Nutrition Matcher...")
print("-" * 70)

try:
    from nutrition_matcher import NutritionMatcher
    
    matcher = NutritionMatcher('Dataset ML/nutrition.csv')
    print(f"✓ Loaded nutrition database: {len(matcher.df)} foods")
    print(f"✓ Mapped {len([v for v in matcher.nutrition_map.values() if v])}/35 classes")
    
    # Test query
    nutrition = matcher.get_nutrition('rendang')
    if nutrition:
        print(f"✓ Sample query (rendang):")
        print(f"  - Matched: {nutrition['matched_name']}")
        print(f"  - Calories: {nutrition['calories']} kcal")
        print(f"  - Match score: {nutrition['match_score']}%")
    
except Exception as e:
    print(f"✗ Nutrition Matcher failed: {e}")
    sys.exit(1)

print()

# Test 3: YOLOv8 Model Loading
print("TEST 3: Testing YOLOv8 Model...")
print("-" * 70)

try:
    print("Loading YOLOv8n (will download if not cached)...")
    yolo = YOLO('yolov8n.pt')
    print("✓ YOLOv8n model loaded")
    
    # Test inference on dummy image
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    results = yolo.predict(dummy_img, verbose=False)
    print(f"✓ YOLOv8 inference works (detected {len(results[0].boxes)} objects in dummy image)")
    
except Exception as e:
    print(f"✗ YOLOv8 loading failed: {e}")
    sys.exit(1)

print()

# Test 4: EfficientNet Classifier
print("TEST 4: Testing EfficientNet Classifier...")
print("-" * 70)

try:
    classifier = FoodClassifier('food_model_efficientnet.pth')
    print(f"✓ EfficientNet loaded (device: {classifier.device})")
    print(f"✓ Can classify {classifier.num_classes} classes")
    
    # Test with dummy image
    dummy_food = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    predictions = classifier.predict(dummy_food, top_k=1)
    if predictions:
        print(f"✓ Classification works (predicted: {predictions[0]['class']})")
    
except Exception as e:
    print(f"✗ EfficientNet loading failed: {e}")
    sys.exit(1)

print()

# Test 5: Full System Integration
print("TEST 5: Testing Full System Integration...")
print("-" * 70)

try:
    from yolo_food_detector import YOLOFoodDetector
    
    print("Initializing YOLOFoodDetector...")
    detector = YOLOFoodDetector(
        yolo_model='yolov8n.pt',
        classifier_model='food_model_efficientnet.pth',
        nutrition_csv='Dataset ML/nutrition.csv'
    )
    print("✓ All components integrated successfully")
    
    # Test detection on dummy frame
    test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    detections = detector.detect_and_classify(test_frame)
    print(f"✓ Detection pipeline works (found {len(detections)} objects)")
    
except Exception as e:
    print(f"✗ Integration failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Summary
print("=" * 70)
print(" ✅ ALL TESTS PASSED!")
print("=" * 70)
print()
print("System is ready to use. Run:")
print("  python yolo_food_detector.py")
print()
print("Or for specific YOLO model:")
print("  python yolo_food_detector.py --yolo yolov8s.pt")
print()
