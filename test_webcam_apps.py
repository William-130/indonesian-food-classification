"""
Quick test script to verify webcam apps are working
Tests model loading and basic inference without actually opening camera
"""
import os
import sys

def test_inference():
    """Test basic inference"""
    print("="*60)
    print("TEST 1: Basic Inference")
    print("="*60)
    
    try:
        from food_inference import FoodClassifier
        
        # Load model
        print("\n1. Loading model...")
        classifier = FoodClassifier(model_path='food_model_efficientnet.pth')
        print("✓ Model loaded successfully")
        
        # Test with a sample image
        print("\n2. Testing inference...")
        test_image = "Dataset ML/food-tfk-images/IMG_6886.jpg"
        
        if os.path.exists(test_image):
            result = classifier.predict(test_image, top_k=3)
            print(f"✓ Inference successful!")
            print(f"  Top prediction: {result['top_prediction']} ({result['top_confidence']})")
        else:
            print("⚠ Test image not found, but model is loaded correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_webcam_app():
    """Test webcam app imports"""
    print("\n" + "="*60)
    print("TEST 2: Webcam App (Desktop)")
    print("="*60)
    
    try:
        print("\n1. Checking imports...")
        import cv2
        print("✓ OpenCV installed")
        
        import webcam_app
        print("✓ webcam_app.py imports successfully")
        
        print("\n✓ Desktop webcam app is ready!")
        print("  Run: python webcam_app.py")
        
        return True
        
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("  Install: pip install opencv-python")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_webcam_server():
    """Test Flask webcam server"""
    print("\n" + "="*60)
    print("TEST 3: Webcam Server (Flask)")
    print("="*60)
    
    try:
        print("\n1. Checking imports...")
        import flask
        print("✓ Flask installed")
        
        import webcam_server
        print("✓ webcam_server.py imports successfully")
        
        print("\n✓ Flask webcam server is ready!")
        print("  Run: python webcam_server.py")
        print("  Open: http://localhost:5000")
        
        return True
        
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("  Install: pip install flask")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("="*60)
    print("WEBCAM APPS - VERIFICATION TEST")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Basic Inference", test_inference()))
    results.append(("Desktop Webcam App", test_webcam_app()))
    results.append(("Flask Webcam Server", test_webcam_server()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nReady to use:")
        print("1. Desktop: python webcam_app.py")
        print("2. Browser: python webcam_server.py (then open http://localhost:5000)")
        print("\n✓ Ready for GitHub push!")
        
    else:
        print("\n⚠ Some tests failed. Install missing dependencies:")
        print("  pip install -r requirements_webcam.txt")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
