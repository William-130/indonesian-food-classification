"""
Inference script untuk Indonesian Food Classification
EfficientNet-B0 model
"""

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import pandas as pd
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from training.food_model import EfficientNetB0Food

class FoodClassifier:
    def __init__(self, model_path='food_model_efficientnet.pth', num_classes=35):
        """
        Load model dan prepare untuk inference
        
        Args:
            model_path: Path ke model weights
            num_classes: Number of classes (35)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Class names (from CSV header)
        self.class_names = [
            'asinan-jakarta', 'ayam-betutu', 'ayam-bumbu-rujak', 
            'ayam-goreng-lengkuas', 'bika-ambon', 'bir-pletok', 
            'bubur-manado', 'cendol', 'es-dawet', 'gado-gado', 
            'gudeg', 'gulai-ikan-mas', 'keladi', 'kerak-telor', 
            'klappertart', 'kolak', 'kue-lumpur', 'kunyit-asam', 
            'laksa-bogor', 'lumpia-semarang', 'mie-aceh', 'nagasari', 
            'nasi-goreng-kampung', 'papeda', 'pempek-palembang', 
            'rawon-surabaya', 'rendang', 'rujak-cingur', 
            'sate-ayam-madura', 'sate-lilit', 'sate-maranggi', 
            'soerabi', 'soto-ayam-lamongan', 'soto-banjar', 'tahu-telur'
        ]
        
        # Load model
        self.model = EfficientNetB0Food(num_classes=num_classes, pretrained=False)
        # Accept either a raw state_dict or a training checkpoint dict
        loaded = torch.load(model_path, map_location=self.device)
        if isinstance(loaded, dict) and 'model_state_dict' in loaded:
            state_dict = loaded['model_state_dict']
        else:
            state_dict = loaded
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ Model loaded successfully on {self.device}")
        print(f"✓ Ready to classify {len(self.class_names)} Indonesian foods")
    
    def predict(self, image_input, top_k=5):
        """
        Predict makanan dari image
        
        Args:
            image_input: Path ke image file (str) atau NumPy array (BGR or RGB)
            top_k: Return top-k predictions
            
        Returns:
            list: Prediction results dengan top-k classes dan confidence
        """
        # Handle different input types
        if isinstance(image_input, str):
            # Load from file path
            image = Image.open(image_input).convert('RGB')
        elif hasattr(image_input, 'shape'):  # NumPy array or tensor
            # Convert BGR (OpenCV) to RGB if needed
            import numpy as np
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                # Assume BGR (OpenCV format), convert to RGB
                image_rgb = image_input[..., ::-1].copy()
                image = Image.fromarray(image_rgb.astype('uint8'))
            else:
                image = Image.fromarray(image_input.astype('uint8'))
        else:
            raise ValueError("image_input must be file path (str) or NumPy array")
        
        # Preprocess
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities[0], top_k)
            
        # Format results as list
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'class': self.class_names[idx.item()],
                'confidence': prob.item()  # Return as float, not string
            })
        
        return predictions
    
    def predict_batch(self, image_paths):
        """Predict multiple images at once"""
        results = []
        for img_path in image_paths:
            result = self.predict(img_path)
            results.append({
                'image': os.path.basename(img_path),
                **result
            })
        return results

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python food_inference.py <image_path>")
        print("Example: python food_inference.py 'Dataset ML/food-tfk-images/IMG_6886.jpg'")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    
    print("="*60)
    print("INDONESIAN FOOD CLASSIFICATION")
    print("="*60)
    
    # Load classifier
    print("\nLoading model...")
    classifier = FoodClassifier()
    
    # Predict
    print(f"\nPredicting: {image_path}")
    print("-"*60)
    
    predictions = classifier.predict(image_path, top_k=5)

    # Display results (top-5)
    if not predictions:
        print("No predictions returned")
        return

    top = predictions[0]
    print(f"\n🍽️  Top Prediction: {top['class']}")
    print(f"✓ Confidence: {top['confidence']:.4f}")

    print(f"\n📊 Top-5 Predictions:")
    print("-"*60)
    for i, pred in enumerate(predictions, 1):
        print(f"{i}. {pred['class']:<25} {pred['confidence']:.4f}")
    
    print("="*60)

if __name__ == "__main__":
    main()
