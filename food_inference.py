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

from food_model import EfficientNetB0Food

class FoodClassifier:
    def __init__(self, model_path='food_model_efficientnet.pth', num_classes=35):
        """
        Load model dan prepare untuk inference
        
        Args:
            model_path: Path ke model weights
            num_classes: Number of classes (35)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
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
    
    def predict(self, image_path, top_k=5):
        """
        Predict makanan dari image
        
        Args:
            image_path: Path ke image file
            top_k: Return top-k predictions
            
        Returns:
            dict: Prediction results dengan top-k classes dan confidence
        """
        # Load dan preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities[0], top_k)
            
        # Format results
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'class': self.class_names[idx.item()],
                'confidence': f"{prob.item()*100:.2f}%",
                'probability': prob.item()
            })
        
        result = {
            'top_prediction': predictions[0]['class'],
            'top_confidence': predictions[0]['confidence'],
            'all_predictions': predictions
        }
        
        return result
    
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
    
    result = classifier.predict(image_path, top_k=5)
    
    # Display results
    print(f"\n🍽️  Top Prediction: {result['top_prediction']}")
    print(f"✓ Confidence: {result['top_confidence']}")
    
    print(f"\n📊 Top-5 Predictions:")
    print("-"*60)
    for i, pred in enumerate(result['all_predictions'], 1):
        print(f"{i}. {pred['class']:<25} {pred['confidence']:>8}")
    
    print("="*60)

if __name__ == "__main__":
    main()
