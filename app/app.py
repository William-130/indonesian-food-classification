"""
Flask Web App for Indonesian Food Classification
Upload image → Identify food → Show nutrition info
Using advanced trained EfficientNetB0 model (99.70% accuracy)
"""

from flask import Flask, render_template, request, jsonify
import os
import sys
import torch
from PIL import Image
import io
import base64

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model import AdvancedFoodClassifier
from src.core.nutrition_matcher import NutritionMatcher
from torchvision import transforms

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize model and nutrition matcher  
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'checkpoints', 'advanced_training', 'stage2_best_model.pth')
NUTRITION_CSV = os.path.join(BASE_DIR, 'Dataset ML', 'nutrition_35_foods.csv')

# Class names
CLASS_NAMES = [
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

print("Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AdvancedFoodClassifier(num_classes=35)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

nutrition_matcher = NutritionMatcher(NUTRITION_CSV)
print("✓ Model loaded successfully!")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
        
        # Read image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Convert image to base64 for display
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Predict
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        food_name = CLASS_NAMES[predicted_idx.item()]
        confidence = confidence.item() * 100
        
        # Get nutrition info
        nutrition = nutrition_matcher.get_nutrition(food_name)
        
        if nutrition:
            response = {
                'success': True,
                'food_name': food_name,
                'confidence': f"{confidence:.2f}%",
                'nutrition': {
                    'calories': f"{nutrition['calories']} kcal",
                    'protein': f"{nutrition['protein']} g",
                    'fat': f"{nutrition['fat']} g",
                    'carbohydrate': f"{nutrition['carbohydrate']} g"
                },
                'image': f"data:image/jpeg;base64,{img_base64}"
            }
        else:
            response = {
                'success': True,
                'food_name': food_name,
                'confidence': f"{confidence:.2f}%",
                'nutrition': None,
                'image': f"data:image/jpeg;base64,{img_base64}"
            }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'EfficientNetB0',
        'accuracy': '99.70%',
        'classes': 35
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🍽️  INDONESIAN FOOD CLASSIFICATION WEB APP")
    print("="*70)
    print(f"Model: EfficientNetB0 (Advanced 2-Stage Training)")
    print(f"Test Accuracy: 99.70%")
    print(f"Classes: 35 Indonesian Foods")
    print("="*70)
    print("\n🚀 Starting server at http://localhost:5000")
    print("   Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
