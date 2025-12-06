# 🍽️ Indonesian Food Classification

Deep Learning model untuk klasifikasi 35 jenis makanan Indonesia menggunakan **EfficientNet-B0**.

## 📊 Model Info

- **Architecture**: EfficientNet-B0 (Transfer Learning)
- **Classes**: 35 Indonesian Traditional Foods
- **Accuracy**: ~82-86% (Train/Dev/Test)
- **Nutrition Database**: 35 foods with complete nutrition data

## 🎯 35 Supported Foods

```
 1. Asinan Jakarta          13. Keladi               25. Pempek Palembang
 2. Ayam Betutu             14. Kerak Telor          26. Rawon Surabaya
 3. Ayam Bumbu Rujak        15. Klappertart          27. Rendang
 4. Ayam Goreng Lengkuas    16. Kolak                28. Rujak Cingur
 5. Bika Ambon              17. Kue Lumpur           29. Sate Ayam Madura
 6. Bir Pletok              18. Kunyit Asam          30. Sate Lilit
 7. Bubur Manado            19. Laksa Bogor          31. Sate Maranggi
 8. Cendol                  20. Lumpia Semarang      32. Soerabi
 9. Es Dawet                21. Mie Aceh             33. Soto Ayam Lamongan
10. Gado-gado               22. Nagasari             34. Soto Banjar
11. Gudeg                   23. Nasi Goreng Kampung  35. Tahu Telur
12. Gulai Ikan Mas          24. Papeda
```

## 📁 Structure

```
Machine Learning/
├── Dataset ML/              # Training data (dev.csv, train.csv, test.csv)
│   ├── nutrition_35_foods.csv
│   └── food-tfk-images/
├── models/                  # Trained models
│   └── food_model_efficientnet.pth
├── src/core/               # Core modules
│   ├── food_inference.py   # EfficientNet classifier
│   └── nutrition_matcher.py # Nutrition database
└── training/               # Training scripts
    ├── train_food_classification.py
    ├── food_model.py
    └── food_dataset.py
```

## 🚀 Usage

### Training

```bash
python training/train_food_classification.py
```

### Inference

```python
from src.core.food_inference import FoodClassifier
from src.core.nutrition_matcher import NutritionMatcher
from PIL import Image

# Load model
classifier = FoodClassifier(model_path='models/food_model_efficientnet.pth')
nutrition_matcher = NutritionMatcher()

# Classify image
image = Image.open('path/to/food.jpg')
result = classifier.predict(image)

# Get nutrition info
nutrition = nutrition_matcher.get_nutrition(result['class_name'])

print(f"Food: {nutrition['name']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Calories: {nutrition['calories']} kcal")
print(f"Protein: {nutrition['proteins']}g")
```

## 📦 Dependencies

```bash
pip install torch torchvision pillow pandas numpy
```

## 📄 Files

### Core
- `src/core/food_inference.py` - EfficientNet-B0 classifier
- `src/core/nutrition_matcher.py` - Nutrition database matcher

### Training
- `training/train_food_classification.py` - Main training script
- `training/food_model.py` - EfficientNet model architecture
- `training/food_dataset.py` - Dataset loader with augmentation

### Data
- `Dataset ML/nutrition_35_foods.csv` - Nutrition data (calories, protein, fat, carbs)
- `Dataset ML/{train,dev,test}.csv` - Training/validation/test splits
- `Dataset ML/food-tfk-images/` - Image dataset (35 classes)

### Models
- `models/food_model_efficientnet.pth` - Best trained model
- `models/checkpoints/` - Training checkpoints

### Results
- `results/reports/classification_report.txt` - Per-class metrics
- `results/reports/training_history_food.json` - Training history

---

**Repository**: https://github.com/William-130/indonesian-food-classification
