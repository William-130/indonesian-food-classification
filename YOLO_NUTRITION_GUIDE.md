# 🍽️ YOLOv8 Food Detection + Nutrition System

## 📋 Overview

Sistem deteksi makanan real-time menggunakan:
- **YOLOv8** untuk object detection
- **EfficientNet-B0** untuk klasifikasi makanan (35 kelas makanan Indonesia)
- **TheFuzz** untuk fuzzy matching dengan database nutrisi
- **1347 makanan** dalam database nutrisi

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_yolo.txt
```

### 2. Test Nutrition Matcher

```bash
python nutrition_matcher.py
```

Output: `nutrition_mapping.json` dengan mapping 35 class ke database nutrisi

### 3. Run YOLOv8 Detection

```bash
python yolo_food_detector.py
```

**Controls:**
- `SPACE` - Pause/Resume detection
- `S` - Save screenshot
- `Q` - Quit

---

## 📁 File Structure

```
yolo_food_detector.py      # Main YOLOv8 detection app
nutrition_matcher.py       # Fuzzy matching nutrition database
nutrition_mapping.json     # Pre-computed nutrition mapping
requirements_yolo.txt      # Python dependencies
Dataset ML/
  └── nutrition.csv        # 1347 foods nutrition database
```

---

## 🔄 How It Works

### Step 1: Object Detection (YOLOv8)
```python
# YOLOv8 detects objects in frame
yolo_results = self.yolo.predict(frame, conf=0.5)
```

### Step 2: Food Classification (EfficientNet)
```python
# Classify cropped region
predictions = self.classifier.predict(crop, top_k=1)
food_class = predictions[0]['class']  # e.g., "rendang"
```

### Step 3: Nutrition Matching (TheFuzz)
```python
# Fuzzy match with nutrition database
nutrition = self.nutrition_matcher.get_nutrition(food_class)
# Returns: {calories: 193, proteins: 22.6, fat: 7.9, carbs: 7.8}
```

### Step 4: Display Results
- Bounding box around detected food
- Food name + confidence
- Nutrition info (kalori, protein, lemak, karbohidrat)

---

## 🎯 Nutrition Matching Examples

| Class Name | Matched Nutrition Data | Score |
|------------|----------------------|-------|
| `rendang` | Rendang sapi masakan | 100% |
| `gado-gado` | Gado-gado | 100% |
| `pempek` | Pempek adaan | 100% |
| `soto-betawi` | Soto betawi masakan | 100% |
| `ayam-goreng` | Ayam | 100% |
| `mie-aceh` | Mie aceh rebus | 100% |

**Total Matched:** 35/35 classes (100%)

---

## ⚙️ Configuration

### YOLOv8 Models

Choose model size (speed vs accuracy trade-off):

```bash
# Fastest (recommended for webcam)
python yolo_food_detector.py --yolo yolov8n.pt

# Balanced
python yolo_food_detector.py --yolo yolov8s.pt

# Most accurate (slower)
python yolo_food_detector.py --yolo yolov8m.pt
```

### Custom Camera

```bash
python yolo_food_detector.py --camera 1  # Use camera ID 1
```

### Adjust Detection Threshold

Edit `yolo_food_detector.py`:

```python
self.conf_threshold = 0.5  # Lower = more detections (less precise)
self.iou_threshold = 0.45  # IoU for Non-Max Suppression
```

---

## 📊 Performance

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1050 | RTX 3050+ |
| RAM | 8 GB | 16 GB |
| VRAM | 2 GB | 4 GB |

### FPS Benchmarks

| Model | GPU (RTX 3050) | CPU (i7-10th) |
|-------|----------------|---------------|
| YOLOv8n | ~25-30 FPS | ~8-10 FPS |
| YOLOv8s | ~20-25 FPS | ~5-7 FPS |
| YOLOv8m | ~15-18 FPS | ~3-4 FPS |

---

## 🔍 Nutrition Database

### Database Stats

- **Total Foods:** 1,347 items
- **Nutrition Info:**
  - Calories (kcal)
  - Protein (g)
  - Fat (g)
  - Carbohydrate (g)

### Example Entry

```csv
id,calories,proteins,fat,carbohydrate,name,image
193,193,22.6,7.9,7.8,Rendang sapi masakan,https://...
```

### Fuzzy Matching

Uses `thefuzz` library with `token_set_ratio` scorer:

```python
best_match = process.extractOne(
    "rendang",
    nutrition_database,
    scorer=fuzz.token_set_ratio
)
# Returns: ("Rendang sapi masakan", 100)
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'ultralytics'"

```bash
pip install ultralytics
```

### Issue: Low FPS

1. Use faster model: `yolov8n.pt`
2. Reduce camera resolution in code:
   ```python
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   ```
3. Increase detection interval (skip frames)

### Issue: No nutrition data found

- Check `nutrition_mapping.json` exists
- Run `python nutrition_matcher.py` to regenerate
- Verify `Dataset ML/nutrition.csv` path

### Issue: Camera not opening

```bash
# Try different camera IDs
python yolo_food_detector.py --camera 0
python yolo_food_detector.py --camera 1
```

---

## 📝 Example Output

```
🚀 Loading YOLOv8 Food Detection System...
📦 Loading YOLOv8 model...
🍱 Loading EfficientNet classifier...
🥗 Loading nutrition database...
✅ All systems ready!

============================================================
🍽️  YOLOv8 FOOD DETECTION + NUTRITION INFO
============================================================
Controls:
  [SPACE] - Pause/Resume detection
  [S]     - Save screenshot
  [Q]     - Quit
============================================================

FPS: 28.3
Detected: Rendang (89.2%)
  - Rendang sapi masakan
  - Cal: 193 kcal
  - Protein: 22.6g
  - Fat: 7.9g
  - Carbs: 7.8g
```

---

## 🎨 Visualization

### Bounding Box Colors

- **Green:** High confidence (>70%)
- **Yellow:** Medium confidence (50-70%)
- **Orange:** Low confidence (<50%)

### Display Format

```
┌─────────────────────────┐
│ Food Name (89.2%)       │ ← Bounding box
├─────────────────────────┤
│ Rendang sapi masakan    │
│ Cal: 193 kcal           │ ← Nutrition info
│ Protein: 22.6g          │
│ Fat: 7.9g               │
│ Carbs: 7.8g             │
└─────────────────────────┘
```

---

## 🔄 Workflow Diagram

```
Camera Frame
     ↓
[YOLOv8 Detection]
     ↓
Bounding Boxes
     ↓
[Crop & Classify with EfficientNet]
     ↓
Food Class (e.g., "rendang")
     ↓
[Fuzzy Match with Nutrition DB]
     ↓
Nutrition Info
     ↓
[Display on Frame]
     ↓
Output Video
```

---

## 📚 API Reference

### NutritionMatcher

```python
from nutrition_matcher import NutritionMatcher

matcher = NutritionMatcher('Dataset ML/nutrition.csv')

# Get nutrition info
nutrition = matcher.get_nutrition('rendang')
# Returns: {matched_name, match_score, calories, proteins, fat, carbs}

# Format as text
text = matcher.format_nutrition_text('rendang')
# Returns: "Rendang sapi masakan\nCal: 193 kcal\n..."
```

### YOLOFoodDetector

```python
from yolo_food_detector import YOLOFoodDetector

detector = YOLOFoodDetector(
    yolo_model='yolov8n.pt',
    classifier_model='food_model_efficientnet.pth',
    nutrition_csv='Dataset ML/nutrition.csv'
)

# Detect and classify
detections = detector.detect_and_classify(frame)
# Returns: [{bbox, yolo_class, food_class, nutrition}, ...]

# Run webcam
detector.run(camera_id=0)
```

---

## 🎓 Credits

- **YOLOv8:** Ultralytics (https://github.com/ultralytics/ultralytics)
- **EfficientNet:** Google Research
- **TheFuzz:** SeatGeek (https://github.com/seatgeek/thefuzz)
- **Nutrition Database:** 1,347 Indonesian foods

---

## 📄 License

This project uses:
- YOLOv8 (AGPL-3.0)
- EfficientNet (Apache 2.0)
- TheFuzz (GPL-2.0)

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Add more Indonesian food classes
- [ ] Improve nutrition matching accuracy
- [ ] Add portion size estimation
- [ ] Multi-object tracking
- [ ] Mobile app deployment

---

**Happy Food Detection!** 🍱🔍
