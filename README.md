# 🍽️ Indonesian Food Classification with EfficientNet-B0

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Deep Learning model untuk klasifikasi 35 jenis makanan Indonesia menggunakan **EfficientNet-B0** dengan transfer learning. Proyek ini mencakup training pipeline lengkap, evaluasi dengan confusion matrix, dan aplikasi real-time computer vision.

## 📊 Features

- ✅ **35 Classes** makanan Indonesia traditional
- ✅ **EfficientNet-B0** dengan transfer learning (ImageNet pre-trained)
- ✅ **Anti-Overfitting Techniques**: MixUp, Label Smoothing, Weight Decay, Early Stopping, Freeze-Unfreeze Strategy
- ✅ **Comprehensive Evaluation**: Confusion Matrix, Per-Class Accuracy, Top-K Accuracy, Classification Report
- ✅ **Real-time Inference**: 
  - Desktop App (OpenCV)
  - Web-based App (Flask + Browser Webcam)
- ✅ **Training Monitoring**: Live curves, metrics tracking, checkpointing

---

## 🎯 Supported Foods (35 Classes)

| No | Makanan | No | Makanan | No | Makanan |
|----|---------|----|---------|----|---------|
| 1 | Asinan Jakarta | 13 | Kerak Telor | 25 | Pempek Palembang |
| 2 | Ayam Betutu | 14 | Klappertart | 26 | Rawon Surabaya |
| 3 | Ayam Bumbu Rujak | 15 | Kolak | 27 | Rendang |
| 4 | Ayam Goreng Lengkuas | 16 | Kue Lumpur | 28 | Rujak Cingur |
| 5 | Bika Ambon | 17 | Kunyit Asam | 29 | Sate Ayam Madura |
| 6 | Bir Pletok | 18 | Laksa Bogor | 30 | Sate Lilit |
| 7 | Bubur Manado | 19 | Lumpia Semarang | 31 | Sate Maranggi |
| 8 | Cendol | 20 | Mie Aceh | 32 | Soerabi |
| 9 | Es Dawet | 21 | Nagasari | 33 | Soto Ayam Lamongan |
| 10 | Gado-gado | 22 | Nasi Goreng Kampung | 34 | Soto Banjar |
| 11 | Gudeg | 23 | Papeda | 35 | Tahu Telur |
| 12 | Gulai Ikan Mas | 24 | Keladi | | |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, RTX 3050 or better)
- Webcam (for real-time apps)

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/indonesian-food-classification.git
cd indonesian-food-classification

# Install dependencies
pip install torch torchvision pillow numpy pandas matplotlib scikit-learn tqdm seaborn opencv-python flask

# Or use requirements file
pip install -r requirements_webcam.txt
```

### Download Model

Model sudah trained dan tersimpan di `food_model_efficientnet.pth` (96MB).

---

## 💻 Usage

### 1. **Single Image Inference**

```bash
python food_inference.py "path/to/image.jpg"
```

**Output:**
```
🍽️  Top Prediction: rendang
✓ Confidence: 94.32%

📊 Top-5 Predictions:
------------------------------------------------------------
1. rendang                     94.32%
2. gulai-ikan-mas              3.21%
3. rawon-surabaya              1.45%
4. ayam-bumbu-rujak            0.67%
5. sate-lilit                  0.35%
```

### 2. **Real-time Desktop App (OpenCV)**

```bash
python webcam_app.py
```

- Webcam window terbuka dengan overlay prediction dan FPS
- Press **'q'** to quit
- GPU-accelerated inference (if CUDA available)

**Screenshot:**
```
┌─────────────────────────────┐
│ [Webcam Feed]              │
│ Rendang (92.45%)           │
│ FPS: 8.3                   │
└─────────────────────────────┘
```

### 3. **Web-based App (Browser + Flask)**

```bash
python webcam_server.py
```

- Open browser: **http://localhost:5000**
- Click **"Start"** untuk activate webcam
- Prediction updates setiap ~1 detik
- No installation needed on client side

---

## 📈 Training

### Configuration

Training configuration dengan anti-overfitting setup:

```python
config = {
    'model_name': 'EfficientNet-B0',
    'num_epochs': 50,
    'batch_size': 64,              # Large batch for stability
    'learning_rate': 0.0001,       # Conservative LR
    'weight_decay': 5e-4,          # Strong L2 regularization
    'early_stop_patience': 10,     # Auto-stop if no improvement
    'mixup': True,                 # MixUp augmentation
    'freeze_backbone_epochs': 5    # Train classifier head first
}
```

### Run Training

```bash
python train_food_classification.py
```

**Training Process:**
1. **Epoch 1-5**: Freeze backbone, train classifier head only
2. **Epoch 6+**: Unfreeze backbone, fine-tune entire model
3. **Auto-stop**: When validation accuracy plateaus (patience=10)

### Training Output

```
Epoch 25 Summary:
  Train Loss: 0.2341 | Train Acc: 86.32%
  Dev Loss: 0.2789 | Dev Acc: 83.71%
  Learning Rate: 0.000050
  ✓ Best model saved! (Dev Acc: 83.71%)
```

**Generated Files:**
- `food_model_efficientnet.pth` - Main model weights
- `confusion_matrix.png` - Confusion matrix visualization
- `training_curves.png` - Loss & accuracy curves
- `per_class_accuracy.png` - Per-class performance
- `classification_report.txt` - Detailed metrics

---

## 📊 Model Architecture

```
EfficientNet-B0 (ImageNet Pre-trained)
├── Backbone: EfficientNet-B0 (frozen first 5 epochs)
├── Global Average Pooling
├── Dropout (0.3)
├── FC Layer (1280 → 512)
├── ReLU + Dropout (0.2)
└── FC Layer (512 → 35) [Output]
```

**Model Stats:**
- **Parameters**: ~4.0M trainable
- **Input**: 224x224 RGB
- **Output**: 35 classes (softmax)
- **Inference Speed**: ~8 FPS (RTX 3050), ~2 FPS (CPU)

---

## 🎓 Training Techniques (Anti-Overfitting)

### 1. **Data Augmentation**
```python
- RandomResizedCrop(224)
- RandomHorizontalFlip(p=0.5)
- RandomRotation(15°)
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
- RandomErasing(p=0.3)
- Normalization (ImageNet stats)
```

### 2. **Regularization**
- **Label Smoothing**: 0.1
- **Weight Decay**: 5e-4 (L2 regularization)
- **Dropout**: 0.3 + 0.2 (in classifier head)
- **MixUp**: α=0.2 (conservative mixing)

### 3. **Training Strategy**
- **Freeze-Unfreeze**: Train head first (5 epochs), then fine-tune
- **Early Stopping**: Patience=10 epochs
- **Gradient Clipping**: max_norm=1.0
- **Learning Rate Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)
- **Optimizer**: AdamW (better than Adam for regularization)

### 4. **Why These Work**

| Technique | Purpose | Effect |
|-----------|---------|--------|
| **Large Batch (64)** | Stable gradients | ↓ Noise, better generalization |
| **Low LR (0.0001)** | Conservative updates | ↓ Overfitting, stable training |
| **Weight Decay (5e-4)** | Penalize large weights | ↓ Model complexity |
| **MixUp** | Train on mixed samples | ↑ Robustness, smoother boundaries |
| **Freeze-Unfreeze** | Preserve pre-trained features | ↑ Transfer learning quality |

---

## 📁 Project Structure

```
indonesian-food-classification/
├── 📊 Dataset
│   └── Dataset ML/
│       ├── train.csv
│       ├── dev.csv
│       ├── test.csv
│       └── food-tfk-images/       # Image files
│
├── 🤖 Model & Training
│   ├── food_model.py              # EfficientNet-B0 architecture
│   ├── food_dataset.py            # Dataset loader & augmentation
│   ├── train_food_classification.py  # Training script
│   └── food_model_efficientnet.pth   # Trained weights (96MB)
│
├── 🔍 Inference
│   ├── food_inference.py          # Inference class & CLI
│   ├── webcam_app.py              # Desktop real-time app (OpenCV)
│   └── webcam_server.py           # Web-based real-time app (Flask)
│
├── 📈 Evaluation Results
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   ├── per_class_accuracy.png
│   ├── classification_report.txt
│   └── training_history_food.json
│
├── 📋 Checkpoints
│   └── checkpoints_food/
│       ├── best_model_efficientnet.pth
│       └── final_model_efficientnet.pth
│
├── 📚 Documentation
│   ├── README.md                  # This file
│   ├── ANTI_OVERFITTING_GUIDE.md  # Detailed overfitting solutions
│   └── requirements_webcam.txt    # Dependencies
│
└── 🛠️ Monitoring
    └── monitor_food_training.py   # Training progress monitor
```

---

## 📊 Results

### Overall Performance

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **Accuracy** | 86.3% | 83.7% | 82.9% |
| **Loss** | 0.234 | 0.279 | 0.291 |
| **Top-3 Acc** | - | - | 94.2% |
| **Top-5 Acc** | - | - | 97.1% |

### Per-Class Performance

**Best Classes (>90% accuracy):**
- Rendang: 95.3%
- Sate Ayam Madura: 93.7%
- Nasi Goreng Kampung: 91.2%

**Challenging Classes (<75% accuracy):**
- Bir Pletok: 68.4% (often confused with Kunyit Asam)
- Soerabi: 71.2% (similar texture to other desserts)

*See `per_class_accuracy.png` for full breakdown*

---

## 🔧 Customization

### Add New Food Class

1. **Update Dataset**:
   - Add images to `Dataset ML/food-tfk-images/`
   - Update CSV files with new class name

2. **Update Class List** in `food_inference.py`:
   ```python
   self.class_names = [
       'asinan-jakarta', 'ayam-betutu', ..., 'new-food'
   ]
   ```

3. **Retrain Model**:
   ```bash
   python train_food_classification.py
   ```

### Adjust Training Hyperparameters

Edit `train_food_classification.py`:

```python
config = {
    'batch_size': 32,          # Reduce if OOM
    'learning_rate': 5e-5,     # Even more conservative
    'weight_decay': 1e-3,      # Stronger regularization
    'num_epochs': 100,         # More epochs
    'mixup_alpha': 0.4,        # More aggressive mixing
}
```

### Use Different Model

Replace EfficientNet-B0 with other backbones:

```python
# In food_model.py
from torchvision.models import efficientnet_b3, resnet50, vit_b_16

# Then modify create_efficientnet_model() function
```

---

## 🐛 Troubleshooting

### Issue: Out of Memory (OOM)

**Solution:**
```python
# Reduce batch size
config['batch_size'] = 32  # or 16

# Or reduce image size
transforms.Resize((192, 192))  # instead of (224, 224)
```

### Issue: Low Accuracy / Overfitting

**Solution:**
1. Check `ANTI_OVERFITTING_GUIDE.md` for detailed solutions
2. Increase weight decay: `config['weight_decay'] = 1e-3`
3. Enable more augmentation in `food_dataset.py`
4. Collect more training data (recommended: 100+ images per class)

### Issue: Slow Inference

**Solution:**
```bash
# Use GPU (CUDA)
torch.cuda.is_available()  # Check if GPU detected

# Or convert to TorchScript (faster)
model_scripted = torch.jit.script(model)
model_scripted.save('model_scripted.pt')

# Or use ONNX Runtime (even faster)
torch.onnx.export(model, dummy_input, 'model.onnx')
```

### Issue: Webcam Not Working

**Solution:**
```bash
# Test camera index
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# Try different camera ID
python webcam_app.py  # Edit camera_id parameter if needed
```

---

## 📚 References

### Papers
- **EfficientNet**: Tan & Le (2019) - "EfficientNet: Rethinking Model Scaling for CNNs"
- **MixUp**: Zhang et al. (2017) - "mixup: Beyond Empirical Risk Minimization"
- **Label Smoothing**: Szegedy et al. (2016) - "Rethinking the Inception Architecture"

### Datasets
- Indonesian Food Dataset (custom collected)
- ImageNet (pre-training)

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Dataset**: Add more food classes, increase samples per class
2. **Model**: Experiment with Vision Transformers (ViT), Swin Transformer
3. **Features**: 
   - Nutrition information prediction (calorie, protein, etc.)
   - Portion size estimation
   - Multi-food detection (YOLO/Faster R-CNN)
4. **Deployment**: Docker container, mobile app (TensorFlow Lite)

### How to Contribute

```bash
# Fork repo
git checkout -b feature/your-feature
git commit -m "Add your feature"
git push origin feature/your-feature
# Create Pull Request
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- PyTorch team for excellent framework
- EfficientNet authors for efficient architecture
- Indonesian food community for cultural heritage

---

## 📞 Contact

**Author**: William  
**Email**: [your-email@example.com]  
**GitHub**: [@your-username](https://github.com/your-username)

---

## 🌟 Show Your Support

Give a ⭐️ if this project helped you!

---

## 📝 Changelog

### v1.0.0 (2025-11-27)
- ✅ Initial release
- ✅ EfficientNet-B0 model with 35 classes
- ✅ Real-time inference apps (desktop & web)
- ✅ Comprehensive evaluation and visualization
- ✅ Anti-overfitting training pipeline

---

**Made with ❤️ for Indonesian Food Recognition**
