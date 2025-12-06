"""
Quick Evaluation Script untuk model yang sudah trained
Load best model dan generate classification report + confusion matrix
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import dari project
sys.path.append(os.path.dirname(__file__))
from food_dataset import IndonesianFoodDatasetCSV
from train_advanced import AdvancedFoodClassifier, TrainingConfig

def evaluate():
    config = TrainingConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("🔍 MODEL EVALUATION")
    print("="*70)
    print(f"Device: {device}")
    
    # Load test dataset
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = IndonesianFoodDatasetCSV(config.TEST_CSV, config.IMAGE_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    class_names = test_dataset.class_names
    print(f"Classes: {len(class_names)}")
    print(f"Test samples: {len(test_dataset)}\n")
    
    # Load best model
    print("Loading best model from Stage 2...")
    model = AdvancedFoodClassifier(num_classes=len(class_names))
    
    # Try loading stage2 best model first
    checkpoint_path = "../models/checkpoints/advanced_training/stage2_best_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Stage 2 model not found. Trying stage1 best model...")
        checkpoint_path = "../models/checkpoints/advanced_training/stage1_best_model.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"Stage 1 model not found. Trying final model...")
        checkpoint_path = "../models/food_model_efficientnet_advanced.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No trained model found!")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        val_acc = checkpoint.get('best_val_acc', 0)
    else:
        # Direct state_dict
        model.load_state_dict(checkpoint)
        epoch = 'unknown'
        val_acc = 'unknown'
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from: {checkpoint_path}")
    if epoch != 'unknown':
        print(f"✓ Model trained to epoch: {epoch}")
    if val_acc != 'unknown':
        print(f"✓ Best val accuracy: {val_acc:.2f}%")
    print()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate accuracy
    correct = sum([1 for pred, label in zip(all_preds, all_labels) if pred == label])
    test_acc = 100 * correct / len(all_labels)
    
    print(f"\n{'='*70}")
    print(f"🎯 TEST ACCURACY: {test_acc:.2f}%")
    print(f"{'='*70}\n")
    
    # Classification report
    print("Generating classification report...")
    all_class_indices = list(range(len(class_names)))
    report = classification_report(
        all_labels, all_preds, 
        labels=all_class_indices, 
        target_names=class_names, 
        zero_division=0
    )
    print("\nClassification Report:")
    print(report)
    
    # Save report
    os.makedirs("../results/reports", exist_ok=True)
    report_path = "../results/reports/classification_report_advanced.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Advanced Training - EfficientNetB0\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Total Samples: {len(all_labels)}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Incorrect: {len(all_labels) - correct}\n\n")
        f.write(report)
    
    print(f"✓ Report saved to: {report_path}")
    
    # Confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names, 
                fmt='d', cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - Test Accuracy: {test_acc:.2f}%', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    os.makedirs("../results/plots", exist_ok=True)
    cm_path = "../results/plots/confusion_matrix_advanced.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to: {cm_path}")
    
    # Per-class accuracy
    print("\n" + "="*70)
    print("PER-CLASS ACCURACY")
    print("="*70)
    
    class_correct = {}
    class_total = {}
    
    for label, pred in zip(all_labels, all_preds):
        class_name = class_names[label]
        if class_name not in class_total:
            class_total[class_name] = 0
            class_correct[class_name] = 0
        class_total[class_name] += 1
        if label == pred:
            class_correct[class_name] += 1
    
    # Sort by accuracy
    class_acc = {name: (class_correct[name] / class_total[name] * 100) 
                 for name in class_total.keys()}
    sorted_classes = sorted(class_acc.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Best Classes:")
    for i, (name, acc) in enumerate(sorted_classes[:10], 1):
        total = class_total[name]
        correct = class_correct[name]
        print(f"{i:2}. {name:30} | Acc: {acc:6.2f}% ({correct}/{total})")
    
    print("\nTop 10 Worst Classes:")
    for i, (name, acc) in enumerate(sorted_classes[-10:][::-1], 1):
        total = class_total[name]
        correct = class_correct[name]
        print(f"{i:2}. {name:30} | Acc: {acc:6.2f}% ({correct}/{total})")
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    evaluate()
