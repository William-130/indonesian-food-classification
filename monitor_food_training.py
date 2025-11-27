"""
Monitor training progress untuk food classification
"""

import os
import json
import time
from pathlib import Path

def monitor_training():
    print("="*60)
    print("FOOD CLASSIFICATION TRAINING MONITOR")
    print("="*60)
    
    # Check if training history exists
    if os.path.exists('training_history_food.json'):
        print("\n📊 Loading training history...")
        
        with open('training_history_food.json', 'r') as f:
            history = json.load(f)
        
        if history['train_loss']:
            epochs_completed = len(history['train_loss'])
            
            print(f"\n✓ Epochs completed: {epochs_completed}/50")
            print("-"*60)
            
            # Latest epoch stats
            print(f"Latest epoch:")
            print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
            print(f"  Train Acc: {history['train_acc'][-1]:.2f}%")
            print(f"  Dev Loss: {history['dev_loss'][-1]:.4f}")
            print(f"  Dev Acc: {history['dev_acc'][-1]:.2f}%")
            print(f"  Learning Rate: {history['learning_rate'][-1]:.6f}")
            
            # Best performance
            best_dev_acc = max(history['dev_acc'])
            best_epoch = history['dev_acc'].index(best_dev_acc) + 1
            
            print(f"\nBest performance:")
            print(f"  Epoch: {best_epoch}")
            print(f"  Dev Accuracy: {best_dev_acc:.2f}%")
            
            # Progress estimate
            if epochs_completed > 1:
                import numpy as np
                recent_accs = history['dev_acc'][-5:]
                trend = "📈 Improving" if recent_accs[-1] > recent_accs[0] else "📉 Declining"
                print(f"\nRecent trend: {trend}")
        else:
            print("\n⏳ Training just started...")
    else:
        print("\n⏳ Waiting for training to start...")
        print("Training file will be created after first epoch completes.")
    
    # Check generated files
    print("\n" + "="*60)
    print("GENERATED FILES")
    print("="*60)
    
    files_to_check = [
        ('food_model_efficientnet.pth', 'Model weights'),
        ('checkpoints_food/best_model_efficientnet.pth', 'Best checkpoint'),
        ('confusion_matrix.png', 'Confusion matrix visualization'),
        ('training_curves.png', 'Training curves'),
        ('per_class_accuracy.png', 'Per-class accuracy'),
        ('classification_report.txt', 'Classification report'),
        ('topk_accuracy.txt', 'Top-K accuracy'),
    ]
    
    for filepath, desc in files_to_check:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            print(f"✓ {desc}")
            print(f"  {filepath} ({size:.2f} MB)")
        else:
            print(f"⏳ {desc}")
            print(f"  {filepath} (pending)")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    monitor_training()
