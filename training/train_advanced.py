"""
Advanced Training Script untuk Indonesian Food Classification
Menggunakan EfficientNetB0 dengan 2-Stage Training Strategy

Strategy:
1. Stage 1: Feature Extraction (frozen base model, 10-15 epochs)
2. Stage 2: Fine-Tuning (unfreeze layers, 20-40 epochs dengan LR rendah)
3. Advanced Data Augmentation untuk makanan Indonesia
4. Anti-Overfitting techniques (Dropout, L2 Regularization, Early Stopping)

Author: William-130
Date: November 2025
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights

from food_dataset import IndonesianFoodDatasetCSV
from food_model import create_efficientnet_model

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid pywin32 issues
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm


# ==================== CONFIGURATION ====================
class TrainingConfig:
    """Konfigurasi training dengan 2-stage strategy"""
    
    # Paths
    DATA_DIR = '../Dataset ML'
    TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
    DEV_CSV = os.path.join(DATA_DIR, 'dev.csv')
    TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
    IMAGE_DIR = os.path.join(DATA_DIR, 'food-tfk-images')
    
    # Output paths
    CHECKPOINT_DIR = '../models/checkpoints/advanced_training'
    RESULTS_DIR = '../results'
    MODEL_SAVE_PATH = '../models/food_model_efficientnet_advanced.pth'
    
    # Model configuration
    NUM_CLASSES = 35
    IMAGE_SIZE = 224
    
    # Stage 1: Feature Extraction (Frozen base)
    STAGE1_EPOCHS = 15
    STAGE1_LR = 1e-3  # 0.001
    STAGE1_BATCH_SIZE = 32
    
    # Stage 2: Fine-Tuning (Unfreeze layers)
    STAGE2_EPOCHS = 40
    STAGE2_LR = 1e-5  # 0.00001
    STAGE2_BATCH_SIZE = 16  # Smaller batch for fine-tuning
    STAGE2_UNFREEZE_LAYERS = 30  # Unfreeze last 30 layers
    
    # Regularization
    DROPOUT_RATE = 0.5
    L2_WEIGHT_DECAY = 0.01
    
    # Training settings
    PATIENCE = 10  # Early stopping patience
    NUM_WORKERS = 4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== DATA AUGMENTATION ====================
def get_train_transforms():
    """
    Advanced data augmentation untuk makanan Indonesia.
    - Rotation: Piring bisa diputar
    - Zoom: Kamera bisa dekat/jauh
    - Shift: Objek tidak selalu di tengah
    - Horizontal Flip: Kiri/kanan sama saja
    - Color Jitter: Pencahayaan restoran beda-beda
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(40),  # 40 derajat rotasi
        transforms.RandomResizedCrop(
            TrainingConfig.IMAGE_SIZE,
            scale=(0.8, 1.0),  # Zoom range
            ratio=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.2, 0.2),  # Width/height shift
        ),
        transforms.ColorJitter(
            brightness=0.2,  # [0.8, 1.2]
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms():
    """Simple transforms untuk validation (no augmentation)"""
    return transforms.Compose([
        transforms.Resize((TrainingConfig.IMAGE_SIZE, TrainingConfig.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ==================== MODEL BUILDER ====================
class AdvancedFoodClassifier(nn.Module):
    """
    EfficientNetB0 dengan custom head untuk anti-overfitting.
    
    Architecture:
    - Base: EfficientNetB0 (pretrained on ImageNet)
    - Head: BatchNorm -> Dense(512) -> Dropout(0.5) -> Dense(35)
    """
    
    def __init__(self, num_classes=35, dropout_rate=0.5, freeze_base=True):
        super(AdvancedFoodClassifier, self).__init__()
        
        # Load EfficientNetB0 pretrained
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.base_model = models.efficientnet_b0(weights=weights)
        
        # Freeze/unfreeze base model
        self.set_base_trainable(not freeze_base)
        
        # Get number of features from base model
        num_features = self.base_model.classifier[1].in_features
        
        # Replace classifier with custom head
        self.base_model.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def set_base_trainable(self, trainable):
        """Set base model trainable/frozen"""
        for param in self.base_model.features.parameters():
            param.requires_grad = trainable
    
    def unfreeze_last_n_layers(self, n=30):
        """Unfreeze last N layers of base model"""
        # Freeze all first
        self.set_base_trainable(False)
        
        # Unfreeze last N layers
        layers = list(self.base_model.features.children())
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        return self.base_model(x)


# ==================== TRAINING FUNCTIONS ====================
def train_one_epoch(model, dataloader, criterion, optimizer, device, stage_name):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'{stage_name} Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc


def train_stage1(model, train_loader, val_loader, config):
    """
    Stage 1: Feature Extraction
    - Base model frozen
    - Train only classifier head
    - Higher learning rate (1e-3)
    """
    print("\n" + "="*70)
    print("STAGE 1: FEATURE EXTRACTION (Base Model Frozen)")
    print("="*70)
    
    # Freeze base model
    model.set_base_trainable(False)
    
    # Setup optimizer & criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.STAGE1_LR,
        weight_decay=config.L2_WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(config.STAGE1_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.STAGE1_EPOCHS}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, "Stage1"
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 
                      os.path.join(config.CHECKPOINT_DIR, 'stage1_best_model.pth'))
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\nStage 1 Complete! Best Val Acc: {best_val_acc:.2f}%")
    
    # Load best model
    model.load_state_dict(torch.load(
        os.path.join(config.CHECKPOINT_DIR, 'stage1_best_model.pth')
    ))
    
    return model, history


def train_stage2(model, train_loader, val_loader, config):
    """
    Stage 2: Fine-Tuning
    - Unfreeze last N layers
    - Very low learning rate (1e-5)
    - Smaller batch size
    """
    print("\n" + "="*70)
    print("STAGE 2: FINE-TUNING (Unfreeze Last Layers)")
    print("="*70)
    
    # Unfreeze last N layers
    model.unfreeze_last_n_layers(config.STAGE2_UNFREEZE_LAYERS)
    print(f"✓ Unfroze last {config.STAGE2_UNFREEZE_LAYERS} layers")
    
    # Setup optimizer & criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.STAGE2_LR,
        weight_decay=config.L2_WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(config.STAGE2_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.STAGE2_EPOCHS}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, "Stage2"
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 
                      os.path.join(config.CHECKPOINT_DIR, 'stage2_best_model.pth'))
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\nStage 2 Complete! Best Val Acc: {best_val_acc:.2f}%")
    
    # Load best model
    model.load_state_dict(torch.load(
        os.path.join(config.CHECKPOINT_DIR, 'stage2_best_model.pth')
    ))
    
    return model, history


# ==================== EVALUATION ====================
def evaluate_model(model, test_loader, class_names, config):
    """Comprehensive model evaluation"""
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(config.DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate accuracy
    correct = sum([1 for pred, label in zip(all_preds, all_labels) if pred == label])
    test_acc = 100 * correct / len(all_labels)
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    
    # Classification report (specify all labels to handle missing classes in test set)
    all_class_indices = list(range(len(class_names)))
    report = classification_report(all_labels, all_preds, labels=all_class_indices, target_names=class_names, zero_division=0)
    print("\nClassification Report:")
    print(report)
    
    # Save report
    os.makedirs(f"{config.RESULTS_DIR}/reports", exist_ok=True)
    with open(f"{config.RESULTS_DIR}/reports/classification_report_advanced.txt", 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs(f"{config.RESULTS_DIR}/plots", exist_ok=True)
    plt.savefig(f"{config.RESULTS_DIR}/plots/confusion_matrix_advanced.png", dpi=150)
    print(f"✓ Saved confusion matrix to {config.RESULTS_DIR}/plots/")
    
    return test_acc


def plot_training_history(stage1_history, stage2_history, config):
    """Plot training curves for both stages"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Stage 1
    axes[0, 0].plot(stage1_history['train_loss'], label='Train Loss')
    axes[0, 0].plot(stage1_history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Stage 1: Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(stage1_history['train_acc'], label='Train Acc')
    axes[0, 1].plot(stage1_history['val_acc'], label='Val Acc')
    axes[0, 1].set_title('Stage 1: Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Stage 2
    axes[1, 0].plot(stage2_history['train_loss'], label='Train Loss')
    axes[1, 0].plot(stage2_history['val_loss'], label='Val Loss')
    axes[1, 0].set_title('Stage 2: Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(stage2_history['train_acc'], label='Train Acc')
    axes[1, 1].plot(stage2_history['val_acc'], label='Val Acc')
    axes[1, 1].set_title('Stage 2: Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_DIR}/plots/training_curves_advanced.png", dpi=150)
    print(f"✓ Saved training curves to {config.RESULTS_DIR}/plots/")


# ==================== MAIN ====================
def main():
    """Main training pipeline with 2-stage strategy"""
    config = TrainingConfig()
    
    print("\n" + "="*70)
    print("🍽️  ADVANCED INDONESIAN FOOD CLASSIFICATION TRAINING")
    print("="*70)
    print(f"Device: {config.DEVICE}")
    print(f"Strategy: 2-Stage Training (Feature Extraction + Fine-Tuning)")
    print(f"Model: EfficientNetB0 (ImageNet pretrained)")
    print(f"Classes: {config.NUM_CLASSES}")
    print("="*70)
    
    # Create directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Load data
    print("\nLoading datasets...")
    train_df = pd.read_csv(config.TRAIN_CSV)
    dev_df = pd.read_csv(config.DEV_CSV)
    test_df = pd.read_csv(config.TEST_CSV)
    
    class_names = train_df.columns[3:].tolist()
    print(f"✓ Loaded {len(train_df)} train, {len(dev_df)} dev, {len(test_df)} test samples")
    
    # Create datasets with augmentation
    train_dataset = IndonesianFoodDatasetCSV(config.TRAIN_CSV, config.IMAGE_DIR, transform=get_train_transforms())
    val_dataset = IndonesianFoodDatasetCSV(config.DEV_CSV, config.IMAGE_DIR, transform=get_val_transforms())
    test_dataset = IndonesianFoodDatasetCSV(config.TEST_CSV, config.IMAGE_DIR, transform=get_val_transforms())
    
    # Stage 1 dataloaders
    train_loader_stage1 = DataLoader(
        train_dataset, batch_size=config.STAGE1_BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.STAGE1_BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True
    )
    
    # Stage 2 dataloaders (smaller batch size)
    train_loader_stage2 = DataLoader(
        train_dataset, batch_size=config.STAGE2_BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True
    )
    
    # Test dataloader
    test_loader = DataLoader(
        test_dataset, batch_size=32,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True
    )
    
    # Create model
    print("\nInitializing model...")
    model = AdvancedFoodClassifier(
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
        freeze_base=True
    ).to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # ==================== STAGE 1: FEATURE EXTRACTION ====================
    model, stage1_history = train_stage1(model, train_loader_stage1, val_loader, config)
    
    # ==================== STAGE 2: FINE-TUNING ====================
    model, stage2_history = train_stage2(model, train_loader_stage2, val_loader, config)
    
    # ==================== EVALUATION ====================
    test_acc = evaluate_model(model, test_loader, class_names, config)
    
    # ==================== SAVE FINAL MODEL ====================
    print("\nSaving final model...")
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"✓ Model saved to {config.MODEL_SAVE_PATH}")
    
    # Save training history
    history = {
        'stage1': stage1_history,
        'stage2': stage2_history,
        'test_accuracy': test_acc,
        'config': {
            'stage1_epochs': config.STAGE1_EPOCHS,
            'stage1_lr': config.STAGE1_LR,
            'stage2_epochs': config.STAGE2_EPOCHS,
            'stage2_lr': config.STAGE2_LR,
            'dropout': config.DROPOUT_RATE,
            'l2_decay': config.L2_WEIGHT_DECAY,
        }
    }
    
    with open(f"{config.RESULTS_DIR}/reports/training_history_advanced.json", 'w') as f:
        json.dump(history, f, indent=4)
    print(f"✓ Training history saved")
    
    # Plot training curves
    plot_training_history(stage1_history, stage2_history, config)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Model: {config.MODEL_SAVE_PATH}")
    print(f"Results: {config.RESULTS_DIR}/")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
