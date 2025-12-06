"""
Training Script dengan EfficientNet-B0
Indonesian Food Classification - 35 Classes
Includes: Training, Validation, Confusion Matrix, Visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    precision_recall_fscore_support,
    top_k_accuracy_score
)
import json
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from training.food_model import create_efficientnet_model
from training.food_dataset import create_dataloaders

class FoodTrainer:
    def __init__(self, model, train_loader, dev_loader, test_loader, class_names, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.config = config
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        if config['scheduler'] == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=5,
                factor=0.5
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config['num_epochs']
            )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'dev_loss': [],
            'dev_acc': [],
            'learning_rate': []
        }
        
        self.best_dev_acc = 0.0
        
        # Early stopping and training tricks
        self.early_stop_patience = config.get('early_stop_patience', 8)
        self.epochs_no_improve = 0
        self.mixup = config.get('mixup', False)
        self.mixup_alpha = config.get('mixup_alpha', 0.4)
        self.grad_clip = config.get('grad_clip', 1.0)
        # Option to unfreeze backbone at a given epoch (0 = never/unfrozen)
        self.unfreeze_at_epoch = config.get('unfreeze_at_epoch', 0)

    def mixup_data(self, x, y, alpha=0.4):
        """Returns mixed inputs, pairs of targets, and lambda"""
        if alpha <= 0:
            return x, y, y, 1.0
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Optional MixUp
            self.optimizer.zero_grad()
            if self.mixup:
                images, labels_a, labels_b, lam = self.mixup_data(images, labels, self.mixup_alpha)
                outputs = self.model(images)
                loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
            else:
                # Forward
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward with gradient clipping
            loss.backward()
            if self.grad_clip is not None and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, dataloader, desc='Validation'):
        """Validate on dev or test set"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=desc)
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)
    
    def train(self, num_epochs):
        """Main training loop"""
        print(f"\nStarting training for {num_epochs} epochs...")
        print("="*60)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-"*60)
            # Unfreeze backbone if configured at this epoch
            if self.unfreeze_at_epoch and (epoch + 1) == self.unfreeze_at_epoch:
                print(f"\nUnfreezing backbone at epoch {epoch+1} and rebuilding optimizer...")
                # Unfreeze all parameters and recreate optimizer to include them
                for p in self.model.parameters():
                    p.requires_grad = True
                self.optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=self.config['learning_rate'],
                    weight_decay=self.config['weight_decay']
                )
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate on dev set
            dev_loss, dev_acc, _, _ = self.validate(self.dev_loader, desc='Validation')
            
            # Update learning rate
            if self.config['scheduler'] == 'plateau':
                self.scheduler.step(dev_acc)
            else:
                self.scheduler.step()
            
            # Get current LR
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['dev_loss'].append(dev_loss)
            self.history['dev_acc'].append(dev_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Print summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Dev Loss: {dev_loss:.4f} | Dev Acc: {dev_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if dev_acc > self.best_dev_acc:
                self.best_dev_acc = dev_acc
                self.save_checkpoint('best_model_efficientnet.pth', epoch, dev_acc)
                print(f"  ✓ Best model saved! (Dev Acc: {dev_acc:.2f}%)")
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            # Early stopping
            if self.epochs_no_improve >= self.early_stop_patience:
                print(f"\nEarly stopping triggered. No improvement for {self.epochs_no_improve} epochs.")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, dev_acc)
        
        # Save final model
        self.save_checkpoint('final_model_efficientnet.pth', num_epochs, dev_acc)
        
        print("\n" + "="*60)
        print("Training completed!")
        print(f"Best validation accuracy: {self.best_dev_acc:.2f}%")
    
    def save_checkpoint(self, filename, epoch, dev_acc):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(os.path.dirname(__file__), '../models/checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'dev_acc': dev_acc,
            'config': self.config,
            'history': self.history,
            'class_names': self.class_names
        }
        
        filepath = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        
        # Also save standalone model weights
        if 'best' in filename:
            model_path = os.path.join(os.path.dirname(__file__), '../models/food_model_efficientnet.pth')
            torch.save(self.model.state_dict(), model_path)
    
    def evaluate_and_visualize(self):
        """Comprehensive evaluation dengan confusion matrix dan visualizations"""
        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION")
        print("="*60)
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_acc, test_preds, test_labels = self.validate(
            self.test_loader, 
            desc='Testing'
        )
        
        print(f"\nTest Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        
        # Classification report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        
        # Get unique classes in predictions and labels
        unique_classes = sorted(list(set(test_labels) | set(test_preds)))
        print(f"Note: {len(unique_classes)} out of {len(self.class_names)} classes found in test set")
        
        report = classification_report(
            test_labels, 
            test_preds, 
            labels=list(range(len(self.class_names))),  # Specify all possible labels
            target_names=self.class_names,
            digits=4,
            zero_division=0  # Handle classes with no predictions
        )
        print(report)
        
        # Save classification report
        report_dir = os.path.join(os.path.dirname(__file__), '../results/reports')
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, 'classification_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Classification Report - Indonesian Food Classification\n")
            f.write("="*60 + "\n")
            f.write(f"Test Accuracy: {test_acc:.2f}%\n")
            f.write("="*60 + "\n\n")
            f.write(report)
        
        # Confusion Matrix
        print("\nGenerating confusion matrix...")
        self.plot_confusion_matrix(test_labels, test_preds)
        
        # Training curves
        print("Plotting training curves...")
        self.plot_training_curves()
        
        # Per-class accuracy
        print("\nPlotting per-class accuracy...")
        self.plot_per_class_accuracy(test_labels, test_preds)
        
        # Top-k accuracy
        self.compute_topk_accuracy()
        
        print("\n✓ All visualizations saved!")
        print("  - confusion_matrix.png")
        print("  - training_curves.png")
        print("  - per_class_accuracy.png")
        print("  - classification_report.txt")
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        # Include all possible labels
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(self.class_names))))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure dengan 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))
        
        # Plot 1: Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'},
                   ax=axes[0])
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, pad=20)
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        axes[0].tick_params(axis='both', labelsize=8)
        plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')
        plt.setp(axes[0].get_yticklabels(), rotation=0)
        
        # Plot 2: Normalized (percentages)
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Proportion'},
                   ax=axes[1])
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, pad=20)
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        axes[1].tick_params(axis='both', labelsize=8)
        plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
        plt.setp(axes[1].get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        plots_dir = os.path.join(os.path.dirname(__file__), '../results/plots')
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix saved to: {plot_path}")
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['dev_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.history['dev_acc'], 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 0].plot(epochs, self.history['learning_rate'], 'g-', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics
        axes[1, 1].axis('off')
        stats_text = f"""
        Training Summary
        {'='*40}
        
        Total Epochs: {len(epochs)}
        
        Best Validation Accuracy: {max(self.history['dev_acc']):.2f}%
        Final Validation Accuracy: {self.history['dev_acc'][-1]:.2f}%
        
        Best Train Accuracy: {max(self.history['train_acc']):.2f}%
        Final Train Accuracy: {self.history['train_acc'][-1]:.2f}%
        
        Final Train Loss: {self.history['train_loss'][-1]:.4f}
        Final Val Loss: {self.history['dev_loss'][-1]:.4f}
        
        Model: EfficientNet-B0
        Classes: {len(self.class_names)}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                       verticalalignment='center')
        
        plt.tight_layout()
        plots_dir = os.path.join(os.path.dirname(__file__), '../results/plots')
        plot_path = os.path.join(plots_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training curves saved to: {plot_path}")
    
    def plot_per_class_accuracy(self, y_true, y_pred):
        """Plot per-class accuracy"""
        # Calculate per-class accuracy
        per_class_acc = []
        for i in range(len(self.class_names)):
            mask = y_true == i
            if mask.sum() > 0:
                acc = (y_pred[mask] == i).sum() / mask.sum() * 100
                per_class_acc.append(acc)
            else:
                per_class_acc.append(0)
        
        # Sort by accuracy
        sorted_indices = np.argsort(per_class_acc)
        sorted_names = [self.class_names[i] for i in sorted_indices]
        sorted_accs = [per_class_acc[i] for i in sorted_indices]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        colors = ['red' if acc < 70 else 'orange' if acc < 85 else 'green' 
                 for acc in sorted_accs]
        bars = ax.barh(range(len(sorted_names)), sorted_accs, color=colors, alpha=0.7)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names, fontsize=9)
        ax.set_xlabel('Accuracy (%)', fontsize=12)
        ax.set_title('Per-Class Accuracy (Test Set)', fontsize=14, fontweight='bold', pad=20)
        ax.axvline(x=70, color='red', linestyle='--', alpha=0.5, label='70%')
        ax.axvline(x=85, color='orange', linestyle='--', alpha=0.5, label='85%')
        ax.grid(axis='x', alpha=0.3)
        ax.legend()
        
        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars, sorted_accs)):
            ax.text(acc + 1, i, f'{acc:.1f}%', va='center', fontsize=8)
        
        plt.tight_layout()
        plots_dir = os.path.join(os.path.dirname(__file__), '../results/plots')
        plot_path = os.path.join(plots_dir, 'per_class_accuracy.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Per-class accuracy saved to: {plot_path}")
    
    def compute_topk_accuracy(self):
        """Compute Top-1, Top-3, Top-5 accuracy"""
        self.model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Computing Top-K'):
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())
        
        all_probs = np.vstack(all_probs)
        all_labels = np.concatenate(all_labels)
        
        # Compute top-k accuracy
        top1_acc = accuracy_score(all_labels, np.argmax(all_probs, axis=1)) * 100
        
        if len(self.class_names) >= 3:
            top3_acc = top_k_accuracy_score(all_labels, all_probs, k=3) * 100
        else:
            top3_acc = top1_acc
            
        if len(self.class_names) >= 5:
            top5_acc = top_k_accuracy_score(all_labels, all_probs, k=5) * 100
        else:
            top5_acc = top1_acc
        
        print(f"\nTop-K Accuracy:")
        print(f"  Top-1 Accuracy: {top1_acc:.2f}%")
        print(f"  Top-3 Accuracy: {top3_acc:.2f}%")
        print(f"  Top-5 Accuracy: {top5_acc:.2f}%")
        
        # Save to file
        with open('topk_accuracy.txt', 'w') as f:
            f.write("Top-K Accuracy Results\n")
            f.write("="*40 + "\n")
            f.write(f"Top-1 Accuracy: {top1_acc:.2f}%\n")
            f.write(f"Top-3 Accuracy: {top3_acc:.2f}%\n")
            f.write(f"Top-5 Accuracy: {top5_acc:.2f}%\n")

def main():
    # Configuration - Anti-Overfitting Setup
    config = {
        'model_name': 'EfficientNet-B0',
        'num_epochs': 50,
        'batch_size': 64,              # ✓ Naikkan batch size (lebih stable gradient)
        'learning_rate': 0.0001,       # ✓ Turunkan LR 10x (lebih konservatif)
        'weight_decay': 5e-4,          # ✓ Naikkan weight decay (L2 regularization)
        'scheduler': 'plateau',        # ReduceLROnPlateau (adaptive)
        'data_dir': 'Dataset ML',
        'num_workers': 4,              # ✓ Parallel data loading
        
        # Anti-Overfitting Settings
        'early_stop_patience': 10,     # Stop jika 10 epochs tidak improve
        'mixup': True,                 # MixUp augmentation
        'mixup_alpha': 0.2,           # Conservative mixup (0.2 lebih aman dari 0.4)
        'grad_clip': 1.0,             # Gradient clipping
        'freeze_backbone_epochs': 5,   # Freeze backbone 5 epochs pertama
        'unfreeze_at_epoch': 6        # Unfreeze di epoch 6
    }
    
    print("="*60)
    print("INDONESIAN FOOD CLASSIFICATION")
    print("EfficientNet-B0 - 35 Classes")
    print("="*60)
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_loader, dev_loader, test_loader, class_names = create_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    print(f"\nDataset statistics:")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(dev_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Number of classes: {len(class_names)}")
    
    # Create model
    print("\nCreating model...")
    model = create_efficientnet_model(num_classes=len(class_names), pretrained=True)
    
    # Freeze backbone initially (if configured)
    if config.get('freeze_backbone_epochs', 0) > 0:
        print(f"\n⚠ Freezing backbone for first {config['freeze_backbone_epochs']} epochs...")
        print("   This prevents overfitting by training only classifier head first.")
        for name, param in model.named_parameters():
            if 'classifier' not in name:  # Freeze everything except classifier
                param.requires_grad = False
        
        # Count trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"   Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    # Create trainer
    trainer = FoodTrainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        config=config
    )
    
    # Train
    trainer.train(config['num_epochs'])
    
    # Save training history
    with open('training_history_food.json', 'w') as f:
        json.dump(trainer.history, f, indent=2)
    
    # Evaluate and create visualizations
    trainer.evaluate_and_visualize()
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - food_model_efficientnet.pth (model weights)")
    print("  - confusion_matrix.png")
    print("  - training_curves.png")
    print("  - per_class_accuracy.png")
    print("  - classification_report.txt")
    print("  - topk_accuracy.txt")
    print("  - training_history_food.json")

if __name__ == "__main__":
    main()
