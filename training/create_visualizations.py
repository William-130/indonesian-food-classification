"""
Create beautiful visualizations for training results
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_performance_summary():
    """Create comprehensive performance summary visualization"""
    
    # Data dari classification report
    classes = [
        'asinan-jakarta', 'ayam-betutu', 'ayam-bumbu-rujak', 'ayam-goreng-lengkuas',
        'bika-ambon', 'bir-pletok', 'bubur-manado', 'cendol', 'es-dawet', 'gado-gado',
        'gudeg', 'gulai-ikan-mas', 'keladi', 'kerak-telor', 'klappertart', 'kolak',
        'kue-lumpur', 'laksa-bogor', 'lumpia-semarang', 'mie-aceh', 'nagasari',
        'papeda', 'pempek-palembang', 'rawon-surabaya', 'rendang', 'rujak-cingur',
        'sate-ayam-madura', 'sate-lilit', 'sate-maranggi', 'soerabi',
        'soto-ayam-lamongan', 'soto-banjar', 'tahu-telur'
    ]
    
    f1_scores = [
        1.00, 0.95, 1.00, 1.00, 1.00, 1.00, 0.95, 1.00, 1.00, 1.00,
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
        1.00, 1.00, 1.00
    ]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. F1-Score Bar Chart
    ax1 = plt.subplot(2, 2, 1)
    colors = ['#ff6b6b' if score < 1.0 else '#51cf66' for score in f1_scores]
    bars = ax1.barh(classes, f1_scores, color=colors, alpha=0.8)
    ax1.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
    ax1.set_title('Per-Class F1-Score (99.70% Test Accuracy)', fontsize=14, fontweight='bold')
    ax1.set_xlim([0.85, 1.02])
    ax1.axvline(x=1.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect Score')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        if score < 1.0:
            ax1.text(score + 0.005, i, f'{score:.2f}', va='center', fontweight='bold', color='red')
    
    # 2. Model Comparison
    ax2 = plt.subplot(2, 2, 2)
    models = ['Old Model\n(Single-Stage)', 'New Model\n(2-Stage Training)']
    accuracies = [82.0, 99.70]
    colors_model = ['#ff6b6b', '#51cf66']
    bars = ax2.bar(models, accuracies, color=colors_model, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add improvement annotation
    ax2.annotate('', xy=(1, 99.70), xytext=(0, 82.0),
                arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
    ax2.text(0.5, 90, '+17.7%\nImprovement!', ha='center', fontsize=12, 
            fontweight='bold', color='blue',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # 3. Class Distribution
    ax3 = plt.subplot(2, 2, 3)
    support = [
        10, 11, 13, 13, 7, 8, 10, 14, 11, 6, 4, 4, 9, 6, 8, 6, 4,
        10, 11, 12, 22, 10, 4, 12, 13, 14, 6, 15, 11, 10, 12, 9, 14
    ]
    
    ax3.hist(support, bins=15, color='#4ecdc4', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Number of Test Samples', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Classes', fontsize=12, fontweight='bold')
    ax3.set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')
    ax3.axvline(x=np.mean(support), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(support):.1f}')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Key Metrics Summary
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    summary_text = f"""
    ╔═══════════════════════════════════════════════════════╗
    ║         🏆 ADVANCED TRAINING RESULTS 🏆               ║
    ╠═══════════════════════════════════════════════════════╣
    ║                                                       ║
    ║  📊 Overall Performance:                              ║
    ║     • Test Accuracy:        99.70%                    ║
    ║     • Correct Predictions:  328 / 329                 ║
    ║     • Error Rate:           0.30%                     ║
    ║                                                       ║
    ║  🎯 Class-Level Metrics:                              ║
    ║     • Perfect Classes:      31 / 33 (93.9%)           ║
    ║     • Average F1-Score:     0.998                     ║
    ║     • Macro Avg Precision:  0.94                      ║
    ║     • Weighted Avg:         1.00                      ║
    ║                                                       ║
    ║  📈 Improvement:                                      ║
    ║     • Old Model:            82.0%                     ║
    ║     • New Model:            99.70%                    ║
    ║     • Gain:                 +17.7%                    ║
    ║                                                       ║
    ║  ⚙️ Training Strategy:                                ║
    ║     • Stage 1:              Feature Extraction        ║
    ║     • Stage 2:              Fine-Tuning               ║
    ║     • Architecture:         EfficientNetB0           ║
    ║     • Regularization:       Dropout + L2 + Aug       ║
    ║                                                       ║
    ║  🔥 Only 1 Misclassification:                         ║
    ║     • ayam-betutu → bubur-manado (1 error)           ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('🍽️ Indonesian Food Classification - Advanced Training Results', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    os.makedirs('../results/plots', exist_ok=True)
    plt.savefig('../results/plots/performance_summary.png', dpi=150, bbox_inches='tight')
    print("✓ Performance summary saved to: ../results/plots/performance_summary.png")
    plt.close()


def create_training_comparison():
    """Create training strategy comparison visualization"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Old Training Strategy
    ax1 = axes[0]
    stages_old = ['ImageNet\nPretrained', 'Train All\nLayers']
    accuracy_old = [0, 82.0]
    ax1.plot(stages_old, accuracy_old, 'o-', linewidth=3, markersize=12, 
            color='#ff6b6b', label='Single-Stage Training')
    ax1.fill_between(range(len(stages_old)), accuracy_old, alpha=0.3, color='#ff6b6b')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_title('❌ Old Training Strategy', fontsize=16, fontweight='bold', color='#ff6b6b')
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=12)
    
    # Add final accuracy annotation
    ax1.annotate(f'{accuracy_old[-1]:.1f}%', 
                xy=(1, accuracy_old[-1]), 
                xytext=(1.2, accuracy_old[-1]),
                fontsize=16, fontweight='bold', color='#ff6b6b',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#ff6b6b', linewidth=2))
    
    # New Training Strategy
    ax2 = axes[1]
    stages_new = ['ImageNet\nPretrained', 'Stage 1:\nFeature\nExtraction', 'Stage 2:\nFine-Tuning']
    accuracy_new = [0, 100.0, 99.70]  # Val 100%, Test 99.70%
    ax2.plot(stages_new, accuracy_new, 'o-', linewidth=3, markersize=12, 
            color='#51cf66', label='2-Stage Training')
    ax2.fill_between(range(len(stages_new)), accuracy_new, alpha=0.3, color='#51cf66')
    ax2.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax2.set_title('✅ New Advanced Training Strategy', fontsize=16, fontweight='bold', color='#51cf66')
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=12)
    
    # Add annotations
    ax2.annotate('Val: 100%\nTest: 99.7%', 
                xy=(2, accuracy_new[-1]), 
                xytext=(2.3, accuracy_new[-1]),
                fontsize=14, fontweight='bold', color='#51cf66',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#51cf66', linewidth=2))
    
    plt.suptitle('🔄 Training Strategy Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig('../results/plots/training_strategy_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Training strategy comparison saved to: ../results/plots/training_strategy_comparison.png")
    plt.close()


def create_error_analysis():
    """Create detailed error analysis visualization"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Only 1 error: ayam-betutu predicted as bubur-manado
    categories = [
        'Perfect\nPredictions\n(31 classes)',
        'Near-Perfect\n(ayam-betutu)\n90.91%',
        'Near-Perfect\n(bubur-manado)\n100%*'
    ]
    
    values = [31, 1, 1]
    colors = ['#51cf66', '#ffd43b', '#ff922b']
    explode = (0.05, 0.1, 0.1)
    
    wedges, texts, autotexts = ax.pie(values, labels=categories, autopct='%1.1f%%',
                                       colors=colors, explode=explode, startangle=90,
                                       textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    ax.set_title('Class Performance Distribution\n(Only 1 Error in 329 Test Samples)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add legend with details
    legend_text = [
        '✅ 31 classes: 100% accuracy',
        '⚠️ ayam-betutu: 10/11 correct (1 → bubur-manado)',
        '✅ bubur-manado: 10/10 correct (received 1 from ayam-betutu)'
    ]
    ax.legend(legend_text, loc='center', bbox_to_anchor=(0.5, -0.1), 
             fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig('../results/plots/error_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Error analysis saved to: ../results/plots/error_analysis.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("📊 CREATING ADVANCED VISUALIZATIONS")
    print("="*70 + "\n")
    
    print("Creating performance summary...")
    create_performance_summary()
    
    print("\nCreating training strategy comparison...")
    create_training_comparison()
    
    print("\nCreating error analysis...")
    create_error_analysis()
    
    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS CREATED!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. performance_summary.png")
    print("  2. training_strategy_comparison.png")
    print("  3. error_analysis.png")
    print("  4. confusion_matrix_advanced.png (already exists)")
    print("\nLocation: results/plots/")
    print("="*70 + "\n")
