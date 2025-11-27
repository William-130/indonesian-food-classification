"""
EfficientNet-B0 Model untuk Indonesian Food Classification
35 Classes
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

class EfficientNetB0Food(nn.Module):
    def __init__(self, num_classes=35, pretrained=True, dropout=0.3):
        """
        EfficientNet-B0 untuk klasifikasi makanan Indonesia
        
        Args:
            num_classes: Jumlah class (default 35)
            pretrained: Use pre-trained weights dari ImageNet
            dropout: Dropout rate
        """
        super(EfficientNetB0Food, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        if pretrained:
            self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Get number of features from backbone
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout/2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def create_efficientnet_model(num_classes=35, pretrained=True):
    """
    Factory function to create EfficientNet-B0 model
    """
    print(f"Creating EfficientNet-B0 model...")
    print(f"  Number of classes: {num_classes}")
    print(f"  Pre-trained: {pretrained}")
    
    model = EfficientNetB0Food(num_classes=num_classes, pretrained=pretrained)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model

if __name__ == "__main__":
    # Test model creation
    print("Testing EfficientNet-B0 Model...")
    print("="*60)
    
    model = create_efficientnet_model(num_classes=35, pretrained=True)
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    output = model(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output sample: {output[0, :5]}")
    
    # Test with CUDA if available
    if torch.cuda.is_available():
        print("\nTesting on CUDA...")
        model = model.cuda()
        x = x.cuda()
        output = model(x)
        print(f"  CUDA forward pass successful!")
        print(f"  Output shape: {output.shape}")
