"""
Advanced Food Classifier Model Architecture
Standalone version for web app inference
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

class AdvancedFoodClassifier(nn.Module):
    """
    EfficientNetB0 dengan custom classifier head
    - BatchNorm → Dense(512) → Dropout → Output
    - Optimized untuk Indonesian Food Classification
    """
    def __init__(self, num_classes=35, dropout_rate=0.5):
        super(AdvancedFoodClassifier, self).__init__()
        
        # Load pretrained EfficientNetB0
        self.base_model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Get number of features from base model
        in_features = self.base_model.classifier[1].in_features
        
        # Replace classifier with custom head
        self.base_model.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)
