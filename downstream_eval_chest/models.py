"""
Model architecture for CheXpert multi-label classification.
Uses standard torchvision DenseNet-121 with ImageNet weights, modified for grayscale input.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights
from typing import Optional


class DenseNet121Classifier(nn.Module):
    """
    DenseNet-121 based multi-label classifier for 14 CheXpert classes.
    Uses standard torchvision model with ImageNet weights, modified for 1-channel grayscale input.
    
    Args:
        num_classes: Number of output classes (default: 14 for CheXpert)
        pretrained: Whether to use ImageNet pretrained weights (default: True)
        from_scratch: If True, initialize from scratch (overrides pretrained)
        dropout_rate: Dropout rate for classifier head (default: 0.5)
    """
    
    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        from_scratch: bool = False,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load DenseNet-121 with ImageNet weights (same as validation approach)
        if from_scratch:
            self.backbone = models.densenet121(weights=None)
        else:
            self.backbone = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        
        # Modify first conv layer to accept 1-channel grayscale input instead of 3-channel RGB
        # This matches the validation approach where images are converted to grayscale
        # Initialize the new conv layer by averaging the RGB channels of the pretrained weights
        old_conv0 = self.backbone.features.conv0
        new_conv0 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Initialize new layer: average the 3 RGB channels from pretrained weights
        if not from_scratch and old_conv0.weight.shape[1] == 3:
            # Average across RGB channels: [64, 3, 7, 7] -> [64, 1, 7, 7]
            with torch.no_grad():
                new_conv0.weight.data = old_conv0.weight.data.mean(dim=1, keepdim=True)
        else:
            # From scratch: use default initialization
            nn.init.kaiming_normal_(new_conv0.weight, mode='fan_out', nonlinearity='relu')
        
        self.backbone.features.conv0 = new_conv0
        
        # Get number of features from classifier
        num_features = self.backbone.classifier.in_features
        
        # Replace classifier with dropout + linear for regularization
        # Dropout helps prevent overfitting, especially when using synthetic data
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 1, H, W] (grayscale, 1 channel)
        
        Returns:
            Logits [B, num_classes]
        """
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification head.
        
        Args:
            x: Input images [B, 1, H, W] (grayscale, 1 channel)
        
        Returns:
            Features [B, num_features]
        """
        features = self.backbone.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out


def load_checkpoint(
    model: DenseNet121Classifier,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> DenseNet121Classifier:
    """
    Load model from checkpoint.
    
    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load on
    
    Returns:
        Model with loaded weights
    """
    # weights_only=False needed for PyTorch 2.6+ when checkpoint contains numpy scalars
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    return model

