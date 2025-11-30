"""
This module defines the neural network architecture for mask classification.
We use transfer learning with a pretrained backbone for better performance.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class FaceMaskClassifier(nn.Module):
    """
    Face Mask Classification Model using Transfer Learning
    
    Design Decision: We use ResNet-50 as the backbone because:
    1. It's well-proven for image classification tasks
    2. Pretrained on ImageNet, which includes faces and objects
    3. Good balance between accuracy and computational cost
    4. ResNet's skip connections help with gradient flow
    
    Alternative considered: EfficientNet (more parameter-efficient but slower)
    """
    
    def __init__(self, num_classes=3, pretrained=True, dropout_rate=0.5):
        """
        Args:
            num_classes: Number of output classes (3 for our task)
            pretrained: Whether to use ImageNet pretrained weights
            dropout_rate: Dropout probability to prevent overfitting
        """
        super(FaceMaskClassifier, self).__init__()
        
        # Load pretrained ResNet-50
        # Design Decision: We use pretrained weights as our dataset is relatively small
        # (~900 faces). Transfer learning from ImageNet helps avoid overfitting.
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Get number of features from last layer
        num_features = self.backbone.fc.in_features
        
        # Replace the final fully connected layer
        # Design Decision: Add dropout and additional layer for better regularization
        # This helps the model learn task-specific features while preventing overfitting
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2),  # Less dropout in second layer
            nn.Linear(512, num_classes)
        )
        
        # Initialize the new layers with Xavier initialization
        # Design Decision: Proper initialization helps with training stability
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize new layers with Xavier uniform initialization"""
        for m in self.backbone.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def freeze_backbone(self):
        """
        Freeze backbone layers for fine-tuning
        
        Design Decision: During initial training, we can freeze the backbone
        and only train the classification head. This is useful when:
        1. Dataset is very small
        2. Want faster initial training
        3. Prevent catastrophic forgetting of pretrained features
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze the final FC layers
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
        
        print("Backbone frozen. Only training final layers.")
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for full fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen. Training all layers.")


class FaceMaskEfficientNet(nn.Module):
    """
    Alternative model using EfficientNet-B0
    
    Design Decision: EfficientNet is more parameter-efficient than ResNet
    and can achieve similar accuracy with fewer parameters.
    Use this if computational resources are limited or for deployment.
    """
    
    def __init__(self, num_classes=3, pretrained=True, dropout_rate=0.5):
        super(FaceMaskEfficientNet, self).__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Get number of features
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.backbone(x)


def create_model(model_name='resnet50', num_classes=3, pretrained=True):
    """
    Factory function to create models
    
    Args:
        model_name: 'resnet50' or 'efficientnet'
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        Model instance
    
    Design Decision: Using a factory function makes it easy to experiment
    with different architectures without changing training code.
    """
    if model_name == 'resnet50':
        return FaceMaskClassifier(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'efficientnet':
        return FaceMaskEfficientNet(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # Test model creation
    model = create_model('resnet50', num_classes=3)
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"Model created successfully")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")