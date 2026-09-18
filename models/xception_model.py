"""
Phase 3 - Xception Model Module
================================
Xception with ImageNet pretrained weights via timm.
Fixed architecture across all experiments.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class XceptionClassifier(nn.Module):
    """
    Xception backbone with custom classification head.
    Uses timm for pretrained weights.
    """
    
    def __init__(self, num_classes=5, pretrained=True, 
                 dropout_rate=0.5, hidden_dim=512, classifier_dropout=0.3):
        super(XceptionClassifier, self).__init__()
        
        import timm
        
        # Load Xception backbone
        self.backbone = timm.create_model('xception', pretrained=pretrained)
        num_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)  # Remove original classifier
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.num_features = num_features
        self.num_classes = num_classes
        
        logger.info(f"XceptionClassifier: features={num_features}, "
                    f"classes={num_classes}, hidden={hidden_dim}")
    
    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, x):
        """Extract penultimate-layer features for embedding analysis."""
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        # Pass through first two layers of classifier (dropout + linear + relu)
        x = self.classifier[0](features)  # Dropout
        x = self.classifier[1](x)         # Linear -> hidden_dim
        x = self.classifier[2](x)         # ReLU
        return x


def build_model(config, num_classes=None):
    """
    Build Xception model from config.
    
    Args:
        config: base config dict
        num_classes: override number of classes (auto-detected from dataset if None)
    
    Returns:
        XceptionClassifier model
    """
    model_cfg = config.get('model', {})
    
    if num_classes is None:
        num_classes = 2  # Default to binary
    
    model = XceptionClassifier(
        num_classes=num_classes,
        pretrained=model_cfg.get('pretrained', True),
        dropout_rate=model_cfg.get('dropout_rate', 0.5),
        hidden_dim=model_cfg.get('classifier_hidden', 512),
        classifier_dropout=model_cfg.get('classifier_dropout', 0.3),
    )
    
    return model


def get_num_classes(config, dataset_name, task=None):
    """Get number of classes for a dataset/task."""
    if dataset_name == 'aptos':
        return config['data']['datasets']['aptos']['num_classes']
    elif dataset_name == 'skin':
        return config['data']['datasets']['skin']['num_classes']
    elif dataset_name == 'hyperkvasir':
        return 2  # All subtasks are binary
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
