import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class ScaleAttention(nn.Module):
    def __init__(self, feature_dim=2048):
        super(ScaleAttention, self).__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, features):
        attn_scores = self.attention_net(features)
        attn_weights = F.softmax(attn_scores, dim=1)
        attended_features = torch.sum(features * attn_weights, dim=1)
        return attended_features, attn_weights

class MultiScaleBreastCancerModel(nn.Module):
    # num_classes is now 1 because we are using BCEWithLogitsLoss
    def __init__(self, num_classes=1, freeze_backbone=True):
        super(MultiScaleBreastCancerModel, self).__init__()
        
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        self.attention = ScaleAttention(feature_dim=2048)
        
        # --- THE ULTIMATE MERGE: Deep Classifier Block ---
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, num_classes) # Outputs a single logit
        )

    def forward(self, images_dict):
        batch_size = images_dict['40X'].size(0)
        scales = ['40X', '100X', '200X', '400X']
        scale_features = []
        
        for scale in scales:
            x = images_dict[scale]
            feat = self.feature_extractor(x)
            feat = feat.view(batch_size, -1)
            scale_features.append(feat)
            
        stacked_features = torch.stack(scale_features, dim=1)
        fused_features, attn_weights = self.attention(stacked_features)
        output = self.classifier(fused_features)
        
        return output, attn_weights