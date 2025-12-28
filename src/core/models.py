import torch
import torch.nn as nn
from transformers import SwinModel, SwinConfig
from timm.models.layers import trunc_normal_

class InstanceRelocationModel(nn.Module):
    def __init__(self, num_ratios=3, img_size=640):
        super().__init__()
        self.num_ratios = num_ratios
        
        self.backbone = SwinModel(
            config=SwinConfig(
                img_size=img_size,
                patch_size=4,
                num_channels=1,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_path_rate=0.1,
            )
        )
        
        self.conv = nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1)
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256 * 20 * 20, 128),
                nn.ReLU(),
                nn.Linear(128, 8)
            ) for _ in range(self.num_ratios)
        ])
        
        self.apply(self._init_weights)
    
    def forward(self, x):
        features = self.backbone(x).last_hidden_state
        features = features.permute(0, 2, 1).reshape(x.size(0), 768, 20, 20)
        
        features = self.conv(features)
        features = features.reshape(features.size(0), -1)
        
        outputs = [head(features) for head in self.heads]
        output = torch.cat(outputs, dim=1)
        
        output = torch.sigmoid(output)
        output = 2 * output - 1
        return output

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

class InstanceRelocationModel_Tiny(nn.Module):
    def __init__(self, num_ratios=3, img_size=640):
        super().__init__()
        self.num_ratios = num_ratios
        
        self.backbone = SwinModel(
            config=SwinConfig(
                img_size=img_size,
                patch_size=4,
                num_channels=1,
                embed_dim=24,         
                depths=[2, 2, 4, 2],  
                num_heads=[1, 2, 4, 8],  
                window_size=8,         
                mlp_ratio=2.0,        
                qkv_bias=True,
                drop_path_rate=0.3,   
            )
        )
        
        self.conv = nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1) 
        self.shared_fc1 = nn.Linear(64 * 20 * 20, 64)
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(64, 8)
            ) for _ in range(self.num_ratios)
        ])
        
        self.apply(self._init_weights)
    
    def forward(self, x):
        features = self.backbone(x).last_hidden_state
        features = features.permute(0, 2, 1).reshape(x.size(0), 192, 20, 20)
        
        features = self.conv(features)
        features = features.reshape(features.size(0), -1)
        
        features = self.shared_fc1(features)
        
        outputs = [head(features) for head in self.heads]
        output = torch.cat(outputs, dim=1)
        
        output = torch.sigmoid(output)
        output = 2 * output - 1
        return output

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Parameter):
            trunc_normal_(m, std=0.02, a=-0.04, b=0.04)