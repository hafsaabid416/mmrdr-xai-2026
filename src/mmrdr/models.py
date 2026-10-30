
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

from . import NUM_GRADES, NUM_LESIONS

       
def _backbone(name: str, pretrained: bool) -> tuple[nn.Module, int, nn.Module]:
    """Returns (feature extractor producing a spatial map, feature dim, Grad-CAM target layer)."""
    if name == "resnet50":
        m = torchvision.models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        feats = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        return feats, 2048, m.layer4
    if name == "convnext_tiny":
        m = torchvision.models.convnext_tiny(weights="IMAGENET1K_V1" if pretrained else None)
        return m.features, 768, m.features[-1]
    raise ValueError(f"unknown backbone {name}")
             #returns feature extractor producing a spatial map, feature dim, grad-cam target layer

class DRNet(nn.Module):
    def __init__(self, backbone: str = "resnet50", pretrained: bool = True, grade_head: str = "corn",
                 lesion_head: bool = True, bottleneck: bool = False, dropout: float = 0.2):
        super().__init__()
        self.init_args = dict(backbone=backbone, pretrained=False, grade_head=grade_head,
                              lesion_head=lesion_head, bottleneck=bottleneck, dropout=dropout)
        if bottleneck and not lesion_head:
            raise ValueError("bottleneck model requires the lesion head")
        if grade_head not in {"softmax", "corn"}:
            raise ValueError(grade_head)
        self.grade_head_type, self.bottleneck = grade_head, bottleneck
        self.features, dim, self.cam_layer = _backbone(backbone, pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        n_out = NUM_GRADES if grade_head == "softmax" else NUM_GRADES - 1
        self.lesion_head = nn.Linear(dim, NUM_LESIONS) if lesion_head else None
        # Concept bottleneck 
        self.grade_head = nn.Linear(NUM_LESIONS if bottleneck else dim, n_out)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.drop(torch.flatten(self.pool(self.features(x)), 1))
        out = {}
        if self.lesion_head is not None:
            out["lesion_logits"] = self.lesion_head(f)
        out["grade_logits"] = self.grade_head(torch.sigmoid(out["lesion_logits"]) if self.bottleneck else f)
        return out


def build_model(cfg: dict) -> DRNet:
    m = cfg["model"]
    return DRNet(backbone=m.get("backbone", "resnet50"), pretrained=m.get("pretrained", True),
                 grade_head=m["grade_head"], lesion_head=m.get("lesion_head", False),
                 bottleneck=m.get("bottleneck", False), dropout=m.get("dropout", 0.2))


def grade_probs(logits: torch.Tensor, head: str, temperature: float = 1.0) -> torch.Tensor:
    
    z = logits / temperature
    if head == "softmax":
        return torch.softmax(z, dim=1)
    cum = torch.cumprod(torch.sigmoid(z), dim=1)                     
    ones, zeros = torch.ones_like(cum[:, :1]), torch.zeros_like(cum[:, :1])
    upper = torch.cat([ones, cum], 1)                                
    lower = torch.cat([cum, zeros], 1)                               
    return (upper - lower).clamp_min(0)
