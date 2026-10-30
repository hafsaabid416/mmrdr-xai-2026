
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

    #class-balanced weights normalised to mean 1.
def effective_number_weights(counts: np.ndarray, beta: float = 0.999) -> torch.Tensor:
    
    w = (1.0 - beta) / (1.0 - np.power(beta, counts.astype(np.float64)))
    return torch.tensor(w / w.mean(), dtype=torch.float32)

         #corn loss
def corn_loss(logits: torch.Tensor, y: torch.Tensor, sample_w: torch.Tensor | None = None) -> torch.Tensor:
    
    total, n = logits.new_zeros(()), 0
    for k in range(logits.shape[1]):
        sel = y > k - 1
        if sel.sum() == 0:
            continue
        target = (y[sel] > k).float()
        l = F.binary_cross_entropy_with_logits(logits[sel, k], target, reduction="none")
        if sample_w is not None:
            l = l * sample_w[sel]
        total = total + l.sum()
        n += int(sel.sum())
    return total / max(n, 1)

       #grade loss
class DRLoss(torch.nn.Module):
    

    def __init__(self, grade_head: str, class_weights: torch.Tensor | None, lesion_weight: float = 0.5,
                 lesion_pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.head = grade_head
        self.register_buffer("cw", class_weights if class_weights is not None else torch.empty(0))
        self.register_buffer("pw", lesion_pos_weight if lesion_pos_weight is not None else torch.empty(0))
        self.lam = lesion_weight

    def forward(self, out: dict, y: torch.Tensor, lesions: torch.Tensor) -> dict[str, torch.Tensor]:
        cw = self.cw if self.cw.numel() else None
        if self.head == "softmax":
            lg = F.cross_entropy(out["grade_logits"], y, weight=cw)
        else:
            lg = corn_loss(out["grade_logits"], y, cw[y] if cw is not None else None)
        parts = {"grade": lg}
        loss = lg
        if "lesion_logits" in out:
            ll = F.binary_cross_entropy_with_logits(out["lesion_logits"], lesions,
                                                    pos_weight=self.pw if self.pw.numel() else None)
            parts["lesion"] = ll
            loss = loss + self.lam * ll
        parts["loss"] = loss
        return parts
