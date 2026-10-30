#Grad-CAM on the last conv stage and sanity check.

#Two targets: the expected grade E[grade], which works for both softmax and CORN heads, and a single lesion logit.


from __future__ import annotations

import cv2
import numpy as np
import torch
from scipy.stats import spearmanr

from . import LESIONS
from .models import DRNet, grade_probs


class GradCAM:
    def __init__(self, model: DRNet):
        self.model = model
        self._act = None
        self._grad = None
        self._handle = model.cam_layer.register_forward_hook(self._fwd)

    def remove(self) -> None:
        
        self._handle.remove()
        self._act = self._grad = None

    def _fwd(self, _m, _inp, out):
        self._act = out
        out.register_hook(lambda g: setattr(self, "_grad", g))

    def __call__(self, x: torch.Tensor, target: str = "grade") -> np.ndarray:
        
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            out = self.model(x.float())
            if target == "grade":
                p = grade_probs(out["grade_logits"].float(), self.model.grade_head_type)
                score = (p * torch.arange(p.shape[1], device=p.device)).sum()
            else:
                score = out["lesion_logits"][0, LESIONS.index(target)]
            score.backward()
        w = self._grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self._act).sum(1))[0].detach().float().cpu().numpy()
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]), interpolation=cv2.INTER_LINEAR)
        return cam / cam.max() if cam.max() > 0 else cam

         #RGB overlay the heatmap is hidden outside the retina
def overlay(img_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    out = cv2.addWeighted(img_bgr, 1 - alpha, heat, alpha, 0)
    out[img_bgr.max(axis=2) == 0] = 0
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def randomised_copy(model: DRNet, seed: int = 0) -> DRNet:
    
    torch.manual_seed(seed)
    dev = next(model.parameters()).device
    m = DRNet(**model.init_args)
    m.load_state_dict(model.state_dict())
    m = m.to(dev)
    for mod in list(m.cam_layer.modules()) + [m.grade_head] + ([m.lesion_head] if m.lesion_head is not None else []):
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters()
    return m.eval()

    #Spearman rank correlation of two maps at low resolution. 
def map_similarity(a: np.ndarray, b: np.ndarray, size: int = 32) -> float:
    
    ra = cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA).ravel()
    rb = cv2.resize(b, (size, size), interpolation=cv2.INTER_AREA).ravel()
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(spearmanr(ra, rb).statistic)
