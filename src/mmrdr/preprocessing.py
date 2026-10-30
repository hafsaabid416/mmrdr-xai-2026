#image preprocessing 

#all functions take and return uint8 BGR images 

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import cv2
import numpy as np


@lru_cache(maxsize=64)
def _circle(h: int, w: int, shrink: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    r = shrink * max(h, w) / 2
    return (xx - (w - 1) / 2) ** 2 + (yy - (h - 1) / 2) ** 2 <= r ** 2

         #boolean mask of the retinal field of view
def fov_mask(img: np.ndarray, shrink: float = 0.99) -> np.ndarray:
    
    return _circle(img.shape[0], img.shape[1], shrink)

     
def pad_square(img: np.ndarray) -> np.ndarray:
    
    h, w = img.shape[:2]
    s = max(h, w)
    out = np.zeros((s, s) + img.shape[2:], img.dtype)                           #Centre the image on a black square canvas
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    return out
                 

def preprocess(img: np.ndarray, size: int = 512) -> np.ndarray:
    
    img = img.copy()
    img[~fov_mask(img)] = 0
    return cv2.resize(pad_square(img), (size, size), interpolation=cv2.INTER_AREA)


def retina_region(img: np.ndarray) -> np.ndarray:
    
    return fov_mask(img) & (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 0)


def graham(img: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    
    out = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigma), -4, 128)
    out[~retina_region(img)] = 0
    return out


def clahe_lab(img: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lab[..., 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    out[~retina_region(img)] = 0
    return out


NORMALISERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "none": lambda x: x,
    "graham": graham,
    "clahe": clahe_lab,
}
