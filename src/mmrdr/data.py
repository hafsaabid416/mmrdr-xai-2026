#Dataset, augmentation and loaders.

#Images are read from a cache of preprocessed 512×512 JPEGs (scripts/prepare_images.py)

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from . import LESIONS
from .preprocessing import NORMALISERS

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def augment(img: np.ndarray, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    
    size = img.shape[0]
    b, g = cfg.get("brightness", 0.2), cfg.get("gamma", 0.2)
    gain, offset, gamma = rng.uniform(1 - b, 1 + b), rng.uniform(-20, 20), rng.uniform(1 - g, 1 + g)
    v = np.clip(np.arange(256, dtype=np.float32) * gain + offset, 0, 255)               
    lut = np.clip(255.0 * (v / 255.0) ** gamma, 0, 255).astype(np.uint8)                
    lut[0] = 0                                                                       
    out = cv2.LUT(img, lut)
    if rng.random() < cfg.get("p_blur", 0.3):                                           
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, cfg.get("max_blur_sigma", 1.5)))
    if rng.random() < cfg.get("p_jpeg", 0.3):                                           
        _, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 90))])
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if rng.random() < 0.5:                                                              
        out = out[:, ::-1]
    s = cfg.get("scale", 0.05)                                                         
    M = cv2.getRotationMatrix2D((size / 2, size / 2), rng.uniform(-180, 180), rng.uniform(1 - s, 1 + s))
    return cv2.warpAffine(np.ascontiguousarray(out), M, (size, size), flags=cv2.INTER_LINEAR, borderValue=0)

    
def to_input(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1) * 255
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1) * 255
    x = x.to(device, non_blocking=True).float()
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


class FundusDataset(Dataset):
    

    def __init__(self, df: pd.DataFrame, image_dir: str | Path, normaliser: str = "none",
                 train: bool = False, aug_cfg: dict | None = None, seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.normalise = NORMALISERS[normaliser]
        self.train = train
        self.aug_cfg = aug_cfg or {}
        self.seed = seed
        self.epoch = 0
        self.grades = self.df["grade"].to_numpy(np.int64)
        self.lesions = self.df[list(LESIONS)].to_numpy(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

        #Preprocessed, normalised BGR uint8 image (no augmentation) also used for visualisation

    def load(self, i: int) -> np.ndarray:
        
        path = self.image_dir / Path(self.df.at[i, "image"]).name
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return self.normalise(img)

    def __getitem__(self, i: int):
        img = self.load(i)
        if self.train:
            rng = np.random.default_rng((self.seed, self.epoch, i))      
            img = augment(img, rng, self.aug_cfg)
        x = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))   
        return torch.from_numpy(x), self.grades[i], torch.from_numpy(self.lesions[i]), i


def make_loader(ds: FundusDataset, batch_size: int, num_workers: int, shuffle: bool,
                balanced_sampler: bool = False) -> DataLoader:
    sampler = None
    if balanced_sampler:
        counts = np.bincount(ds.grades)
        w = 1.0 / counts[ds.grades]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(ds), replacement=True)
        shuffle = False
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=shuffle or sampler is not None,
                      persistent_workers=num_workers > 0)
