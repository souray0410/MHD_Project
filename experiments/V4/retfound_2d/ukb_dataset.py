"""Small, privacy-preserving UK Biobank input adapter for RETFound checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
CACHE_VERSION = "paired_roi_square_resize_rgb_v1"


class RETFoundUKBDataset(Dataset):
    """Read CFP or OCT images from LOOK's existing paired 224x224 cache."""

    def __init__(
        self,
        labels_csv: str | Path,
        cache_root: str | Path,
        *,
        modality: str,
        split: str = "train",
        limit: int | None = None,
    ) -> None:
        if modality not in {"cfp", "oct"}:
            raise ValueError("modality must be 'cfp' or 'oct'")
        frame = pd.read_csv(
            labels_csv,
            dtype={"participant_id": str},
            usecols=["participant_id", "fundus_path", "oct_path", "label_id", "split"],
        )
        frame = frame.loc[frame["split"] == split].reset_index(drop=True)
        if limit is not None:
            frame = frame.iloc[:limit].copy()
        if frame.empty:
            raise ValueError(f"No UKB rows for split={split!r}")
        labels = frame["label_id"].astype(int)
        if labels.min() < 0 or labels.max() > 4:
            raise ValueError("Expected UKB label_id values in [0, 4]")
        self.frame = frame
        self.cache_root = Path(cache_root) / CACHE_VERSION
        self.modality_index = 0 if modality == "cfp" else 1

    def __len__(self) -> int:
        return len(self.frame)

    def _cache_path(self, row: pd.Series) -> Path:
        identity = (
            f"{CACHE_VERSION}\0{224}\0{row['fundus_path']}\0{row['oct_path']}"
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self.cache_root / digest[:2] / f"{digest}.npy"

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        path = self._cache_path(row)
        if not path.is_file():
            raise FileNotFoundError(f"Preprocessed UKB cache entry is missing: {path}")
        pair = np.load(path, allow_pickle=False)
        if pair.shape != (2, 224, 224, 3) or pair.dtype != np.uint8:
            raise ValueError(f"Invalid UKB pair cache shape/dtype: {pair.shape}/{pair.dtype}")
        image = torch.from_numpy(pair[self.modality_index].copy()).permute(2, 0, 1).float() / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "image": image,
            "target": torch.tensor(int(row["label_id"]), dtype=torch.long),
        }

