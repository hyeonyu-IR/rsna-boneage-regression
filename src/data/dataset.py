from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from pathlib import Path
from typing import Union

DEFAULT_IMAGE_EXTS = ["png", "jpg", "jpeg"]


def _normalize_img_dir(img_dir: Union[str, Path]) -> Path:
    img_dir = Path(img_dir)

    # If img_dir contains exactly one child directory with the same name, use the child
    child = img_dir / img_dir.name
    if img_dir.exists() and child.exists() and child.is_dir():
        return child

    return img_dir


def _find_image_file(img_dir: Union[str, Path], case_id: Union[int, str], exts: Optional[List[str]] = None) -> Path:
    """
    RSNA Bone Age (Kaggle): images are typically stored as <id>.png under train/test folders.
    We search allowed extensions and raise a clear error if missing.
    """
    img_dir = Path(img_dir)
    exts = exts or DEFAULT_IMAGE_EXTS

    # Normalize case_id (CSV uses int ids for train; test uses "Case ID")
    cid = str(int(case_id)) if str(case_id).isdigit() else str(case_id)

    for ext in exts:
        p = img_dir / f"{cid}.{ext}"
        if p.exists():
            return p

    # Some variants include different naming; last resort glob.
    glob_hits = list(img_dir.glob(f"{cid}.*"))
    if glob_hits:
        return glob_hits[0]

    raise FileNotFoundError(
        f"Could not find image for case_id={case_id} in {img_dir} with extensions={exts}. "
        f"Example expected path: {img_dir / (cid + '.png')}"
    )


def _sex_to_binary(sex_value) -> Optional[int]:
    """
    Convert sex column variants to binary:
      - train CSV uses 'male' boolean (True/False)
      - test CSV uses 'Sex' with 'M'/'F'
    Returns 1 for male, 0 for female, None if unknown.
    """
    if sex_value is None or (isinstance(sex_value, float) and np.isnan(sex_value)):
        return None

    if isinstance(sex_value, (bool, np.bool_)):
        return int(bool(sex_value))

    s = str(sex_value).strip().lower()
    if s in ["m", "male", "1", "true", "t"]:
        return 1
    if s in ["f", "female", "0", "false"]:
        return 0
    return None


@dataclass
class BoneAgeSample:
    image_path: Path
    case_id: int
    boneage_months: Optional[float] = None
    sex_binary: Optional[int] = None


class BoneAgeDataset(Dataset):
    """
    PyTorch Dataset for RSNA Bone Age dataset (Kaggle).

    Train CSV columns (from your file):
      - id (int)
      - boneage (int months)
      - male (bool)

    Test CSV columns (from your file):
      - Case ID (int)
      - Sex ('M'/'F')
      - (no boneage label)

    __getitem__ returns:
      - if is_train: (x, y, meta)
      - else:       (x, meta)

    where:
      x    : torch.Tensor (C,H,W) float32 in [0,1] if transform=None
      y    : torch.Tensor shape (1,) float32 (months) for labeled data
      meta : dict with keys {'case_id','sex','image_path'}
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        img_dir: Union[str, Path],
        is_train: bool,
        image_exts: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        use_sex: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.img_dir = _normalize_img_dir(img_dir)

        if not any(self.img_dir.glob("*.*")):
            raise FileNotFoundError(
                f"No files found in image directory: {self.img_dir}. "
                "Check that images were extracted and the folder path is correct."
    )

        self.is_train = bool(is_train)
        self.image_exts = image_exts or DEFAULT_IMAGE_EXTS
        self.transform = transform
        self.use_sex = bool(use_sex)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        df = pd.read_csv(self.csv_path)

        if self.is_train:
            required = {"id", "boneage", "male"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"Train CSV missing columns {sorted(missing)}. Found={list(df.columns)}")

            self.case_ids = df["id"].astype(int).to_numpy()
            self.labels = df["boneage"].astype(float).to_numpy()
            self.sex = df["male"].apply(_sex_to_binary).to_numpy()
        else:
            required = {"Case ID", "Sex"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"Test CSV missing columns {sorted(missing)}. Found={list(df.columns)}")

            self.case_ids = df["Case ID"].astype(int).to_numpy()
            self.labels = None
            self.sex = df["Sex"].apply(_sex_to_binary).to_numpy()

        # Pre-resolve image paths (fail fast with clear errors)
        self.image_paths: List[Path] = []
        for cid in self.case_ids:
            self.image_paths.append(_find_image_file(self.img_dir, cid, self.image_exts))

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        case_id = int(self.case_ids[idx])

        sex_bin = -1
        if self.use_sex:
            v = self.sex[idx]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                sex_bin = int(v)

        img = Image.open(img_path).convert("L")  # keep deterministic; convert to RGB in transforms if needed

        if self.transform is not None:
            x = self.transform(img)  # expected to output torch.Tensor (C,H,W)
        else:
            arr = np.array(img, dtype=np.float32) / 255.0
            x = torch.from_numpy(arr)[None, ...]  # (1,H,W)

        meta = {"case_id": case_id, "sex": sex_bin, "image_path": str(img_path)}

        if self.is_train:
            y = torch.tensor([float(self.labels[idx])], dtype=torch.float32)
            return x, y, meta

        return x, meta
