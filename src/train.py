from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import yaml
from torchvision import models

from src.data.dataset import BoneAgeDataset
from src.data.transforms import TransformConfig, build_train_transform, build_eval_transform
from src.data.split import SplitConfig, make_train_val_split


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class RegressionModel(nn.Module):
    def __init__(self, backbone_name: str = "resnet50", use_sex: bool = True):
        super().__init__()
        self.use_sex = use_sex

        if backbone_name == "resnet34":
            m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            n_feat = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
        elif backbone_name == "resnet50":
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            n_feat = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

        in_head = n_feat + (1 if self.use_sex else 0)
        self.head = nn.Sequential(
            nn.Linear(in_head, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, sex: torch.Tensor | None = None) -> torch.Tensor:
        feat = self.backbone(x)
        if self.use_sex:
            if sex is None:
                raise ValueError("Model configured with use_sex=True but sex input was None.")
            sex = sex.float().view(-1, 1)
            feat = torch.cat([feat, sex], dim=1)
        return self.head(feat)


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - target)).detach().cpu().item())


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2)).detach().cpu().item())


def load_config(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_run_dir(repo_root: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = repo_root / "outputs" / "runs" / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main(cfg_path: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(Path(cfg_path))

    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    data_root = Path(cfg["data"]["root"])
    train_csv = data_root / cfg["data"]["train_csv"]
    train_img_dir = data_root / cfg["data"]["train_img_dir"]

    use_sex = bool(cfg["data"].get("use_sex", True))
    image_exts = cfg["data"].get("image_exts", ["png", "jpg", "jpeg"])

    # Split
    train_df = pd.read_csv(train_csv)
    split_cfg = SplitConfig(
        val_frac=float(cfg["data"]["val_frac"]),
        stratify_age_bins=int(cfg["data"]["stratify_age_bins"]),
        seed=seed,
    )
    tr_idx, va_idx = make_train_val_split(train_df, split_cfg, label_col="boneage")

    # Transforms
    tcfg = TransformConfig(
        img_size=int(cfg["preprocess"]["img_size"]),
        grayscale_to_rgb=bool(cfg["preprocess"]["grayscale_to_rgb"]),
        normalize=str(cfg["preprocess"]["normalize"]),
        augment_enabled=bool(cfg["preprocess"]["augment"]["enabled"]),
        rotate_deg=float(cfg["preprocess"]["augment"]["rotate_deg"]),
        brightness=float(cfg["preprocess"]["augment"]["brightness"]),
        contrast=float(cfg["preprocess"]["augment"]["contrast"]),
        hflip=bool(cfg["preprocess"]["augment"]["hflip"]),
    )
    train_tf = build_train_transform(tcfg)
    eval_tf = build_eval_transform(tcfg)

    # Datasets
    full_train = BoneAgeDataset(
        csv_path=train_csv,
        img_dir=train_img_dir,
        is_train=True,
        image_exts=image_exts,
        transform=train_tf,   # will be used for train subset
        use_sex=use_sex,
    )
    # For val, reuse dataset but with eval transforms
    full_val = BoneAgeDataset(
        csv_path=train_csv,
        img_dir=train_img_dir,
        is_train=True,
        image_exts=image_exts,
        transform=eval_tf,
        use_sex=use_sex,
    )

    ds_tr = Subset(full_train, tr_idx.tolist())
    ds_va = Subset(full_val, va_idx.tolist())

    dl_cfg = cfg["dataloader"]
    train_loader = DataLoader(
        ds_tr,
        batch_size=int(dl_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(dl_cfg["num_workers"]),
        pin_memory=bool(dl_cfg["pin_memory"]),
        drop_last=False,
    )
    val_loader = DataLoader(
        ds_va,
        batch_size=int(dl_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(dl_cfg["num_workers"]),
        pin_memory=bool(dl_cfg["pin_memory"]),
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model
    model = RegressionModel(backbone_name="resnet34", use_sex=use_sex).to(device)

    # Loss
    loss_name = str(cfg["train"]["loss"]).lower()
    if loss_name == "l1":
        criterion = nn.L1Loss()
    elif loss_name == "smoothl1":
        criterion = nn.SmoothL1Loss(beta=1.0)
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    run_dir = get_run_dir(repo_root)
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print("Run dir:", run_dir)

    best_val_mae = float("inf")
    history = []

    n_epochs = int(cfg["train"]["epochs"])

    for epoch in range(1, n_epochs + 1):
        # ---- train ----
        model.train()
        tr_losses = []
        tr_maes = []
        tr_rmses = []

        for x, y, meta in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            sex = None
            if use_sex:
                # meta["sex"] is a list-like of ints/None
                s = meta["sex"]
                # Replace None with -1; you can later handle unknown differently
                s = torch.tensor([(-1 if v is None else int(v)) for v in s], device=device)
                sex = s

            pred = model(x, sex=sex)

            loss = criterion(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            tr_losses.append(float(loss.detach().cpu().item()))
            tr_maes.append(mae(pred, y))
            tr_rmses.append(rmse(pred, y))

        # ---- val ----
        model.eval()
        va_losses = []
        va_maes = []
        va_rmses = []

        with torch.no_grad():
            for x, y, meta in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                sex = None
                if use_sex:
                    s = meta["sex"]
                    s = torch.tensor([(-1 if v is None else int(v)) for v in s], device=device)
                    sex = s

                pred = model(x, sex=sex)
                loss = criterion(pred, y)

                va_losses.append(float(loss.detach().cpu().item()))
                va_maes.append(mae(pred, y))
                va_rmses.append(rmse(pred, y))

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(tr_losses)),
            "train_mae": float(np.mean(tr_maes)),
            "train_rmse": float(np.mean(tr_rmses)),
            "val_loss": float(np.mean(va_losses)),
            "val_mae": float(np.mean(va_maes)),
            "val_rmse": float(np.mean(va_rmses)),
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train MAE={row['train_mae']:.2f} RMSE={row['train_rmse']:.2f} | "
            f"val MAE={row['val_mae']:.2f} RMSE={row['val_rmse']:.2f}"
        )

        # save best
        if row["val_mae"] < best_val_mae:
            best_val_mae = row["val_mae"]
            ckpt = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "best_val_mae": best_val_mae,
                "cfg": cfg,
            }
            torch.save(ckpt, run_dir / "best.pt")

        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    metrics = {"best_val_mae": best_val_mae}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Done. Best val MAE:", best_val_mae)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()

    main(args.config)
