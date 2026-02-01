from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from PIL import Image
import yaml

from src.data.transforms import TransformConfig, build_eval_transform


# Keep model definition identical to train.py (copy/paste to avoid divergence)
from torchvision import models


class RegressionModel(nn.Module):
    def __init__(self, backbone_name: str = "resnet34", use_sex: bool = True):
        super().__init__()
        self.use_sex = use_sex

        if backbone_name == "resnet34":
            m = models.resnet34(weights=None)  # weights not needed at inference load
            n_feat = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
        elif backbone_name == "resnet50":
            m = models.resnet50(weights=None)
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

    def forward(self, x: torch.Tensor, sex: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.backbone(x)
        if self.use_sex:
            if sex is None:
                raise ValueError("Model configured with use_sex=True but sex input was None.")
            sex = sex.float().view(-1, 1)
            feat = torch.cat([feat, sex], dim=1)
        return self.head(feat)


def load_checkpoint(ckpt_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "model_state" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model_state': {ckpt_path}")
    cfg = ckpt.get("cfg", None)
    if cfg is None:
        raise KeyError(
            "Checkpoint missing 'cfg'. Ensure train.py saves {'cfg': cfg} in best.pt."
        )
    return ckpt, cfg


def build_infer_transform_from_cfg(cfg: Dict[str, Any]):
    tcfg = TransformConfig(
        img_size=int(cfg["preprocess"]["img_size"]),
        grayscale_to_rgb=bool(cfg["preprocess"]["grayscale_to_rgb"]),
        normalize=str(cfg["preprocess"]["normalize"]),
        augment_enabled=False,  # inference must be deterministic
    )
    return build_eval_transform(tcfg)


def sex_to_binary(sex: Optional[str]) -> int:
    """
    Return: 1 for male, 0 for female, -1 unknown.
    """
    if sex is None:
        return -1
    s = str(sex).strip().lower()
    if s in ["m", "male", "1", "true"]:
        return 1
    if s in ["f", "female", "0", "false"]:
        return 0
    return -1


@torch.no_grad()
def predict_bone_age(
    ckpt_path: Path,
    image_path: Path,
    sex: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Returns:
      (months, years)
    """
    ckpt, cfg = load_checkpoint(ckpt_path)

    use_sex = bool(cfg["data"].get("use_sex", True))

    # If you later add backbone_name to cfg, read it here.
    backbone_name = "resnet34"

    model = RegressionModel(backbone_name=backbone_name, use_sex=use_sex)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)

    tf = build_infer_transform_from_cfg(cfg)

    img = Image.open(image_path).convert("RGB")  # transforms will convert to grayscale/3ch if configured
    x = tf(img).unsqueeze(0).to(dev)

    sex_tensor = None
    if use_sex:
        s = sex_to_binary(sex)
        sex_tensor = torch.tensor([s], device=dev)

    pred = model(x, sex=sex_tensor)  # shape (1,1)
    months = float(pred.squeeze().detach().cpu().item())
    years = months / 12.0
    return months, years


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image (png/jpg)")
    parser.add_argument("--sex", type=str, default=None, help="M/F (optional)")
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu | cuda:0 ... (optional)")
    args = parser.parse_args()

    months, years = predict_bone_age(
        ckpt_path=Path(args.model),
        image_path=Path(args.image),
        sex=args.sex,
        device=args.device,
    )

    print(f"Predicted Bone Age: {months:.1f} months ({years:.2f} years)")


if __name__ == "__main__":
    main()
