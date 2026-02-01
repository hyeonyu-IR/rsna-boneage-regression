from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import argparse
from pathlib import Path

import torch

from src.infer import load_checkpoint, RegressionModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--out", type=str, default=None, help="Output .ts path (optional)")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    ckpt, cfg = load_checkpoint(ckpt_path)

    use_sex = bool(cfg["data"].get("use_sex", True))
    backbone_name = "resnet34"

    model = RegressionModel(backbone_name=backbone_name, use_sex=use_sex)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # Scriptable forward. We export with both signatures supported:
    # - if use_sex: forward(x, sex)
    # - else: forward(x)
    example_x = torch.randn(1, 3, int(cfg["preprocess"]["img_size"]), int(cfg["preprocess"]["img_size"]))
    if use_sex:
        example_sex = torch.tensor([0])
        scripted = torch.jit.trace(model, (example_x, example_sex))
    else:
        scripted = torch.jit.trace(model, (example_x,))

    out_path = Path(args.out) if args.out else ckpt_path.with_suffix(".ts")
    scripted.save(str(out_path))
    print("Saved TorchScript:", out_path)
    print("Recommendation: Use the .ts file for deployment instead of .pt")



if __name__ == "__main__":
    main()
