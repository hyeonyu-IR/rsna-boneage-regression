from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yaml


def safe_read_yaml(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_read_history_best(history_path: Path) -> Tuple[Optional[float], Optional[int]]:
    """
    Returns (best_val_mae, best_epoch) from history.csv if present.
    """
    try:
        df = pd.read_csv(history_path)
        if "val_mae" not in df.columns:
            return None, None
        i = df["val_mae"].astype(float).idxmin()
        best_val_mae = float(df.loc[i, "val_mae"])
        best_epoch = int(df.loc[i, "epoch"]) if "epoch" in df.columns else None
        return best_val_mae, best_epoch
    except Exception:
        return None, None


def get_cfg_fields(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not cfg:
        return {}

    out: Dict[str, Any] = {}

    # Common fields we care about
    out["use_sex"] = cfg.get("data", {}).get("use_sex", None)
    out["img_size"] = cfg.get("preprocess", {}).get("img_size", None)
    out["normalize"] = cfg.get("preprocess", {}).get("normalize", None)
    out["augment_enabled"] = cfg.get("preprocess", {}).get("augment", {}).get("enabled", None)
    out["rotate_deg"] = cfg.get("preprocess", {}).get("augment", {}).get("rotate_deg", None)
    out["brightness"] = cfg.get("preprocess", {}).get("augment", {}).get("brightness", None)
    out["contrast"] = cfg.get("preprocess", {}).get("augment", {}).get("contrast", None)
    out["hflip"] = cfg.get("preprocess", {}).get("augment", {}).get("hflip", None)

    out["batch_size"] = cfg.get("dataloader", {}).get("batch_size", None)
    out["num_workers"] = cfg.get("dataloader", {}).get("num_workers", None)

    out["epochs"] = cfg.get("train", {}).get("epochs", None)
    out["lr"] = cfg.get("train", {}).get("lr", None)
    out["weight_decay"] = cfg.get("train", {}).get("weight_decay", None)
    out["loss"] = cfg.get("train", {}).get("loss", None)

    # If later you add backbone_name to config, this will start populating automatically
    out["backbone"] = cfg.get("model", {}).get("backbone", None)

    return out


def infer_backbone_from_files(run_dir: Path, cfg: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Best-effort backbone inference:
    - If cfg has model.backbone, use it
    - Else, fall back to None (explicit is better than guessing incorrectly)
    """
    if cfg:
        bb = cfg.get("model", {}).get("backbone", None)
        if bb:
            return str(bb)
    return None


def collect_run_row(run_dir: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {"run": run_dir.name}

    cfg_path = run_dir / "config_used.yaml"
    metrics_path = run_dir / "metrics.json"
    history_path = run_dir / "history.csv"

    cfg = safe_read_yaml(cfg_path) if cfg_path.exists() else None
    metrics = safe_read_json(metrics_path) if metrics_path.exists() else None

    # Best val MAE: prefer metrics.json, fallback to history.csv
    best_val_mae = None
    best_epoch = None

    if metrics and "best_val_mae" in metrics:
        try:
            best_val_mae = float(metrics["best_val_mae"])
        except Exception:
            best_val_mae = None

    if best_val_mae is None and history_path.exists():
        best_val_mae, best_epoch = safe_read_history_best(history_path)

    # Also capture final epoch val_mae if present
    final_val_mae = None
    final_train_mae = None
    if history_path.exists():
        try:
            h = pd.read_csv(history_path)
            if "val_mae" in h.columns:
                final_val_mae = float(h["val_mae"].iloc[-1])
            if "train_mae" in h.columns:
                final_train_mae = float(h["train_mae"].iloc[-1])
            if best_epoch is None and "val_mae" in h.columns and "epoch" in h.columns:
                i = h["val_mae"].astype(float).idxmin()
                best_epoch = int(h.loc[i, "epoch"])
        except Exception:
            pass

    row["best_val_mae"] = best_val_mae
    row["best_epoch"] = best_epoch
    row["final_val_mae"] = final_val_mae
    row["final_train_mae"] = final_train_mae

    # Config fields
    row.update(get_cfg_fields(cfg))

    # backbone (best-effort)
    row["backbone"] = row.get("backbone") or infer_backbone_from_files(run_dir, cfg)

    # Artifacts
    row["has_best_pt"] = (run_dir / "best.pt").exists()
    row["has_best_ts"] = (run_dir / "best.ts").exists() or any(run_dir.glob("*.ts"))
    row["report_md"] = (run_dir / "run_report.md").exists()

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="outputs/runs", help="Runs directory")
    parser.add_argument("--top", type=int, default=10, help="Show top N runs")
    parser.add_argument("--save_csv", action="store_true", help="Save comparison table to outputs/runs_compare.csv")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    runs_dir = (repo_root / args.runs_dir).resolve()

    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs dir not found: {runs_dir}")

    run_dirs = sorted([p for p in runs_dir.glob("run_*") if p.is_dir()])
    if not run_dirs:
        raise RuntimeError(f"No run folders found in: {runs_dir}")

    rows = [collect_run_row(d) for d in run_dirs]
    df = pd.DataFrame(rows)

    # Sort by best_val_mae (ascending), but keep runs with missing metrics at bottom
    df["best_val_mae_sort"] = df["best_val_mae"].fillna(1e9)
    df = df.sort_values(["best_val_mae_sort", "run"], ascending=[True, True]).drop(columns=["best_val_mae_sort"])

    # Print summary
    show_cols = [
        "run",
        "best_val_mae",
        "best_epoch",
        "final_val_mae",
        "final_train_mae",
        "backbone",
        "img_size",
        "loss",
        "use_sex",
        "augment_enabled",
        "rotate_deg",
        "brightness",
        "contrast",
        "batch_size",
        "lr",
        "weight_decay",
        "has_best_pt",
        "has_best_ts",
        "report_md",
    ]
    show_cols = [c for c in show_cols if c in df.columns]

    topn = min(args.top, len(df))
    print("\n=== Run Comparison (Top {}) ===\n".format(topn))
    print(df[show_cols].head(topn).to_string(index=False))

    # Print “winner” path convenience
    best = df.iloc[0]
    best_run_dir = runs_dir / str(best["run"])
    print("\n=== Best Run ===")
    print("Run folder:", best_run_dir)
    if (best_run_dir / "best.pt").exists():
        print("Checkpoint (pt):", best_run_dir / "best.pt")
    ts_candidates = list(best_run_dir.glob("*.ts"))
    if ts_candidates:
        print("TorchScript (ts):", ts_candidates[0])

    if args.save_csv:
        out_csv = runs_dir / "runs_compare.csv"
        df.to_csv(out_csv, index=False)
        print("\nSaved:", out_csv)


if __name__ == "__main__":
    main()
