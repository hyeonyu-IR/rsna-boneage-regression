from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import yaml
from torchvision import models

# --- Make repo root importable (Windows-safe) ---
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.transforms import TransformConfig, build_eval_transform


# =========================
# Model (must match train.py)
# =========================
class RegressionModel(nn.Module):
    def __init__(self, backbone_name: str = "resnet34", use_sex: bool = True):
        super().__init__()
        self.use_sex = use_sex
        bb = backbone_name.lower().strip()

        if bb == "resnet34":
            m = models.resnet34(weights=None)
            n_feat = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
        elif bb == "resnet50":
            m = models.resnet50(weights=None)
            n_feat = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
        else:
            raise ValueError(f"Unsupported backbone: {bb}")

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
                raise ValueError("use_sex=True but sex tensor is None.")
            sex = sex.float().view(-1, 1)
            feat = torch.cat([feat, sex], dim=1)
        return self.head(feat)


# =========================
# Helpers
# =========================
def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def sex_to_bin(choice: str) -> int:
    if choice == "Male":
        return 1
    if choice == "Female":
        return 0
    return -1


def build_infer_transform(cfg: Dict[str, Any]):
    tcfg = TransformConfig(
        img_size=int(cfg["preprocess"]["img_size"]),
        grayscale_to_rgb=bool(cfg["preprocess"]["grayscale_to_rgb"]),
        normalize=str(cfg["preprocess"]["normalize"]),
        augment_enabled=False,
    )
    return build_eval_transform(tcfg)


def find_runs(runs_root: Path) -> List[Path]:
    return sorted([p for p in runs_root.glob("run_*") if p.is_dir()])


def extract_run_metadata(run_dir: Path) -> Dict[str, Any]:
    """
    Pull useful metadata from:
      - val_error_summary.json (preferred if you ran 06_eval_run.py)
      - metrics.json
      - config_used.yaml
    """
    meta: Dict[str, Any] = {"run": run_dir.name, "path": str(run_dir)}

    cfg_path = run_dir / "config_used.yaml"
    metrics_path = run_dir / "metrics.json"
    evalsum_path = run_dir / "val_error_summary.json"

    cfg = load_yaml(cfg_path) if cfg_path.exists() else None
    metrics = safe_read_json(metrics_path) if metrics_path.exists() else None
    evalsum = safe_read_json(evalsum_path) if evalsum_path.exists() else None

    # backbone
    backbone = None
    if cfg:
        backbone = cfg.get("model", {}).get("backbone", None)
    if backbone is None and metrics:
        backbone = metrics.get("backbone", None)

    # best mae
    best_mae = None
    best_epoch = None
    if evalsum:
        best_mae = evalsum.get("mae_months", None)
    if best_mae is None and metrics:
        best_mae = metrics.get("best_val_mae", None)

    if evalsum:
        best_epoch = evalsum.get("best_epoch", None)
    if best_epoch is None and metrics:
        best_epoch = metrics.get("best_epoch", None)

    # other config fields
    img_size = None
    loss = None
    use_sex = None
    normalize = None

    if cfg:
        img_size = cfg.get("preprocess", {}).get("img_size", None)
        normalize = cfg.get("preprocess", {}).get("normalize", None)
        loss = cfg.get("train", {}).get("loss", None)
        use_sex = cfg.get("data", {}).get("use_sex", None)

    meta.update(
        {
            "backbone": backbone,
            "best_mae_months": best_mae,
            "best_epoch": best_epoch,
            "img_size": img_size,
            "loss": loss,
            "use_sex": use_sex,
            "normalize": normalize,
            "has_best_pt": (run_dir / "best.pt").exists(),
            "has_best_ts": any(run_dir.glob("*.ts")),
            "has_eval_summary": evalsum_path.exists(),
        }
    )
    return meta


@st.cache_data
def load_training_csv(cfg: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Load the RSNA training CSV specified by config.
    """
    try:
        data_root = Path(cfg["data"]["root"])
        train_csv = data_root / cfg["data"]["train_csv"]
        if not train_csv.exists():
            return None
        df = pd.read_csv(train_csv)
        return df
    except Exception:
        return None


def lookup_ground_truth(df: pd.DataFrame, image_id: int) -> Optional[Dict[str, Any]]:
    """
    RSNA bone age Kaggle training CSV commonly includes:
      - id
      - boneage (months)
      - male (True/False or 0/1)
    Your template may have slightly different columns; handle flexibly.
    """
    # Determine id column
    id_col = None
    for c in ["id", "image_id", "ID"]:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        return None

    rows = df[df[id_col].astype(int) == int(image_id)]
    if len(rows) == 0:
        return None
    r = rows.iloc[0].to_dict()

    # Determine label column
    y_col = None
    for c in ["boneage", "bone_age", "age", "label"]:
        if c in df.columns:
            y_col = c
            break

    out = {"id": int(image_id)}
    if y_col is not None:
        out["boneage_true_months"] = float(r[y_col])
    if "male" in df.columns:
        out["male"] = r["male"]
    return out


def enable_mc_dropout(model: nn.Module) -> None:
    """
    Enable Dropout during inference for MC Dropout, but keep BatchNorm in eval mode.
    """
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            m.eval()
        if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d) or isinstance(m, nn.Dropout3d):
            m.train()


@torch.no_grad()
def predict_deterministic_ts(
    ts_model,
    x: torch.Tensor,
    use_sex: bool,
    sex_bin_value: int,
) -> float:
    if use_sex:
        s = torch.tensor([int(sex_bin_value)], device=x.device)
        pred = ts_model(x, s)
    else:
        pred = ts_model(x)
    return float(pred.squeeze().detach().cpu().item())


def load_eager_pt_model(run_dir: Path, cfg: Dict[str, Any], device: torch.device) -> RegressionModel:
    """
    Load best.pt checkpoint into eager model.
    Used for MC Dropout uncertainty.
    """
    ckpt_path = run_dir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone = str(cfg.get("model", {}).get("backbone", "resnet34")).lower()
    use_sex = bool(cfg.get("data", {}).get("use_sex", True))

    model = RegressionModel(backbone_name=backbone, use_sex=use_sex)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def mc_dropout_predict(
    eager_model: RegressionModel,
    x: torch.Tensor,
    use_sex: bool,
    sex_bin_value: int,
    n_samples: int = 20,
) -> Tuple[float, float]:
    """
    Returns (mean_months, std_months)
    """
    enable_mc_dropout(eager_model)

    preds = []
    for _ in range(int(n_samples)):
        if use_sex:
            s = torch.tensor([int(sex_bin_value)], device=x.device)
            p = eager_model(x, s)
        else:
            p = eager_model(x)
        preds.append(float(p.squeeze().detach().cpu().item()))

    mean = float(np.mean(preds))
    std = float(np.std(preds, ddof=1)) if len(preds) > 1 else 0.0
    eager_model.eval()
    return mean, std


def months_to_years(months: float) -> float:
    return months / 12.0


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Bone Age Estimator v2 (Local)", layout="centered")

st.title("Bone Age Estimator v2 (Local)")
st.caption("Local research prototype for RSNA bone age regression. Not for clinical use without validation.")

runs_root = REPO_ROOT / "outputs" / "runs"
runs = find_runs(runs_root)

if not runs:
    st.error(f"No runs found under: {runs_root}")
    st.stop()

# Build run options with metadata (and sort with best_mae if present)
run_metas = [extract_run_metadata(r) for r in runs]

def sort_key(m: Dict[str, Any]):
    v = m.get("best_mae_months", None)
    return (v if isinstance(v, (int, float)) else 1e9, m["run"])

run_metas = sorted(run_metas, key=sort_key)

def run_label(m: Dict[str, Any]) -> str:
    mae = m.get("best_mae_months", None)
    bb = m.get("backbone", None)
    if isinstance(mae, (int, float)):
        return f"{m['run']}  |  {bb}  |  MAE={mae:.3f} mo"
    return f"{m['run']}  |  {bb}"

labels = [run_label(m) for m in run_metas]
label_to_meta = {labels[i]: run_metas[i] for i in range(len(labels))}

with st.sidebar:
    st.header("Run selection")
    selected_label = st.selectbox("Choose a trained run", labels, index=0)
    meta = label_to_meta[selected_label]
    run_dir = Path(meta["path"])

    st.divider()
    st.header("Compute")
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device_choice = st.selectbox("Device", [default_device, "cpu"], index=0)

    st.divider()
    st.header("Uncertainty (MC Dropout)")
    enable_unc = st.checkbox("Enable uncertainty estimate (MC Dropout)", value=True)
    n_mc = st.slider("MC samples", min_value=5, max_value=50, value=20, step=5, disabled=not enable_unc)

    st.divider()
    st.header("Input")
    sex_choice = st.selectbox("Sex", ["Female", "Male", "Unknown"], index=0)

# Load config_used.yaml (required for transform parity)
cfg_path = run_dir / "config_used.yaml"
if not cfg_path.exists():
    st.error(f"Missing config_used.yaml in {run_dir}")
    st.stop()

cfg = load_yaml(cfg_path)
use_sex = bool(cfg.get("data", {}).get("use_sex", True))
backbone = str(cfg.get("model", {}).get("backbone", "resnet34")).lower()
img_size = int(cfg["preprocess"]["img_size"])
loss = str(cfg["train"]["loss"])
normalize = str(cfg["preprocess"]["normalize"])

# Build transform (deterministic)
tf = build_infer_transform(cfg)

# Load TorchScript for fast deterministic prediction
ts_candidates = list(run_dir.glob("*.ts"))
ts_model = None
device = torch.device(device_choice)

if ts_candidates:
    ts_path = str(ts_candidates[0])
    try:
        ts_model = torch.jit.load(ts_path, map_location=device)
        ts_model.eval()
    except Exception as e:
        st.warning(f"Failed to load TorchScript ({Path(ts_path).name}); will fall back to best.pt for deterministic inference. Error: {e}")
        ts_model = None
else:
    st.warning("No TorchScript (*.ts) found. Deterministic inference will use best.pt. (You can export via scripts/03_export_torchscript.py)")

# Optionally load eager .pt for MC dropout uncertainty
eager_model = None
if enable_unc:
    if not (run_dir / "best.pt").exists():
        st.warning("best.pt not found; cannot compute uncertainty.")
        enable_unc = False
    else:
        try:
            eager_model = load_eager_pt_model(run_dir, cfg, device=device)
        except Exception as e:
            st.warning(f"Failed to load best.pt for uncertainty: {e}")
            enable_unc = False

# Metadata panel
colA, colB = st.columns([1, 1])
with colA:
    st.subheader("Selected model")
    st.write(f"**Run:** {run_dir.name}")
    st.write(f"**Backbone:** {backbone}")
    st.write(f"**Image size:** {img_size}×{img_size}")
    st.write(f"**Loss:** {loss}")
with colB:
    st.subheader("Validation (if available)")
    best_mae = meta.get("best_mae_months", None)
    if isinstance(best_mae, (int, float)):
        st.metric("Best val MAE (months)", f"{best_mae:.3f}")
    else:
        st.write("Best val MAE: (not available)")
    st.write(f"**Normalization:** {normalize}")
    st.write(f"**use_sex:** {use_sex}")

st.divider()

# File upload
uploaded = st.file_uploader("Upload a hand radiograph (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"])
if uploaded is None:
    st.stop()

img = Image.open(uploaded).convert("RGB")
st.image(img, caption="Uploaded image", use_container_width=True)

# Parse ID from filename (RSNA training images: 1443.png -> id=1443)
image_id = None
try:
    name = Path(uploaded.name).stem
    if name.isdigit():
        image_id = int(name)
except Exception:
    image_id = None

# Prepare tensor
x = tf(img).unsqueeze(0).to(device)

# Sex tensor value
sex_bin_value = sex_to_bin(sex_choice)  # 0/1/-1

# Deterministic prediction
with torch.no_grad():
    if ts_model is not None:
        months_pred = predict_deterministic_ts(ts_model, x, use_sex=use_sex, sex_bin_value=sex_bin_value)
    else:
        # fall back to eager model from pt for deterministic if ts not available
        if eager_model is None:
            eager_model = load_eager_pt_model(run_dir, cfg, device=device)
        eager_model.eval()
        if use_sex:
            s = torch.tensor([int(sex_bin_value)], device=device)
            months_pred = float(eager_model(x, s).squeeze().detach().cpu().item())
        else:
            months_pred = float(eager_model(x).squeeze().detach().cpu().item())

years_pred = months_to_years(months_pred)

# Uncertainty
months_mean = months_pred
months_std = None
if enable_unc and eager_model is not None:
    months_mean, months_std = mc_dropout_predict(
        eager_model=eager_model,
        x=x,
        use_sex=use_sex,
        sex_bin_value=sex_bin_value,
        n_samples=int(n_mc),
    )

# Ground truth lookup (only if we can parse ID and training CSV exists)
df_train = load_training_csv(cfg)
gt = None
if image_id is not None and df_train is not None:
    gt = lookup_ground_truth(df_train, image_id)

# Display results
st.subheader("Prediction")

c1, c2, c3 = st.columns(3)
c1.metric("Bone age (months)", f"{months_pred:.1f}")
c2.metric("Bone age (years)", f"{years_pred:.2f}")
if months_std is not None:
    c3.metric("Uncertainty (σ, months)", f"{months_std:.2f}")
else:
    c3.write("")

if months_std is not None:
    lo = months_mean - 1.96 * months_std
    hi = months_mean + 1.96 * months_std
    st.write(f"**MC Dropout estimate:** {months_mean:.1f} months  (95% interval ≈ {lo:.1f}–{hi:.1f})")

# Plausibility guardrail (display only; do not clamp)
if months_pred < 0 or months_pred > 240:
    st.warning("⚠️ Prediction is outside typical pediatric range (0–240 months). Input may be out-of-distribution.")

# Ground truth panel
if gt is not None and "boneage_true_months" in gt:
    true_m = float(gt["boneage_true_months"])
    err = months_pred - true_m
    st.subheader("Ground truth (RSNA training CSV)")
    g1, g2, g3 = st.columns(3)
    g1.metric("True (months)", f"{true_m:.1f}")
    g2.metric("Abs error (months)", f"{abs(err):.1f}")
    g3.metric("Signed error (months)", f"{err:+.1f}")
elif image_id is not None:
    st.info("Filename looks like an RSNA ID, but ground truth could not be found in the training CSV.")
else:
    st.info("Ground truth lookup is available when the uploaded filename is an integer ID (e.g., 1443.png).")

with st.expander("Details (debug)"):
    st.write("Repo root:", str(REPO_ROOT))
    st.write("Run dir:", str(run_dir))
    st.write("TorchScript:", ts_candidates[0].name if ts_candidates else "(none)")
    st.write("Device:", str(device))
    st.write("Uploaded name:", uploaded.name)
    st.write("Parsed image_id:", image_id)
    st.write("use_sex:", use_sex)
    st.write("sex_bin_value:", sex_bin_value)
