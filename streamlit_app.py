from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st
import torch
from PIL import Image

from src.infer import load_checkpoint, RegressionModel, sex_to_binary
from src.infer import build_infer_transform_from_cfg


st.set_page_config(page_title="Bone Age Estimator", layout="centered")


@st.cache_resource
def load_model_from_ckpt(ckpt_path: str):
    ckpt, cfg = load_checkpoint(Path(ckpt_path))
    use_sex = bool(cfg["data"].get("use_sex", True))
    backbone_name = "resnet34"

    model = RegressionModel(backbone_name=backbone_name, use_sex=use_sex)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tf = build_infer_transform_from_cfg(cfg)
    return model, tf, cfg, device


def main():
    st.title("RSNA Bone Age Regression (Prototype)")
    st.caption("Upload a hand radiograph and obtain a bone age estimate (months and years).")

    # Default: point to a run checkpoint (you can hardcode your latest run here if you want)
    ckpt_path = st.text_input(
        "Checkpoint path (best.pt)",
        value="outputs/runs/run_YYYYMMDD_HHMMSS/best.pt",
    )

    if not Path(ckpt_path).exists():
        st.warning("Checkpoint path does not exist yet. Please update it to a real run folder.")
        st.stop()

    model, tf, cfg, device = load_model_from_ckpt(ckpt_path)
    use_sex = bool(cfg["data"].get("use_sex", True))

    sex = None
    if use_sex:
        sex = st.selectbox("Sex (optional but recommended)", ["F", "M", "Unknown"], index=0)

    file = st.file_uploader("Upload hand X-ray (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])
    if file is None:
        st.stop()

    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    # Predict
    x = tf(img).unsqueeze(0).to(device)

    sex_tensor = None
    if use_sex:
        s = None if sex == "Unknown" else sex
        s_bin = sex_to_binary(s)
        sex_tensor = torch.tensor([s_bin], device=device)

    with torch.no_grad():
        pred = model(x, sex=sex_tensor) if use_sex else model(x)
        months = float(pred.squeeze().detach().cpu().item())
        years = months / 12.0

    st.subheader("Prediction")
    st.metric("Bone age (months)", f"{months:.1f}")
    st.metric("Bone age (years)", f"{years:.2f}")


if __name__ == "__main__":
    main()
