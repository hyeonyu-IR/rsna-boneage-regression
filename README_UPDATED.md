# Bone Age Regression Template (RSNA / Kaggle)

A reusable, research-grade template for **pediatric bone age estimation** (regression in months) from hand radiographs.
This repository supports:

- Reproducible training runs with structured artifacts (`outputs/runs/run_*`)
- Dataset audits and sanity checks via notebooks (`notebooks/`)
- Automated reporting (per-run curves + markdown summary)
- **Side-by-side run comparison** (`runs_compare.csv` + evaluation summary)
- **Per-run evaluation** with stratified error analysis (age bins, sex) + worst-case review
- **Deployment artifacts** via TorchScript export (`best.ts`)
- **Streamlit app v2** with:
  - automatic run discovery + metadata panel
  - optional MC Dropout uncertainty estimate
  - ground-truth lookup for RSNA samples (when filename is an integer ID)

> **Disclaimer:** This is a research prototype and not validated for clinical decision-making.

---

## 1) Repository layout

```
boneage-regression-template/
  app/
    streamlit_app.py
    streamlit_app_v2.py
  configs/
    baseline.yaml
  notebooks/
    01_dataset_audit.ipynb
    02_quick_sanity_check.ipynb
  scripts/
    03_export_torchscript.py
    04_make_report.py
    05_compare_runs.py
    06_eval_run.py
  src/
    data/
      dataset.py
      split.py
      transforms.py
    infer.py
    train.py
  outputs/
    runs/
      run_YYYYMMDD_HHMMSS/
        best.pt
        best.ts
        config_used.yaml
        history.csv
        metrics.json
        val_error_summary.json
        val_predictions.csv
        worst_cases.csv
        error_by_agebin_12mo.csv
        error_by_sex.csv
        run_report.md
        mae_curve.png
        rmse_curve.png
```

**Important:** `outputs/` and `data/` should be ignored by git (see `.gitignore`).

---

## 2) Environment setup (Windows)

Activate your environment (example: `medimg`) and install Streamlit if needed:

```bash
conda activate medimg
pip install streamlit
```

(Optional) confirm PyTorch sees GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

---

## 3) Dataset location

This project expects the RSNA Bone Age dataset downloaded from Kaggle and arranged like:

```
C:\Users\hyeon\Documents\miniconda_medimg_env\data\boneage\
  boneage-training-dataset\
    1.png
    2.png
    ...
  boneage-test-dataset\
    ...
  boneage-training-dataset.csv
  boneage-test-dataset.csv
```

Note: If your unzip created nested folders (e.g., `boneage-training-dataset/boneage-training-dataset/`), move images up one level.

---

## 4) Notebook sanity checks (recommended)

From repo root:

```bash
conda activate medimg
jupyter lab
```

Open:

- `notebooks/01_dataset_audit.ipynb`
- `notebooks/02_quick_sanity_check.ipynb`

These help verify:
- images are loadable
- sizes are reasonable
- CSV IDs resolve to image files
- dataloader returns valid batches

---

## 5) Training

### Train a baseline model
```bash
conda activate medimg
python -m src.train --config configs/baseline.yaml
```

This creates a new run folder under:

```
outputs/runs/run_YYYYMMDD_HHMMSS/
```

Artifacts include:
- `best.pt` (training checkpoint; contains cfg + state dict)
- `config_used.yaml` (exact config snapshot)
- `history.csv` (per-epoch metrics)
- `metrics.json` (best val summary)

---

## 6) Export TorchScript (deployment artifact)

After training:

```bash
conda activate medimg
python scripts/03_export_torchscript.py --ckpt outputs/runs/run_YYYYMMDD_HHMMSS/best.pt
```

This produces:
- `best.ts` in the same run folder

**Recommendation:** Use `.ts` for apps/deployment; keep `.pt` for research/resume training.

---

## 7) Inference (CLI)

Predict bone age from a single image:

```bash
conda activate medimg
python -m src.infer --model outputs/runs/run_YYYYMMDD_HHMMSS/best.pt --image "C:\path\to\image.png" --sex F
```

Output:
- predicted months and years

---

## 8) Generate a per-run report

### (A) Report for latest run
```bash
conda activate medimg
python scripts/04_make_report.py
```

### (B) Report for a specific prior run
```bash
conda activate medimg
python scripts/04_make_report.py --run run_YYYYMMDD_HHMMSS
```

This writes into that run folder:
- `run_report.md`
- `mae_curve.png`
- `rmse_curve.png`

---

## 9) Compare runs (model selection)

### Create ranked table across all runs
```bash
conda activate medimg
python scripts/05_compare_runs.py --save_csv
```

Outputs:
- `outputs/runs/runs_compare.csv`

---

## 10) Evaluate runs (side-by-side, rigorous)

Evaluate one or multiple runs using the run’s saved config and split logic:

```bash
conda activate medimg
python scripts/06_eval_run.py --runs run_YYYYMMDD_HHMMSS run_YYYYMMDD_HHMMSS --save_summary
```

Per run, writes:
- `val_error_summary.json`
- `val_predictions.csv`
- `worst_cases.csv`
- `error_by_agebin_12mo.csv`
- `error_by_sex.csv`

Cross-run summary:
- `outputs/runs/eval_summary.csv`

---

## 11) Streamlit apps (local)

### Streamlit v1 (minimal)
```bash
conda activate medimg
streamlit run app/streamlit_app.py
```

### Streamlit v2 (recommended)
Includes:
- auto run discovery
- metadata panel
- RSNA ground-truth lookup when filename is an integer ID (e.g., `1443.png`)
- optional MC Dropout uncertainty (loads `best.pt`)
- deterministic prediction via TorchScript when available (`best.ts`)

Run:
```bash
conda activate medimg
streamlit run app/streamlit_app_v2.py
```

---

## 12) Git hygiene (important)

### Typical workflow
```bash
git status
git add .
git commit -m "Your message"
git push
```

### Prevent pushing large artifacts
This repo should include a `.gitignore` that excludes:
- `outputs/`
- `data/`
- `*.pt`, `*.ts`, `*.onnx`, etc.

If you already pushed `outputs/` once, remove it from tracking while keeping files locally:

```bash
git rm -r --cached outputs
git add .gitignore
git commit -m "Stop tracking outputs; add gitignore"
git push
```

---

## 13) Next steps

Recommended research upgrades (after your deployment pipeline is stable):
- Cross-validation training and reporting
- Uncertainty calibration and reliability diagrams
- Domain shift evaluation on local institutional radiographs (with IRB/governance)
- Robust input quality checks (e.g., non-hand detection)
