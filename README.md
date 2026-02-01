# RSNA Bone Age Regression

A research-grade deep learning pipeline for automated bone age estimation from pediatric hand radiographs using the RSNA Bone Age dataset.

This repository provides a fully reproducible framework for model training, evaluation, inference, and deployment, designed to support academic research and real-world clinical experimentation.

---

## Key Features

- Reproducible training pipeline driven by YAML configuration
- Stratified regression splitting for robust validation
- Torchvision backbone models (ResNet family)
- Automated experiment logging and run reports
- Single-image inference module
- TorchScript export for production environments
- Streamlit-based web application for rapid clinical testing
- Notebook-based dataset auditing and sanity checks

---

## Repository Structure

```
boneage-regression/
│
├── configs/              # Training configuration files  
├── notebooks/            # Dataset audits and sanity checks  
├── src/
│   ├── data/             # Dataset, transforms, splitting  
│   ├── train.py          # Training pipeline  
│   └── infer.py          # Single-image inference  
│
├── scripts/
│   ├── 01_train.py
│   ├── 03_export_torchscript.py
│   └── 04_make_report.py
│
└── outputs/
    └── runs/            # Experiment artifacts
```

---

## Dataset

This project uses the **RSNA Pediatric Bone Age Dataset**, available via Kaggle:

https://www.kaggle.com/c/rsna-bone-age

After downloading, organize your data directory:

```
data/boneage/
├── boneage-training-dataset/
├── boneage-test-dataset/
├── boneage-training-dataset.csv
└── boneage-test-dataset.csv
```

Update the path in:

```
configs/baseline.yaml
```

---

## Installation

Create and activate your environment:

```bash
conda create -n medimg python=3.10
conda activate medimg

pip install torch torchvision pandas matplotlib pillow pyyaml streamlit
```

(Adjust CUDA installation as appropriate.)

---

## Training

Run the baseline experiment:

```bash
python -m src.train --config configs/baseline.yaml
```

Outputs are saved automatically:

```
outputs/runs/run_YYYYMMDD_HHMMSS/
```

Including:

- best model checkpoint  
- training history  
- metrics  
- configuration snapshot  

---

## Generate an Experiment Report

```bash
python scripts/04_make_report.py
```

Produces:

- MAE curve  
- RMSE curve  
- structured markdown report  

These artifacts support reproducible research and manuscript preparation.

---

## Inference

After training:

```bash
python -m src.infer     --model outputs/runs/<run>/best.pt     --image path/to/hand_xray.png     --sex M
```

Example output:

```
Predicted Bone Age: 132.4 months (11.0 years)
```

---

## Deployment (Streamlit)

Launch the web application:

```bash
streamlit run app/streamlit_app.py
```

Upload a radiograph to obtain an immediate bone age prediction.

Designed for rapid clinical prototyping.

---

## Baseline Performance

| Model | Validation MAE |
|--------|----------------|
| ResNet34 | ~7.6 months |

This performance is within the clinically credible range for automated bone age estimation using a single-model architecture.

---

## Roadmap

Planned improvements:

- Cross-validation pipeline  
- Larger backbone architectures  
- Uncertainty estimation  
- Model ensembling  
- Grad-CAM interpretability  
- DICOM support  
- PACS-compatible inference service  

---

## Reproducibility

Each run stores:

- configuration snapshot  
- training metrics  
- model checkpoint  

ensuring complete experimental traceability.

---

## License

MIT License.

---

## Author

Hyeon Yu, MD  
Professor of Radiology  
University of North Carolina at Chapel Hill  

Interventional Radiology | Artificial Intelligence in Medical Imaging

---

## Citation (Future)

If this repository supports academic work, please cite accordingly.  
A formal citation entry will be provided upon publication.
