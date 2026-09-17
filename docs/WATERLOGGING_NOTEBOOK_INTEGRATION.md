# Waterlogging notebook integration

The uploaded notebook already contains the code required to train and export a
waterlogging model.

## What the notebook does

It trains on historical weather/terrain data starting from 2024, creates 30 ML
features, compares multiple classifiers, tunes a Random Forest, tunes an F2
warning threshold and saves the final pipeline.

The executed notebook reported:

```text
Model                : Tuned Random Forest
Processed rows       : 95,136
Selected features    : 30
Warning threshold    : 0.36

Future-test metrics:
Accuracy             : 0.99937
Precision            : 0.97927
Recall               : 0.99736
F1                   : 0.98824
F2                   : 0.99369
ROC-AUC              : 0.99997
PR-AUC               : 0.99898
```

Important: the notebook itself says weak labels were active. These metrics
measure reproduction of its weak-label rule; they are not independent proof of
real-world disaster accuracy.

## The actual trained model was not embedded inside the .ipynb

The notebook output shows that Colab created:

```text
pkl_files/final_waterlogging_model.pkl
```

and that the downloaded file size was about 1,969,457 bytes.

A Jupyter notebook stores source code and outputs, but not the live Python
kernel's trained `final_model` object. Therefore the exact old model bytes cannot
be recovered from the `.ipynb` alone after the old Colab runtime is gone.

## One-command regeneration

This repository contains a training-only copy of the notebook:

```text
notebooks/waterlogging_train_only.ipynb
```

Run:

```powershell
python scripts\train_waterlogging_from_notebook.py
```

It executes the original training workflow and finally writes:

```text
models/waterlogging/waterlogging.pkl
models/waterlogging/waterlogging_alert_threshold.joblib
models/waterlogging/metadata.json
```

After that:

```powershell
python app.py
```

Live prediction:

```http
POST /api/predict/waterlogging/live
Content-Type: application/json

{
  "lat": 16.3067,
  "lon": 80.4365,
  "state": "Andhra Pradesh",
  "district": "Guntur"
}
```

The Flask live builder reproduces the notebook's feature engineering formulas
for all 30 selected predictors.
