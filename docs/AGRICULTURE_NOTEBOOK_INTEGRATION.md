# Agriculture notebook integration

The uploaded notebook is the agriculture/farmer-advisory model.

## Final production model in the notebook

The final training cell builds **five LightGBM classifiers** and stores them
together in one joblib/pickle bundle:

```text
farmer_advisory_pipeline.pkl
```

The five targets are:

```text
crop_weather_risk
irrigation
heat_stress
waterlogging_risk
spraying_suitability
```

The final strict feature contract is **35 model features**:

- 32 numeric features
- 3 encoded categorical features:
  `crop_enc`, `crop_stage_enc`, `agro_season_enc`

The notebook reports **102,424 ML rows x 51 columns** after its NASA POWER
training pipeline.

## Training source

Historical source used by the source notebook:

```text
NASA POWER daily point API
https://power.larc.nasa.gov/api/temporal/daily/point
```

Training dates:

```text
2021-01-01 -> 2024-12-31
```

Training grid:

```text
28 representative Indian districts / state-zone anchors
```

## Live inference source

The final notebook uses:

```text
Open-Meteo forecast API
Nominatim geocoding
```

The Flask implementation preserves the notebook's final inference feature
formulas.

## Important limitation

The `.ipynb` records that `farmer_advisory_pipeline.pkl` was generated and
downloaded, but the binary model file itself is not embedded in the notebook.

So the exact old model bytes cannot be extracted from the `.ipynb`.

This repo therefore contains a one-command reproduction path:

```powershell
python scripts\train_agriculture_from_notebook.py
```

This creates:

```text
models/agriculture/agriculture.pkl
models/agriculture/metadata.json
```

## Live endpoint

After training:

```http
POST /api/predict/agriculture/live
Content-Type: application/json
```

By place:

```json
{
  "place_name": "Guntur, Andhra Pradesh",
  "crop": "Rice",
  "crop_stage": "Flowering"
}
```

Or by coordinates:

```json
{
  "lat": 16.3067,
  "lon": 80.4365,
  "crop": "Rice",
  "crop_stage": "Flowering"
}
```

## Notebook-derived caveat

The notebook's target labels are generated from agronomic rules. The model is
therefore primarily learning those rule-derived labels from historical weather
features. Its classification metrics should not be interpreted as independent
field-validated crop-loss or agronomic outcome accuracy.
