# Smart Multi-Risk Platform — Antigravity-ready

This repo now contains the APIs you supplied plus the uploaded ML artifacts.

## What is already integrated

### Uploaded models/artifacts

- `models/weather/WeatherGPT_39_Feature_Model.pickle` — actual WeatherGPT bundle
- `models/disaster/disaster_prediction_pipeline.pkl` — actual disaster classifier bundle
- `data/waterlogging/final_test_predictions.pkl` — waterlogging prediction DataFrame

Still missing:

- `models/agriculture/agriculture.pkl`
- `models/waterlogging/waterlogging.pkl` (actual estimator)

`final_test_predictions.pkl` is historical/test output, not a trained model.

## API sources included

```text
Nominatim:
https://nominatim.openstreetmap.org/search

Overpass:
https://overpass-api.de/api/interpreter
https://overpass.kumi.systems/api/interpreter

Bhuvan WMS:
https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms

MOSDAC mdapi package:
https://www.mosdac.gov.in/software/mdapi.zip

Open-Meteo:
https://api.open-meteo.com/v1/forecast
https://archive-api.open-meteo.com/v1/archive
https://air-quality-api.open-meteo.com/v1/air-quality
https://api.open-meteo.com/v1/elevation
https://marine-api.open-meteo.com/v1/marine
```

## Important Python version

The uploaded XGBoost artifacts contain serialization version `3.4.1`.
The repo therefore pins:

```text
xgboost==3.4.1
```

Use Python 3.12+ for this environment. The disaster artifact also stores:

```text
scikit-learn==1.6.1
```

## Windows setup

```powershell
cd smart-multi-risk-platform-antigravity

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

Copy-Item .env.example .env

python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Useful endpoints

```text
GET  /api/health
GET  /api/models/status
GET  /api/models/status?load=1
POST /api/models/reload

GET  /api/sources/geocode?q=Guntur
GET  /api/sources/weather?lat=16.3067&lon=80.4365
GET  /api/sources/weather-features?lat=16.3067&lon=80.4365
GET  /api/sources/air-quality?lat=16.3067&lon=80.4365
GET  /api/sources/elevation?lat=16.3067&lon=80.4365
GET  /api/sources/terrain?lat=16.3067&lon=80.4365
GET  /api/sources/marine?lat=16.3067&lon=80.4365
GET  /api/sources/drainage?lat=16.3067&lon=80.4365&radius_m=2000
GET  /api/sources/bhuvan/layers
GET  /api/sources/mosdac/info

POST /api/predict/weather
POST /api/predict/weather/live
POST /api/predict/disaster
POST /api/predict/disaster/live
POST /api/predict/agriculture
POST /api/predict/waterlogging

GET /api/waterlogging/history?state=Andhra%20Pradesh&district=Guntur
```

## Live WeatherGPT call

```json
POST /api/predict/weather/live

{
  "lat": 16.3067,
  "lon": 80.4365
}
```

The project contains a reconstructed 39-feature live builder. The feature names,
lags and saved preprocessing metadata came from the uploaded model, but formulas
such as wind vectors / wet bulb / pressure tendency must be compared with the
original model-training notebook before scientific or operational use.

## Disaster model

The uploaded disaster classifier needs 20 features. The Open-Meteo connectors
can provide weather, AQI and elevation, but the model also needs:

```text
river_discharge
max_quake_mag
quake_depth
tsunami_flag
gdacs_cyclone
gdacs_volcano
wind_pressure_shear
precipitation_saturation_rate
thermal_particulate_index
```

These are intentionally reported as missing instead of being silently set to 0.

See:

```text
docs/MODEL_INSPECTION.md
ANTIGRAVITY_PROMPT.md
```

## Antigravity

Extract this repository, open the folder as a local Antigravity project, then
paste `ANTIGRAVITY_PROMPT.md` into the agent chat.

That gives the agent the model locations, API architecture, safety rules and
next implementation tasks.


## Waterlogging notebook is now wired into the repo

The uploaded `.ipynb` was analyzed and its exact 30-feature waterlogging
pipeline is now represented in the Flask app.

The old Colab execution created a trained `final_waterlogging_model.pkl`, but
that binary model object is not stored inside the `.ipynb` file itself.

To regenerate the trained model in Antigravity/VS Code:

```powershell
pip install -r requirements.txt
python scripts\train_waterlogging_from_notebook.py
```

The command creates:

```text
models/waterlogging/waterlogging.pkl
models/waterlogging/waterlogging_alert_threshold.joblib
```

Then run:

```powershell
python app.py
```

Test live waterlogging inference:

```text
POST http://127.0.0.1:5000/api/predict/waterlogging/live
```

Body:

```json
{
  "lat": 16.3067,
  "lon": 80.4365,
  "state": "Andhra Pradesh",
  "district": "Guntur"
}
```

See `docs/WATERLOGGING_NOTEBOOK_INTEGRATION.md`.


## Agriculture notebook is now integrated

The newly uploaded notebook is the farmer/advisory agriculture model.

It trains one bundle containing five LightGBM classifiers:

```text
crop_weather_risk
irrigation
heat_stress
waterlogging_risk
spraying_suitability
```

The final production contract uses 35 input features.

To generate the actual agriculture model:

```powershell
python scripts\train_agriculture_from_notebook.py
```

This writes:

```text
models/agriculture/agriculture.pkl
```

Then run:

```powershell
python app.py
```

Live agriculture prediction:

```text
POST /api/predict/agriculture/live
```

Example:

```json
{
  "place_name": "Guntur, Andhra Pradesh",
  "crop": "Rice",
  "crop_stage": "Flowering"
}
```

See `docs/AGRICULTURE_NOTEBOOK_INTEGRATION.md`.
