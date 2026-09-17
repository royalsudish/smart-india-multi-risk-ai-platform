# Prompt to give Antigravity

Open this repository as a local project and use the terminal from the project root.

## Goal

Continue building a four-module multi-risk platform:

1. Weather forecasting
2. Agriculture
3. Waterlogging
4. Disaster management

## Rules

- Do not replace the uploaded trained models with dummy models.
- Do not change model feature order.
- Read `docs/MODEL_INSPECTION.md` before editing prediction code.
- API secrets or credentials must go in `.env`, never source code.
- Keep the public Nominatim client throttled and preserve its identifying User-Agent.
- Treat Bhuvan as a WMS/geospatial map service.
- Treat MOSDAC mdapi as the official credentialed downloader utility, not as a fake REST JSON endpoint.
- Do not silently fill missing disaster hazard signals with zero.
- Before changing a feature-engineering formula for WeatherGPT, compare it with the original training notebook if available.

## First commands

PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

If `py -3.13` is unavailable, use Python 3.12+.

Open:

```text
http://127.0.0.1:5000
```

Check model files without loading:

```text
GET /api/models/status
```

Try to load models:

```text
GET /api/models/status?load=1
```

## Work to finish

1. Test the uploaded WeatherGPT model.
2. Compare `backend/services/weather_features.py` with the original WeatherGPT training notebook.
3. Test the disaster model.
4. Add the real source for:
   - river_discharge
   - earthquake magnitude/depth
   - tsunami flag
   - GDACS cyclone
   - GDACS volcano
5. Recreate the exact engineered disaster features from the original training code:
   - wind_pressure_shear
   - precipitation_saturation_rate
   - thermal_particulate_index
6. When the agriculture model is added, inspect its feature order and implement its live feature builder.
7. When the real waterlogging model is added, combine:
   - rainfall totals
   - elevation
   - slope
   - Bhuvan/other land-use feature
   - distance to drain
   - drainage density
8. Build a map dashboard after the model/API pipeline is verified.


## Waterlogging model generation

The uploaded notebook has been converted into a training-only notebook and
connected to the Flask service.

Run:

```powershell
python scripts\train_waterlogging_from_notebook.py
```

Wait until it creates:

```text
models/waterlogging/waterlogging.pkl
```

Then test:

```text
POST /api/predict/waterlogging/live
```

Do not replace the notebook-derived model with a dummy estimator.
Preserve the 30-feature order in `models/waterlogging/metadata.json`.
The warning threshold from the uploaded notebook is 0.36.


## Agriculture model generation

The uploaded agriculture notebook has been analyzed and connected.

Run:

```powershell
python scripts\train_agriculture_from_notebook.py
```

It must create:

```text
models/agriculture/agriculture.pkl
```

Then test:

```text
POST /api/predict/agriculture/live
```

Do not replace the notebook-derived LightGBM bundle with a dummy model.
Preserve the exact feature order in `models/agriculture/metadata.json`.
