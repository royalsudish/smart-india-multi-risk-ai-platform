# Weather model

Put your trained pickle model here:

```text
models/weather/weather.pkl
```

Then edit `metadata.json` and put the **exact training feature order** in `features`.

Example:

```json
{
  "artifact": "weather.pkl",
  "task": "classification",
  "features": ["feature_1", "feature_2", "feature_3"]
}
```

If your saved object is a scikit-learn `Pipeline`, preprocessing can stay inside the `.pkl`.
If scaler/encoder files were saved separately, integrate them into a Pipeline or load them explicitly in `backend/services/model_service.py`.
