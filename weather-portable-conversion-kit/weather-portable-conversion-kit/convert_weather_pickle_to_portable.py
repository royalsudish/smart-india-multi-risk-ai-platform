from pathlib import Path
import json
import pickle
import re
import shutil
import zipfile

SOURCE = Path("WeatherGPT_39_Feature_Model.pickle")
OUT = Path("weather_portable")
MODELS_DIR = OUT / "models"

OUT.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))

def json_safe(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)

with SOURCE.open("rb") as f:
    bundle = pickle.load(f)

if not isinstance(bundle, dict):
    raise TypeError(f"Expected dict bundle, got {type(bundle)}")

models = bundle.get("models")
if not isinstance(models, dict) or not models:
    raise ValueError("bundle['models'] missing or empty")

manifest = {
    "format_version": 1,
    "bundle_type": "WeatherGPT portable XGBoost bundle",
    "feature_columns": json_safe(bundle.get("feature_columns", [])),
    "target_columns": json_safe(bundle.get("target_columns", [])),
    "target_transforms": json_safe(bundle.get("target_transforms", {})),
    "model_keys": json_safe(bundle.get("model_keys", list(models.keys()))),
    "main_forecast_mode": json_safe(bundle.get("main_forecast_mode")),
    "preprocessor_config": json_safe(bundle.get("preprocessor_config", {})),
    "models": {},
}

for key, model in models.items():
    filename = safe_name(key) + ".json"
    out_path = MODELS_DIR / filename

    if not hasattr(model, "save_model"):
        raise TypeError(
            f"Model '{key}' does not support save_model(); got {type(model)}"
        )

    model.save_model(out_path)
    manifest["models"][str(key)] = {
        "file": f"models/{filename}",
        "class_name": type(model).__name__,
        "module": type(model).__module__,
    }
    print("exported:", key, "->", out_path)

(OUT / "manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8"
)

zip_path = Path("weather_portable.zip")
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(OUT.parent))

print()
print("DONE")
print("Portable folder:", OUT.resolve())
print("ZIP:", zip_path.resolve())
