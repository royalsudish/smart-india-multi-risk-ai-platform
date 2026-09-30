from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import math
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "models"
MODEL_NAMES = ["weather", "agriculture", "waterlogging", "disaster", "drainage"]

def _python(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v

class ModelSlot:
    def __init__(self, name):
        self.name = name
        self.folder = MODEL_ROOT / name
        self.metadata = self._metadata()
        self.obj = None
        self.error = None
        self._loaded = False

    def _metadata(self):
        p = self.folder / "metadata.json"
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    @property
    def path(self):
        artifact = self.metadata.get("artifact")
        return self.folder / artifact if artifact else None

    @property
    def is_loaded(self):
        return self.obj is not None

    def ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True

        # Check for portable disaster JSON bundle first or fallback if pickle fails
        def _load_disaster_portable():
            portable_json = self.folder / "portable" / "disaster_xgb.json"
            if not portable_json.exists():
                portable_json = self.folder / "portable" / "disaster_portable" / "disaster_xgb.json"
            if portable_json.exists():
                try:
                    import xgboost as xgb
                    p_dir = portable_json.parent
                    model = xgb.XGBClassifier()
                    model.load_model(str(portable_json))
                    scaler = joblib.load(p_dir / "scaler.joblib") if (p_dir / "scaler.joblib").exists() else None
                    encoder = joblib.load(p_dir / "label_encoder.joblib") if (p_dir / "label_encoder.joblib").exists() else None
                    manifest_p = p_dir / "manifest.json"
                    manifest = json.loads(manifest_p.read_text(encoding="utf-8")) if manifest_p.exists() else {}
                    self.obj = {
                        "model": model,
                        "scaler": scaler,
                        "label_encoder": encoder,
                        "feature_order": manifest.get("feature_order", self.metadata.get("features", [])),
                    }
                    self.error = None
                    return True
                except Exception as p_exc:
                    self.error = f"{type(p_exc).__name__}: {p_exc}"
                    return False
            return False

        # Check for portable weather JSON bundle
        def _load_weather_portable():
            manifest_p = self.folder / "portable" / "manifest.json"
            if not manifest_p.exists():
                manifest_p = self.folder / "portable" / "weather_portable" / "manifest.json"
            if manifest_p.exists():
                try:
                    import xgboost as xgb
                    p_dir = manifest_p.parent
                    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
                    models = {}
                    for key, meta in manifest.get("models", {}).items():
                        m_path = p_dir / meta["file"]
                        reg = xgb.XGBRegressor()
                        reg.load_model(str(m_path))
                        models[key] = reg
                    self.obj = {
                        "models": models,
                        "feature_columns": manifest.get("feature_columns", self.metadata.get("features", [])),
                        "target_columns": manifest.get("target_columns", self.metadata.get("targets", [])),
                        "target_transforms": manifest.get("target_transforms", {}),
                        "model_keys": list(models.keys()),
                        "main_forecast_mode": "portable_xgboost",
                    }
                    self.error = None
                    return True
                except Exception as p_exc:
                    self.error = f"{type(p_exc).__name__}: {p_exc}"
                    return False
            return False

        if not self.path or not self.path.exists():
            if self.name == "disaster" and _load_disaster_portable():
                return
            if self.name == "weather" and _load_weather_portable():
                return
            self.error = f"Model file not found: {self.path}"
            return
        try:
            # joblib.load can read standard pickle files too and is required by
            # the uploaded disaster bundle because it contains joblib array wrappers.
            self.obj = joblib.load(self.path)
        except Exception as exc:
            if self.name == "disaster" and _load_disaster_portable():
                return
            if self.name == "weather" and _load_weather_portable():
                return
            self.error = f"{type(exc).__name__}: {exc}"

    def reload(self):
        self.obj = None
        self.error = None
        self._loaded = False
        self.metadata = self._metadata()

    def status(self):
        file_present = bool(self.path and self.path.exists())
        file_name = str(self.path.name) if self.path else None
        if self.name == "disaster" and not file_present:
            portable_p = self.folder / "portable" / "disaster_xgb.json"
            if portable_p.exists() or (self.folder / "portable" / "disaster_portable" / "disaster_xgb.json").exists():
                file_present = True
                file_name = "portable/disaster_xgb.json"
        if self.name == "weather" and (not file_present or (self.obj is not None and "portable" in getattr(self.obj, "__str__", lambda: "")())):
            manifest_p = self.folder / "portable" / "manifest.json"
            if manifest_p.exists() or (self.folder / "portable" / "weather_portable" / "manifest.json").exists():
                file_present = True

        return {
            "name": self.name,
            "file": file_name,
            "file_present": file_present,
            "loaded": self.obj is not None,
            "load_attempted": self._loaded,
            "error": self.error,
            "kind": self.metadata.get("kind"),
            "task": self.metadata.get("task"),
            "features": self.metadata.get("features", []),
        }

    def _generic_predict(self, obj, features):
        names = self.metadata.get("features") or list(getattr(obj, "feature_names_in_", []))
        if not names:
            raise ValueError("Feature order is unknown. Fill models/%s/metadata.json" % self.name)
        missing = [x for x in names if x not in features]
        if missing:
            raise ValueError("Missing features: " + ", ".join(missing))
        X = pd.DataFrame([[features[x] for x in names]], columns=names)
        pred = obj.predict(X)[0]
        out = {"prediction": _python(pred), "used_features": names}
        if hasattr(obj, "predict_proba"):
            probs = obj.predict_proba(X)[0]
            out["probabilities"] = [_python(x) for x in probs]
            classes = getattr(obj, "classes_", None)
            if classes is not None:
                out["classes"] = [_python(x) for x in classes]
            if len(probs) == 2:
                out["positive_probability"] = float(probs[1])
        return out

    def _weather_bundle_predict(self, bundle, features):
        names = bundle.get("feature_columns") or self.metadata.get("features") or []
        missing = [x for x in names if x not in features]
        if missing:
            raise ValueError("Missing WeatherGPT features: " + ", ".join(missing))

        X = np.array([[features[x] for x in names]], dtype=float)
        models = bundle.get("models") or {}
        transforms = bundle.get("target_transforms") or {}
        model_keys = bundle.get("model_keys") or list(models.keys())

        raw = {}
        for key in model_keys:
            model = models[key]
            value = float(model.predict(X)[0])
            transform = transforms.get(key) or {}
            if transform.get("name") == "log1p":
                value = float(np.expm1(value))
            raw[key] = value

        if "wind_direction_sin" in raw and "wind_direction_cos" in raw:
            deg = math.degrees(math.atan2(
                raw["wind_direction_sin"], raw["wind_direction_cos"]
            )) % 360.0
            raw["wind_direction_10m"] = deg

        target_cols = bundle.get("target_columns") or self.metadata.get("targets") or list(raw.keys())
        prediction = {k: raw[k] for k in target_cols if k in raw}
        return {
            "prediction": prediction,
            "used_features": names,
            "model_count": len(models),
            "bundle_mode": bundle.get("main_forecast_mode", "unknown"),
        }


    def _agriculture_bundle_predict(self, bundle, features):
        import pandas as pd

        models = bundle.get("models") or {}
        target_encoders = bundle.get("target_encoders") or {}
        cat_encoders = bundle.get("cat_encoders") or {}
        all_features = (
            bundle.get("all_features")
            or self.metadata.get("features")
            or []
        )

        payload = dict(features)

        for col in ["crop", "crop_stage", "agro_season"]:
            if col not in payload:
                raise ValueError(
                    f"Missing agriculture categorical input: {col}"
                )

            encoder = cat_encoders.get(col)
            if encoder is None:
                raise ValueError(
                    f"Missing saved encoder for agriculture column: {col}"
                )

            val = str(payload[col])

            # Source-faithful behavior from the uploaded notebook:
            # unknown category -> encoded value 0.
            if val in encoder.classes_:
                payload[col + "_enc"] = int(
                    encoder.transform([val])[0]
                )
            else:
                payload[col + "_enc"] = 0

        missing = [
            name for name in all_features
            if name not in payload
        ]
        if missing:
            raise ValueError(
                "Missing agriculture features: "
                + ", ".join(missing)
            )

        X = pd.DataFrame(
            [[payload[name] for name in all_features]],
            columns=all_features,
        ).astype(np.float32)

        predictions = {}
        probabilities = {}

        for target, model in models.items():
            encoder = target_encoders[target]
            pred_idx = int(model.predict(X)[0])
            label = encoder.inverse_transform([pred_idx])[0]
            predictions[target] = str(label)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                probabilities[target] = {
                    str(class_name): float(prob)
                    for class_name, prob in zip(
                        encoder.classes_,
                        probs,
                    )
                }

        return {
            "prediction": predictions,
            "probabilities": probabilities,
            "used_features": all_features,
        }

    def _disaster_bundle_predict(self, bundle, features):
        model = bundle.get("model")
        scaler = bundle.get("scaler")
        encoder = bundle.get("label_encoder")
        names = bundle.get("feature_order") or self.metadata.get("features") or []
        missing = [x for x in names if x not in features]
        if missing:
            raise ValueError("Missing disaster features: " + ", ".join(missing))

        X = pd.DataFrame([[features[x] for x in names]], columns=names)
        if scaler is not None:
            X = scaler.transform(X)

        encoded = model.predict(X)[0]
        if encoder is not None:
            label = encoder.inverse_transform([int(encoded)])[0]
        else:
            label = encoded

        out = {
            "prediction": _python(label),
            "encoded_prediction": _python(encoded),
            "used_features": names,
        }

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            classes = self.metadata.get("classes", [])
            out["probabilities"] = {
                str(classes[i] if i < len(classes) else i): float(p)
                for i, p in enumerate(probs)
            }
            best_i = int(np.argmax(probs))
            out["confidence"] = float(probs[best_i])

        return out

    def predict(self, features):
        self.ensure_loaded()
        if self.obj is None:
            raise RuntimeError(self.error or "Model did not load")

        obj = self.obj
        if isinstance(obj, dict) and "models" in obj and "feature_columns" in obj:
            return self._weather_bundle_predict(obj, features)

        if (
            isinstance(obj, dict)
            and "target_encoders" in obj
            and "cat_encoders" in obj
            and "all_features" in obj
        ):
            return self._agriculture_bundle_predict(obj, features)

        if isinstance(obj, dict) and "model" in obj and "scaler" in obj:
            return self._disaster_bundle_predict(obj, features)

        return self._generic_predict(obj, features)

    def predict_batch(self, features_df):
        self.ensure_loaded()
        if self.obj is None:
            raise RuntimeError(self.error or "Model did not load")

        obj = self.obj
        names = self.metadata.get("features") or list(getattr(obj, "feature_names_in_", []))
        if not names:
            raise ValueError(f"Feature order is unknown for model {self.name}")

        missing = [x for x in names if x not in features_df.columns]
        if missing:
            raise ValueError(f"Missing features in batch for {self.name}: " + ", ".join(missing))

        X = features_df[names]
        preds = obj.predict(X)
        out = {
            "predictions": [_python(p) for p in preds],
            "used_features": names,
            "row_count": len(X),
        }

        if hasattr(obj, "predict_proba"):
            probs = obj.predict_proba(X)
            out["positive_probabilities"] = [
                float(p[1]) if len(p) > 1 else float(p[0])
                for p in probs
            ]

        return out

class ModelRegistry:
    def __init__(self):
        self.slots = {name: ModelSlot(name) for name in MODEL_NAMES}

    def get(self, name: str) -> Optional[ModelSlot]:
        return self.slots.get(name)

    def status(self, try_load=False):
        if try_load:
            for slot in self.slots.values():
                slot.ensure_loaded()
        return {name: slot.status() for name, slot in self.slots.items()}

    def reload(self):
        for slot in self.slots.values():
            slot.reload()

    def predict(self, name, features):
        if name not in self.slots:
            raise KeyError(name)
        return self.slots[name].predict(features)

    def predict_batch(self, name, features_df):
        if name not in self.slots:
            raise KeyError(name)
        return self.slots[name].predict_batch(features_df)

registry = ModelRegistry()
