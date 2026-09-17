from pathlib import Path
import pickle
import json

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["weather", "agriculture", "disaster", "waterlogging"]

for name in MODELS:
    folder = ROOT / "models" / name
    candidates = list(folder.glob("*.pkl")) + list(folder.glob("*.pickle"))

    print("\n" + "=" * 70)
    print(name.upper())

    if not candidates:
        print("No .pkl file found.")
        continue

    path = candidates[0]
    print("File:", path.name)

    try:
        with open(path, "rb") as f:
            model = pickle.load(f)

        print("Python type:", type(model))
        print("Has predict:", hasattr(model, "predict"))
        print("Has predict_proba:", hasattr(model, "predict_proba"))
        print("n_features_in_:", getattr(model, "n_features_in_", None))
        print("feature_names_in_:", getattr(model, "feature_names_in_", None))
        print("classes_:", getattr(model, "classes_", None))

        meta_path = folder / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print("metadata features:", meta.get("features"))
    except Exception as exc:
        print("ERROR:", exc)
