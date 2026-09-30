import joblib
import json
import numpy as np

pkl_path = "models/drainage/drainage_random_forest.pkl"
try:
    obj = joblib.load(pkl_path)
    print("LOAD_SUCCESS")
    print(f"Type: {type(obj)}")
    print(f"Class name: {obj.__class__.__name__}")
    print(f"Module: {obj.__class__.__module__}")
    
    # Check if dict or object
    if isinstance(obj, dict):
        print("Object is dict with keys:", list(obj.keys()))
        model = obj.get("model") or obj.get("classifier") or obj
    else:
        model = obj

    print(f"Model type: {type(model)}")
    print("Has n_features_in_:", getattr(model, "n_features_in_", None))
    print("Has feature_names_in_:", getattr(model, "feature_names_in_", None))
    if hasattr(model, "feature_names_in_"):
        print("Feature names list:", list(model.feature_names_in_))
        print("Feature count:", len(model.feature_names_in_))
    
    print("Has classes_:", getattr(model, "classes_", None))
    print("Has predict_proba:", hasattr(model, "predict_proba"))
    print("Has steps (Pipeline):", getattr(model, "steps", None))
    print("Has named_steps:", getattr(model, "named_steps", None))
    
    # Inspect attributes of model
    attrs = [a for a in dir(model) if not a.startswith("__")]
    print("Interesting attrs:", [a for a in attrs if "feature" in a or "class" in a or "scaler" in a or "step" in a or "target" in a or "estimat" in a])

    # Check estimators
    if hasattr(model, "estimators_"):
        print("Estimators count:", len(model.estimators_))
        if len(model.estimators_) > 0:
            est0 = model.estimators_[0]
            print("Estimator[0] n_features_in_:", getattr(est0, "n_features_in_", None))

    # Check if there are other objects in a dictionary or bundle
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"Dict key '{k}': type {type(v)}")

except Exception as e:
    import traceback
    print("ERROR:", e)
    traceback.print_exc()
