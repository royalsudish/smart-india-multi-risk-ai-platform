from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRED_FILE = ROOT / "data" / "waterlogging" / "final_test_predictions.pkl"

_df = None

def _load():
    global _df
    if _df is None:
        if not PRED_FILE.exists():
            raise FileNotFoundError(str(PRED_FILE))
        _df = pd.read_pickle(PRED_FILE)
    return _df

def query(state=None, district=None, limit=200):
    df = _load().copy()
    if state and "state" in df.columns:
        df = df[df["state"].astype(str).str.lower() == state.lower()]
    if district and "district" in df.columns:
        df = df[df["district"].astype(str).str.lower() == district.lower()]
    if "datetime" in df.columns:
        df = df.sort_values("datetime", ascending=False)
    df = df.head(max(1, min(int(limit), 1000)))
    return {
        "columns": list(df.columns),
        "rows": df.astype(object).where(df.notna(), None).to_dict(orient="records"),
        "count": len(df)
    }
