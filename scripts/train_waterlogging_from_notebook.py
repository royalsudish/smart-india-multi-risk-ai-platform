from pathlib import Path
import shutil
import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "waterlogging_train_only.ipynb"
EXECUTED = ROOT / "notebooks" / "waterlogging_train_only.executed.ipynb"
MODEL = ROOT / "models" / "waterlogging" / "waterlogging.pkl"


def main():
    if not NOTEBOOK.exists():
        raise FileNotFoundError(NOTEBOOK)

    print("=" * 72)
    print("WATERLOGGING MODEL TRAINING")
    print("=" * 72)
    print("Notebook :", NOTEBOOK)
    print("Repo root:", ROOT)
    print()
    print(
        "This training downloads historical/live API data. "
        "Internet access is required."
    )
    print()

    nb = nbformat.read(NOTEBOOK, as_version=4)

    ep = ExecutePreprocessor(
        timeout=None,
        kernel_name="python3",
        allow_errors=False,
    )

    ep.preprocess(
        nb,
        {"metadata": {"path": str(ROOT)}},
    )

    nbformat.write(nb, EXECUTED)

    if not MODEL.exists():
        raise RuntimeError(
            "Training completed but models/waterlogging/waterlogging.pkl "
            "was not created."
        )

    print()
    print("=" * 72)
    print("SUCCESS")
    print("=" * 72)
    print("Model:", MODEL)
    print("Executed notebook:", EXECUTED)
    print()
    print("Now restart Flask:")
    print("python app.py")


if __name__ == "__main__":
    main()
