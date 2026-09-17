from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT
    / "notebooks"
    / "agriculture_train_only.ipynb"
)
EXECUTED = (
    ROOT
    / "notebooks"
    / "agriculture_train_only.executed.ipynb"
)
MODEL = (
    ROOT
    / "models"
    / "agriculture"
    / "agriculture.pkl"
)


def main():
    print("=" * 72)
    print("AGRICULTURE MODEL TRAINING")
    print("=" * 72)
    print("Notebook :", NOTEBOOK)
    print("Repo root:", ROOT)
    print()
    print(
        "This workflow downloads NASA POWER daily data "
        "for 28 Indian anchor locations for 2021-2024."
    )
    print("Internet access is required.")
    print()

    nb = nbformat.read(NOTEBOOK, as_version=4)

    executor = ExecutePreprocessor(
        timeout=None,
        kernel_name="python3",
        allow_errors=False,
    )

    executor.preprocess(
        nb,
        {
            "metadata": {
                "path": str(ROOT)
            }
        },
    )

    nbformat.write(nb, EXECUTED)

    if not MODEL.exists():
        raise RuntimeError(
            "Training completed but agriculture.pkl "
            "was not created."
        )

    print()
    print("=" * 72)
    print("SUCCESS")
    print("=" * 72)
    print("Model:", MODEL)
    print("Executed notebook:", EXECUTED)
    print()
    print("Restart Flask:")
    print("python app.py")


if __name__ == "__main__":
    main()
