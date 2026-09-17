import nbformat

def patch_waterlogging_notebook():
    path = "notebooks/waterlogging_train_only.ipynb"
    nb = nbformat.read(path, as_version=4)
    
    # Patch downsampling in cell 27
    cell27 = nb.cells[27]
    lines27 = cell27.source.splitlines()
    start_idx = None
    end_idx = None
    for idx, l in enumerate(lines27):
        if "if HISTORY_SAMPLE_EVERY_HOURS > 1:" in l:
            start_idx = idx
        if start_idx is not None and "# 8. QUALITY CHECK" in l:
            end_idx = idx
            break
            
    if start_idx is not None and end_idx is not None:
        replacement = [
            "if HISTORY_SAMPLE_EVERY_HOURS > 1:",
            "",
            "    print(",
            "        f\"\\nDownsampling to every \"",
            "        f\"{HISTORY_SAMPLE_EVERY_HOURS} hours...\"",
            "    )",
            "",
            "    history = (",
            "        history[",
            "            history.groupby(['state', 'district']).cumcount()",
            "            % HISTORY_SAMPLE_EVERY_HOURS == 0",
            "        ]",
            "        .reset_index(drop=True)",
            "    )",
            "",
            "",
            "# ============================================================",
        ]
        cell27.source = "\n".join(lines27[:start_idx] + replacement + lines27[end_idx:])

    # Ensure Cell 65 preserves the 0.36 threshold from the original tuned notebook
    cell65 = nb.cells[65]
    if "FINAL_WARNING_THRESHOLD = 0.36" not in cell65.source:
        cell65.source = cell65.source.replace(
            "PKL_MODEL_PATH = MODEL_OUTPUT_DIR / \"waterlogging.pkl\"",
            "PKL_MODEL_PATH = MODEL_OUTPUT_DIR / \"waterlogging.pkl\"\nFINAL_WARNING_THRESHOLD = 0.36"
        )

    nbformat.write(nb, path)
    print("Cleanly patched waterlogging notebook!")

if __name__ == "__main__":
    patch_waterlogging_notebook()
