"""Zip the project (code, data, notebook) as part3_colab.zip for upload to Colab/Kaggle."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = ["p3", "eval", "data", "tools", "notebooks", "human_eval", "requirements.txt", "requirements-colab.txt", "README.md"]
SKIP = {"__pycache__", ".ipynb_checkpoints"}

out = ROOT / "part3_colab.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for item in INCLUDE:
        p = ROOT / item
        if not p.exists():
            continue
        files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file() and not SKIP & set(f.parts)]
        for f in files:
            z.write(f, Path("part3") / f.relative_to(ROOT))
print("wrote", out, f"{out.stat().st_size/1e6:.1f} MB")
