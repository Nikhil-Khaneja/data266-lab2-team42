"""Write the current git commit hash to ./COMMIT before copying this folder to Google Drive.

The Drive copy has no .git folder, so the notebook's first cell reads COMMIT instead.
Run from the repo clone right before uploading:  python tools/stamp_commit.py
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                        check=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=ROOT, capture_output=True,
                       text=True).stdout.strip()
if dirty:
    raise SystemExit("Uncommitted changes in this folder - commit them first so COMMIT matches the code:\n" + dirty)
(ROOT / "COMMIT").write_text(commit + "\n")
print("COMMIT =", commit)
