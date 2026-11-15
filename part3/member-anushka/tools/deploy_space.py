"""Deploy the Gradio demo to a Hugging Face Space.

Log in first with a write token (run it yourself; the token is cached by huggingface_hub):
    hf auth login
Then, from this folder:
    python tools/deploy_space.py                  # public Space <your-user>/data266-lab2-part3-agents
    python tools/deploy_space.py --private
    python tools/deploy_space.py --dry-run        # build the folder only, no upload
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="data266-lab2-part3-agents", help="Space name (under your account)")
ap.add_argument("--private", action="store_true")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--out", help="with --dry-run: where to build the folder")
args = ap.parse_args()

if not (ROOT / "data" / "images" / "cats.jpg").exists():
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_assets.py")], check=True)

stage = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="p3_space_"))
if stage.exists():
    shutil.rmtree(stage)
stage.mkdir(parents=True)
for f in ("app.py", "requirements.txt", "README.md"):
    shutil.copy(ROOT / "space" / f, stage / f)
shutil.copytree(ROOT / "p3", stage / "p3", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
shutil.copytree(ROOT / "data" / "images", stage / "data" / "images")
print("built", stage, sorted(p.name for p in stage.iterdir()))
if args.dry_run:
    sys.exit(0)

from huggingface_hub import HfApi  # noqa: E402

api = HfApi()
user = api.whoami()["name"]
repo_id = f"{user}/{args.name}"
api.create_repo(repo_id, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)
api.upload_folder(folder_path=str(stage), repo_id=repo_id, repo_type="space",
                  commit_message="Deploy Part 3 multi-agent demo")
print(f"deployed: https://huggingface.co/spaces/{repo_id}")
