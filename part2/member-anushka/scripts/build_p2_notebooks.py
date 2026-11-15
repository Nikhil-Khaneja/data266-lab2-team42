"""Build the Part 2 notebooks.

    python scripts/build_p2_notebooks.py --original path/to/p2_finetune_colab_source_fixed.ipynb

* notebooks/p2_finetune.ipynb  - the executed Colab training notebook, reordered and trimmed to data
  preparation, training and generation (original outputs kept), with the required run-info cell first.
* notebooks/p2_evaluate.ipynb  - evaluation, safety, human evaluation and release (run in Colab).
"""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_cells as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Original cell indices (see docstring of the original notebook) in the order they belong.
#  0 title | 1 config | 2 install | 3 HF login | 4 remove torchao | 34 re-clone diffusers | 35 bitsandbytes
#  5-7 data prep | 8-9 prompts | 10-13 training + table | 36 peak memory | 14-15 generation | 22-23 HF upload
FINETUNE_ORDER = [0, 1, 2, 3, 4, 34, 35, 5, 6, 7, 8, 9, 10, 11, 12, 13, 36, 14, 15, 22, 23]
DROPPED = "16-21, 24-33, 37 (superseded by p2_evaluate.ipynb)"


def cell(kind, source):
    lines = source.strip("\n").splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def notebook(cells, metadata=None):
    meta = metadata or {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 0}


def write(nb, name):
    out = ROOT / "notebooks" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out.relative_to(ROOT)} ({len(nb['cells'])} cells)")


def build_finetune(original_path):
    orig = json.loads(Path(original_path).read_text())
    assert len(orig["cells"]) == 38, "unexpected original notebook"
    cells = [cell("code", C.RUN_INFO), cell("markdown", C.FINETUNE_HEADER)]
    cells += [copy.deepcopy(orig["cells"][i]) for i in FINETUNE_ORDER]
    cells.append(cell("markdown", C.FINETUNE_FOOTER))
    write(notebook(cells, orig["metadata"]), "p2_finetune.ipynb")
    print(f"  kept original cells {FINETUNE_ORDER}; dropped {DROPPED}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", help="executed Colab notebook p2_finetune_colab_source_fixed.ipynb")
    args = ap.parse_args()
    if args.original:
        build_finetune(args.original)
    write(notebook([cell(k, s) for k, s in C.EVAL]), "p2_evaluate.ipynb")


if __name__ == "__main__":
    main()
