import json
import os
import random

SEED = 266


def build_blind_pairs(generated_root, base_model, final_model, metadata_path, output_dir, num_pairs=25):
    with open(metadata_path, "r", encoding="utf-8") as f:
        # Pairs only from images the safety checker passed: a blacked-out image cannot be rated.
        metadata = [m for m in json.load(f) if m["model"] in (base_model, final_model) and not m.get("nsfw")]

    by_key = {}
    for record in metadata:
        key = (record["prompt"], record["seed"])
        by_key.setdefault(key, {})[record["model"]] = record["file_name"]

    complete_pairs = [(key, files) for key, files in by_key.items()
                       if base_model in files and final_model in files]

    rng = random.Random(SEED)
    rng.shuffle(complete_pairs)
    selected = complete_pairs[:num_pairs]

    os.makedirs(output_dir, exist_ok=True)
    blind_items = []
    answer_key = []
    for i, (key, files) in enumerate(selected):
        prompt, seed = key
        pair_id = "pair_{:02d}".format(i)
        flip = rng.random() < 0.5
        left_model = final_model if flip else base_model
        right_model = base_model if flip else final_model

        blind_items.append({
            "pair_id": pair_id,
            "prompt": prompt,
            "left_image": os.path.join(generated_root, left_model, files[left_model]),
            "right_image": os.path.join(generated_root, right_model, files[right_model]),
            "preferred": None,  # rater fills in "left" or "right"
        })
        answer_key.append({"pair_id": pair_id, "left_model": left_model, "right_model": right_model})

    with open(os.path.join(output_dir, "blind_pairs.json"), "w", encoding="utf-8") as f:
        json.dump(blind_items, f, indent=2)
    with open(os.path.join(output_dir, "answer_key.json"), "w", encoding="utf-8") as f:
        json.dump(answer_key, f, indent=2)
    return blind_items


CRITERIA = ("style", "prompt")   # domain style; prompt match
HF_IMAGE_DATASET = "nikhilkhaneja/sd15-naruto-lora-generated-images"


def _hub_path(local_path):
    # blind_pairs.json stores "<generated_root>\<model>\<file>"; the Hub dataset is "<model>/<file>".
    parts = local_path.replace("\\", "/").split("/")
    return "/".join(parts[-2:])


def prepare_rater_sheet(blind_pairs_path, output_path, hf_repo=HF_IMAGE_DATASET):
    """One self-contained HTML page for raters: each pair's prompt and its two images under
    neutral L/R labels, with no file names or model names, so the page itself cannot unblind
    the rater. Nothing is regenerated: images are read from disk or the pushed Hub dataset."""
    import base64
    import io
    from huggingface_hub import hf_hub_download
    from PIL import Image

    with open(blind_pairs_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    def encoded(local_path):
        # The local file when this machine generated it; otherwise the pushed Hub copy.
        file = local_path if os.path.exists(local_path) else hf_hub_download(hf_repo, _hub_path(local_path), repo_type="dataset")
        buffer = io.BytesIO()
        Image.open(file).convert("RGB").save(buffer, format="JPEG", quality=92)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    items = [{"pair_id": p["pair_id"], "prompt": p["prompt"],
              "L": encoded(p["left_image"]), "R": encoded(p["right_image"])} for p in pairs]
    with open(os.path.join(os.path.dirname(__file__), "rater_sheet_template.html"), "r", encoding="utf-8") as f:
        page = f.read().replace("__PAIRS__", json.dumps(items))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(page)
    return output_path


def compute_win_rates(ratings_paths, answer_key_path, final_model):
    """Final-model win rate per criterion: over all votes, per rater, and by majority per pair.

    Each ratings file is what the rater page downloads:
    {"rater": name, "ratings": [{"pair_id", "style": "L"|"R", "prompt": "L"|"R", "note"}]}."""
    with open(answer_key_path, "r", encoding="utf-8") as f:
        key = {a["pair_id"]: a for a in json.load(f)}
    raters = []
    for path in ratings_paths:
        with open(path, "r", encoding="utf-8") as f:
            raters.append(json.load(f))

    def chose_final(pair_id, side):
        return key[pair_id]["left_model" if side == "L" else "right_model"] == final_model

    report = {"raters": [r["rater"] for r in raters]}
    for criterion in CRITERIA:
        votes = {}   # pair_id -> list of True (final chosen) / False
        per_rater = {}
        for r in raters:
            mine = [chose_final(x["pair_id"], x[criterion]) for x in r["ratings"] if x.get(criterion) in ("L", "R")]
            per_rater[r["rater"]] = sum(mine) / len(mine) if mine else None
            for x in r["ratings"]:
                if x.get(criterion) in ("L", "R"):
                    votes.setdefault(x["pair_id"], []).append(chose_final(x["pair_id"], x[criterion]))
        all_votes = [v for vs in votes.values() for v in vs]
        majority = [sum(vs) > len(vs) / 2 for vs in votes.values() if len(vs) == len(raters)]
        unanimous = [len(set(vs)) == 1 for vs in votes.values() if len(vs) == len(raters)]
        report[criterion] = {
            "win_rate_all_votes": sum(all_votes) / len(all_votes) if all_votes else None,
            "votes": len(all_votes),
            "win_rate_by_rater": per_rater,
            "majority_win_rate": sum(majority) / len(majority) if majority else None,
            "pairs_with_all_raters": len(majority),
            "unanimous_share": sum(unanimous) / len(unanimous) if unanimous else None,
        }
    return report


if __name__ == "__main__":
    # run2_lora_r16 has much lower (better) KID than run1_lora_r4 with essentially the
    # same CLIP score, so it is the run selected as final (see evaluate.py).
    import glob
    import sys

    step = sys.argv[1] if len(sys.argv) > 1 else "sheet"
    if step == "pairs":    # already done once; rerunning reshuffles nothing (fixed seed)
        build_blind_pairs(
            generated_root="data/processed/part2/generated",
            base_model="base",
            final_model="run2_lora_r16",
            metadata_path="data/processed/part2/generated/generation_metadata.json",
            output_dir="eval/part2_human_eval",
        )
    elif step == "sheet":  # page to send to all three raters
        print(prepare_rater_sheet("eval/part2_human_eval/blind_pairs.json",
                                  "data/processed/part2/part2_rater_sheet.html"))
    elif step == "score":  # after every rater has returned their file
        print(json.dumps(compute_win_rates(sorted(glob.glob("eval/part2_human_eval/ratings/*.json")),
                                           "eval/part2_human_eval/answer_key.json", "run2_lora_r16"), indent=2))
