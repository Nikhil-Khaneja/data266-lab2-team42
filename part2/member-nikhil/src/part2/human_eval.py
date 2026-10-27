import json
import os
import random

SEED = 266


def build_blind_pairs(generated_root, base_model, final_model, metadata_path, output_dir, num_pairs=25):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = [m for m in json.load(f) if m["model"] in (base_model, final_model)]

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


def compute_win_rate(responses_path, answer_key_path, final_model):
    with open(responses_path, "r", encoding="utf-8") as f:
        responses = {r["pair_id"]: r["preferred"] for r in json.load(f)}
    with open(answer_key_path, "r", encoding="utf-8") as f:
        answer_key = {a["pair_id"]: a for a in json.load(f)}

    wins = 0
    total = 0
    for pair_id, preferred in responses.items():
        if preferred not in ("left", "right"):
            continue
        key = answer_key[pair_id]
        chosen_model = key["left_model"] if preferred == "left" else key["right_model"]
        total += 1
        if chosen_model == final_model:
            wins += 1

    return {"win_rate": wins / total if total else None, "total_rated": total}


if __name__ == "__main__":
    # run2_lora_r16 has much lower (better) KID than run1_lora_r4 with essentially the
    # same CLIP score, so it is the run selected as final (see evaluate.py).
    build_blind_pairs(
        generated_root="data/processed/part2/generated",
        base_model="base",
        final_model="run2_lora_r16",
        metadata_path="data/processed/part2/generated/generation_metadata.json",
        output_dir="eval/part2_human_eval",
    )
