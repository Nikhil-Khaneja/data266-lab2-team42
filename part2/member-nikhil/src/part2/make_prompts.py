import json
import os
import random

SEED = 266

OUT_OF_DOMAIN_PROMPTS = [
    "a photorealistic astronaut on the moon",
    "a watercolor painting of a quiet mountain lake",
    "a steampunk city skyline at sunset",
    "a cute cartoon robot watering plants in a garden",
    "an oil painting of a horse running through a field",
]

COMPOSITIONAL_PROMPTS = [
    "two characters, one in red and one in blue, standing back to back",
    "a character jumping over a waterfall while holding a sword",
    "three characters sitting around a campfire at night",
    "a character wearing a green headband and a character wearing a black cloak facing each other",
    "a character running through a forest chased by a shadow figure",
]


def load_test_captions(test_metadata_path):
    captions = []
    with open(test_metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            captions.append(entry["text"])
    return captions


def build_prompts(test_metadata_path, output_path):
    captions = load_test_captions(test_metadata_path)
    rng = random.Random(SEED)
    in_domain = rng.sample(captions, 10)

    prompts = []
    for text in in_domain:
        prompts.append({"prompt": text, "category": "in_domain"})
    for text in OUT_OF_DOMAIN_PROMPTS:
        prompts.append({"prompt": text, "category": "out_of_domain"})
    for text in COMPOSITIONAL_PROMPTS:
        prompts.append({"prompt": text, "category": "compositional"})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2)
    return prompts


if __name__ == "__main__":
    build_prompts("data/processed/part2/test/metadata.jsonl", "../eval/part2_prompts.json")
