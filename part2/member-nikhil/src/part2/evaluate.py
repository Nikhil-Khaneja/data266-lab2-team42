import json
import os

import lpips
import open_clip
import torch
from PIL import Image
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_lpips_model = None


def get_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        _clip_model = _clip_model.to(DEVICE).eval()
        _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return _clip_model, _clip_preprocess, _clip_tokenizer


def get_lpips():
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex").to(DEVICE)
    return _lpips_model


def load_image(path):
    return Image.open(path).convert("RGB")


def to_uint8_tensor(image, size=299):
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.PILToTensor()])
    return transform(image)


def compute_kid(generated_dir, test_dir, subset_size=50):
    kid = KernelInceptionDistance(subset_size=subset_size, normalize=False).to(DEVICE)

    gen_files = [f for f in os.listdir(generated_dir) if f.endswith(".png")]
    test_files = [f for f in os.listdir(test_dir) if f.endswith(".jpg")]

    gen_tensors = torch.stack([to_uint8_tensor(load_image(os.path.join(generated_dir, f))) for f in gen_files]).to(DEVICE)
    test_tensors = torch.stack([to_uint8_tensor(load_image(os.path.join(test_dir, f))) for f in test_files]).to(DEVICE)

    kid.update(test_tensors, real=True)
    kid.update(gen_tensors, real=False)
    mean, std = kid.compute()
    return float(mean), float(std)


def clip_score(image, prompt):
    model, preprocess, tokenizer = get_clip()
    image_input = preprocess(image).unsqueeze(0).to(DEVICE)
    text_input = tokenizer([prompt]).to(DEVICE)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return float((image_features @ text_features.T).item())


def clip_embedding(image):
    model, preprocess, _ = get_clip()
    image_input = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = model.encode_image(image_input)
        features /= features.norm(dim=-1, keepdim=True)
    return features


def lpips_distance(image_a, image_b, size=256):
    transform = transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    tensor_a = transform(image_a).unsqueeze(0).to(DEVICE)
    tensor_b = transform(image_b).unsqueeze(0).to(DEVICE)
    model = get_lpips()
    with torch.no_grad():
        return float(model(tensor_a, tensor_b).item())


def evaluate_model(model_name, generated_dir, metadata, test_dir):
    model_records = [r for r in metadata if r["model"] == model_name]

    clip_scores = []
    for record in model_records:
        image = load_image(os.path.join(generated_dir, record["file_name"]))
        clip_scores.append(clip_score(image, record["prompt"]))

    by_prompt = {}
    for record in model_records:
        by_prompt.setdefault(record["prompt"], []).append(record)

    lpips_distances = []
    for prompt, records in by_prompt.items():
        if len(records) < 2:
            continue
        image_a = load_image(os.path.join(generated_dir, records[0]["file_name"]))
        image_b = load_image(os.path.join(generated_dir, records[1]["file_name"]))
        lpips_distances.append(lpips_distance(image_a, image_b))

    kid_mean, kid_std = compute_kid(generated_dir, test_dir)

    return {
        "model": model_name,
        "kid_mean": kid_mean,
        "kid_std": kid_std,
        "clip_score_mean": sum(clip_scores) / len(clip_scores) if clip_scores else None,
        "lpips_diversity_mean": sum(lpips_distances) / len(lpips_distances) if lpips_distances else None,
    }


def memorization_check(generated_dir, train_dir, in_domain_files, top_k=10, show_top=3):
    train_files = [f for f in os.listdir(train_dir) if f.endswith(".jpg")]
    train_embeddings = [(f, clip_embedding(load_image(os.path.join(train_dir, f)))) for f in train_files]

    pairs = []
    for gen_file in in_domain_files[:top_k]:
        gen_embedding = clip_embedding(load_image(os.path.join(generated_dir, gen_file)))
        best_file, best_sim = None, -1
        for train_file, train_embedding in train_embeddings:
            sim = float((gen_embedding @ train_embedding.T).item())
            if sim > best_sim:
                best_sim = sim
                best_file = train_file
        pairs.append({"generated": gen_file, "closest_train": best_file, "similarity": best_sim})

    pairs.sort(key=lambda p: p["similarity"], reverse=True)
    return pairs[:show_top]


if __name__ == "__main__":
    with open("data/processed/part2/generated/generation_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    results = []
    for model_name in ["base", "run1_lora_r4", "run2_lora_r16"]:
        generated_dir = os.path.join("data/processed/part2/generated", model_name)
        result = evaluate_model(model_name, generated_dir, metadata, "data/processed/part2/test")
        results.append(result)

    with open("eval/part2_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
