import json
import os
import subprocess
import sys
import time

from safetensors import safe_open

SEED = 266
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "diffusers_scripts", "train_text_to_image_lora.py")

RUN_CONFIGS = {
    "run1_lora_r4": {"rank": 4, "output_dir": "models/part2/run1_lora_r4"},
    "run2_lora_r16": {"rank": 16, "output_dir": "models/part2/run2_lora_r16"},
}

SHARED_ARGS = {
    "train_data_dir": "data/processed/part2/train",
    "resolution": 512,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_train_steps": int(os.environ.get("P2_MAX_STEPS", "1200")),   # override only for smoke tests
    "learning_rate": 1e-4,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 0,
    "seed": SEED,
    "checkpointing_steps": 600,
    "mixed_precision": "fp16",
}

# store_true flags: pass the flag itself, no value. gradient_checkpointing keeps peak
# memory under the lab's 16GB Kaggle/Colab T4 budget (plain run peaked at ~19.5GB).
SHARED_FLAGS = ["gradient_checkpointing"]


def count_trainable_params(lora_weights_path):
    total = 0
    with safe_open(lora_weights_path, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            total += tensor.numel()
    return total


def torch_device_name():
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def run_training(run_name, reuse=True):
    """Trains one run and returns its report. With reuse=True, a run already trained here with the
    same settings (weights + report on disk) is not trained again: its saved report is returned,
    marked reused=True. On a fresh machine there is nothing to reuse, so it always trains."""
    config = RUN_CONFIGS[run_name]
    output_dir = config["output_dir"]
    report_path = os.path.join(output_dir, "run_report.json")
    weights = os.path.join(output_dir, "pytorch_lora_weights.safetensors")
    if reuse and os.path.exists(report_path) and os.path.exists(weights):
        with open(report_path, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("max_train_steps") == SHARED_ARGS["max_train_steps"] and saved.get("rank") == config["rank"]:
            saved["reused"] = True
            saved["trained_at"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(weights)))
            return saved
    os.makedirs(output_dir, exist_ok=True)

    # sys.executable: the same interpreter as the notebook (a bare "python" can resolve to another install).
    args = [sys.executable, SCRIPT_PATH, "--pretrained_model_name_or_path", BASE_MODEL,
             "--rank", str(config["rank"]), "--output_dir", output_dir]
    for key, value in SHARED_ARGS.items():
        args.append("--{}".format(key))
        args.append(str(value))
    for flag in SHARED_FLAGS:
        args.append("--{}".format(flag))

    # The script's step-by-step log goes to a file; the notebook shows only the report.
    log_path = os.path.join(output_dir, "train.log")
    start_time = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT)
    training_time_seconds = time.time() - start_time
    if result.returncode != 0:
        with open(log_path, encoding="utf-8", errors="replace") as log:
            raise RuntimeError("training failed:" + chr(10) + log.read()[-3000:])

    with open(os.path.join(output_dir, "peak_memory.json"), encoding="utf-8") as f:
        peak = json.load(f)

    lora_weights_path = os.path.join(output_dir, "pytorch_lora_weights.safetensors")
    trainable_params = count_trainable_params(lora_weights_path) if os.path.exists(lora_weights_path) else None

    report = {
        "run_name": run_name,
        "rank": config["rank"],
        "trainable_params": trainable_params,
        "learning_rate": SHARED_ARGS["learning_rate"],
        "max_train_steps": SHARED_ARGS["max_train_steps"],
        "effective_batch_size": SHARED_ARGS["train_batch_size"] * SHARED_ARGS["gradient_accumulation_steps"],
        "peak_gpu_memory_gb": round(peak["max_memory_reserved_mb"] / 1024, 2),
        "training_time_seconds": round(training_time_seconds, 1),
        "gpu_hours": round(training_time_seconds / 3600, 3),
        "gpu": torch_device_name(),
        "reused": False,
    }
    with open(os.path.join(output_dir, "run_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    import sys
    report = run_training(sys.argv[1])
    print(json.dumps(report, indent=2))
