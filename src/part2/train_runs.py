import json
import os
import subprocess
import threading
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
    "max_train_steps": 1200,
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


def sample_gpu_memory(samples, stop_event):
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
            )
            samples.append(int(out.decode().strip().splitlines()[0]))
        except Exception:
            pass
        time.sleep(2)


def count_trainable_params(lora_weights_path):
    total = 0
    with safe_open(lora_weights_path, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            total += tensor.numel()
    return total


def run_training(run_name):
    config = RUN_CONFIGS[run_name]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    args = ["python", SCRIPT_PATH, "--pretrained_model_name_or_path", BASE_MODEL,
             "--rank", str(config["rank"]), "--output_dir", output_dir]
    for key, value in SHARED_ARGS.items():
        args.append("--{}".format(key))
        args.append(str(value))
    for flag in SHARED_FLAGS:
        args.append("--{}".format(flag))

    samples = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=sample_gpu_memory, args=(samples, stop_event))
    monitor_thread.start()

    start_time = time.time()
    subprocess.run(args, check=True)
    training_time_seconds = time.time() - start_time

    stop_event.set()
    monitor_thread.join()

    lora_weights_path = os.path.join(output_dir, "pytorch_lora_weights.safetensors")
    trainable_params = count_trainable_params(lora_weights_path) if os.path.exists(lora_weights_path) else None

    report = {
        "run_name": run_name,
        "rank": config["rank"],
        "trainable_params": trainable_params,
        "learning_rate": SHARED_ARGS["learning_rate"],
        "max_train_steps": SHARED_ARGS["max_train_steps"],
        "effective_batch_size": SHARED_ARGS["train_batch_size"] * SHARED_ARGS["gradient_accumulation_steps"],
        "peak_gpu_memory_mb": max(samples) if samples else None,
        "training_time_seconds": training_time_seconds,
    }
    with open(os.path.join(output_dir, "run_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    import sys
    report = run_training(sys.argv[1])
    print(json.dumps(report, indent=2))
