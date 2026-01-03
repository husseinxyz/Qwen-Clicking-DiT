"""
Modal.com training script for Qwen-Clicking-DiT.

Usage:
    modal run modal_train.py
"""

import modal

# Define the image with all dependencies
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .apt_install("git")
    .pip_install("wheel", "setuptools", "pip", extra_options="--upgrade")
    .pip_install(
        "torch>=2.1.0",
        "torchvision",
        "transformers>=4.45.0",
        "accelerate>=0.26.0",
        "datasets",
        "pillow",
        "qwen-vl-utils",
        "peft",
        "bitsandbytes",
        "scipy",
        "sentencepiece",
        "protobuf",
        "tiktoken",
        "einops",
        "ujson",
        "scikit-learn",
        "tqdm",
        "packaging",
        "trl",
        "liger-kernel>=0.3.0",
        "wandb",
        "matplotlib",
        "huggingface_hub",
    )
    .run_commands("mkdir -p /app")
    .add_local_dir("src", remote_path="/app/src", copy=True)
    .add_local_dir("scripts", remote_path="/app/scripts", copy=True)
    .run_commands("ln -sf /app/src/train /app/train")
)

# Create Modal app
app = modal.App("qwen-clicking-dit", image=image)

# Volume to persist model checkpoints
volume = modal.Volume.from_name("qwen-clicking-dit-vol", create_if_missing=True)
VOLUME_PATH = "/vol"

# HuggingFace secret for accessing gated models
hf_secret = modal.Secret.from_name("huggingface-secret")
# W&B secret for logging
wandb_secret = modal.Secret.from_name("wandb-secret")


@app.function(
    gpu="H100",  # or "A100" or "A10G" for cheaper
    timeout=3600 * 4,  # 4 hours max
    volumes={VOLUME_PATH: volume},
    secrets=[hf_secret, wandb_secret],
)
def train(
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 1e-4,
    max_samples: int = 20000,
):
    """Run training on Modal GPU."""
    import os
    import subprocess
    import sys

    # Change to app directory
    os.chdir("/app")
    sys.path.insert(0, "/app/src")

    # Set up environment - need both /app and /app/src for correct imports
    os.environ["PYTHONPATH"] = "/app:/app/src:" + os.environ.get("PYTHONPATH", "")
    os.environ["HF_HOME"] = f"{VOLUME_PATH}/hf_cache"

    # Ensure HF token is set for dataset access
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        # Also login using huggingface-cli
        subprocess.run(["huggingface-cli", "login", "--token", hf_token], check=False)

    # Set up W&B (optional - set your own entity)
    os.environ["WANDB_PROJECT"] = "qwen-click-dit"
    # os.environ["WANDB_ENTITY"] = "your-wandb-entity"

    output_dir = f"{VOLUME_PATH}/outputs/click_dit"
    os.makedirs(output_dir, exist_ok=True)

    # Training command
    cmd = [
        "python", "-m", "src.train.train_click",
        "--model_id", "Qwen/Qwen2.5-VL-3B-Instruct",
        "--data_path", "Salesforce/grounding_dataset",
        "--output_dir", output_dir,
        "--bf16", "True",
        "--num_train_epochs", str(epochs),
        "--per_device_train_batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(grad_accum),
        "--learning_rate", str(lr),
        "--weight_decay", "0.01",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "10",
        "--save_strategy", "epoch",
        "--save_total_limit", "2",
        "--image_min_pixels", "200704",
        "--image_max_pixels", "401408",
        "--freeze_merger", "False",
        "--mlp_head_dim", "512",
        "--mlp_head_dropout", "0.1",
        "--gradient_checkpointing", "True",
        "--dataloader_num_workers", "4",
        "--disable_flash_attn2", "True",
        "--report_to", "wandb",
        "--run_name", "qwen-click-dit-h100",
        "--max_samples", str(max_samples),
        "--use_liger_kernel", "False",
    ]

    print("=" * 60)
    print("Starting training...")
    print(f"Output dir: {output_dir}")
    print(f"Working dir: {os.getcwd()}")
    print("=" * 60)

    # Run training with streaming output
    result = subprocess.run(cmd, cwd="/app")

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with code {result.returncode}")

    # Commit volume to persist checkpoints
    volume.commit()

    print("=" * 60)
    print("Training complete!")
    print(f"Model saved to: {output_dir}")
    print("=" * 60)

    return output_dir


@app.function(
    gpu="A10G",  # Cheaper GPU for inference
    timeout=300,
    volumes={VOLUME_PATH: volume},
)
def inference(image_url: str, prompt: str):
    """Run inference on a trained model."""
    import os
    import sys
    import torch
    from PIL import Image
    import requests
    from io import BytesIO

    os.chdir("/app")
    sys.path.insert(0, "/app/src")

    # Download image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    # Save temporarily
    image_path = "/tmp/input.png"
    image.save(image_path)

    # Load model
    model_path = f"{VOLUME_PATH}/outputs/click_dit"

    from src.inference import load_model, predict as run_predict

    model, processor = load_model(model_path)
    x, y = run_predict(model, processor, image_path, prompt)

    w, h = image.size
    return {
        "normalized": {"x": x, "y": y},
        "pixels": {"x": int(x * w), "y": int(y * h)},
        "image_size": {"width": w, "height": h},
    }


def find_latest_checkpoint(base_path: str) -> str:
    """Find the latest checkpoint directory."""
    import os
    import glob

    # Check if base_path itself has model files
    if os.path.exists(os.path.join(base_path, "model.safetensors")) or \
       glob.glob(os.path.join(base_path, "model-*.safetensors")):
        return base_path

    # Look for checkpoint-* directories
    checkpoints = glob.glob(os.path.join(base_path, "checkpoint-*"))
    if checkpoints:
        # Sort by step number and get latest
        checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
        return checkpoints[-1]

    return base_path


@app.function(
    gpu="A10G",
    timeout=3600,
    volumes={VOLUME_PATH: volume},
    secrets=[hf_secret],
)
def benchmark(num_samples: int = 100):
    """Run benchmark evaluation on trained model."""
    import os
    import sys
    import json

    os.chdir("/app")
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/app")

    # Cache HF datasets on volume
    os.environ["HF_HOME"] = f"{VOLUME_PATH}/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = f"{VOLUME_PATH}/hf_cache/datasets"

    base_path = f"{VOLUME_PATH}/outputs/click_dit"
    model_path = find_latest_checkpoint(base_path)
    output_json = f"{VOLUME_PATH}/outputs/benchmark_results.json"

    print(f"Loading model from {model_path}...")

    from src.inference import load_model
    from scripts.benchmark import run_benchmark, print_results

    model, processor = load_model(model_path, device="cuda")

    results = run_benchmark(
        model, processor,
        dataset_name="Salesforce/grounding_dataset",
        num_samples=num_samples,
        device="cuda",
    )

    print_results(results)

    # Save results
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2, default=float)

    volume.commit()
    print(f"\nResults saved to: {output_json}")

    return results


@app.function(
    gpu="A10G",
    timeout=1800,
    volumes={VOLUME_PATH: volume},
    secrets=[hf_secret],
)
def visualize(num_samples: int = 20):
    """Generate visualization images showing predictions vs targets."""
    import os
    import sys

    os.chdir("/app")
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/app")

    # Cache HF datasets on volume
    os.environ["HF_HOME"] = f"{VOLUME_PATH}/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = f"{VOLUME_PATH}/hf_cache/datasets"

    base_path = f"{VOLUME_PATH}/outputs/click_dit"
    model_path = find_latest_checkpoint(base_path)
    output_dir = f"{VOLUME_PATH}/outputs/visualizations"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {model_path}...")

    from src.inference import load_model
    from scripts.visualize_predictions import visualize_from_dataset

    model, processor = load_model(model_path, device="cuda")

    visualize_from_dataset(
        model, processor,
        output_dir=output_dir,
        num_samples=num_samples,
        device="cuda",
    )

    volume.commit()
    print(f"\nVisualizations saved to: {output_dir}")

    return output_dir


@app.function(
    gpu="A10G",
    timeout=1800,
    volumes={VOLUME_PATH: volume},
    secrets=[hf_secret],
)
def upload_to_hub(repo_id: str = "TESS-Computer/qwen-click-dit"):
    """Upload trained model to HuggingFace Hub."""
    import os
    import sys

    os.chdir("/app")
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/app")

    base_path = f"{VOLUME_PATH}/outputs/click_dit"
    model_path = find_latest_checkpoint(base_path)
    metrics_path = f"{VOLUME_PATH}/outputs/benchmark_results.json"

    print(f"Uploading from {model_path}...")

    from scripts.upload_to_hub import upload_model

    upload_model(
        model_path=model_path,
        repo_id=repo_id,
        private=False,
        metrics_path=metrics_path if os.path.exists(metrics_path) else None,
    )

    print(f"\n✅ Model uploaded to: https://huggingface.co/{repo_id}")
    return repo_id


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60,
)
def list_checkpoints():
    """List checkpoint files on volume."""
    import os
    import subprocess
    result = subprocess.run(
        ["find", "/vol/outputs", "-type", "f", "-name", "*.safetensors"],
        capture_output=True, text=True
    )
    print("Safetensors files:")
    print(result.stdout or "None found")

    result2 = subprocess.run(
        ["ls", "-laR", "/vol/outputs/"],
        capture_output=True, text=True
    )
    print("\nFull directory listing:")
    print(result2.stdout[:5000] if result2.stdout else "Empty")
    return result.stdout


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=600,
)
def list_visualizations():
    """List visualization files on volume."""
    import os
    viz_dir = f"{VOLUME_PATH}/outputs/visualizations"
    if os.path.exists(viz_dir):
        files = os.listdir(viz_dir)
        return [f for f in files if f.endswith('.png')]
    return []


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=600,
)
def get_visualization(filename: str) -> bytes:
    """Get a visualization file as bytes."""
    import os
    filepath = f"{VOLUME_PATH}/outputs/visualizations/{filename}"
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return f.read()
    return b""


@app.local_entrypoint()
def main(
    action: str = "train",
    num_samples: int = 100,
    repo_id: str = "TESS-Computer/qwen-click-dit",
):
    """
    Entry point for modal commands.

    Usage:
        modal run modal_train.py                           # Train
        modal run modal_train.py --action benchmark        # Run benchmark
        modal run modal_train.py --action visualize        # Generate visualizations
        modal run modal_train.py --action upload           # Upload to HF
        modal run modal_train.py --action download-viz     # Download visualizations locally
        modal run modal_train.py --action all              # Benchmark + Visualize + Upload
    """
    import os

    if action == "train":
        print("🚀 Starting training on Modal H100...")
        output_dir = train.remote()
        print(f"✅ Done! Model saved to: {output_dir}")

    elif action == "benchmark":
        print(f"📊 Running benchmark on {num_samples} samples...")
        results = benchmark.remote(num_samples=num_samples)
        print("✅ Benchmark complete!")

    elif action == "visualize":
        print(f"🎨 Generating {num_samples} visualizations...")
        viz_dir = visualize.remote(num_samples=num_samples)
        print(f"✅ Visualizations saved to: {viz_dir}")

    elif action == "upload":
        print(f"📤 Uploading to {repo_id}...")
        upload_to_hub.remote(repo_id=repo_id)
        print("✅ Upload complete!")

    elif action == "download-viz":
        print("📥 Downloading visualizations...")
        files = list_visualizations.remote()
        os.makedirs("visualizations", exist_ok=True)
        for f in files:
            data = get_visualization.remote(f)
            with open(f"visualizations/{f}", "wb") as out:
                out.write(data)
            print(f"  Downloaded: {f}")
        print(f"✅ Downloaded {len(files)} files to ./visualizations/")

    elif action == "all":
        print("🚀 Running full pipeline: benchmark → visualize → upload")
        print("\n📊 Step 1: Benchmark...")
        benchmark.remote(num_samples=num_samples)
        print("\n🎨 Step 2: Visualize...")
        visualize.remote(num_samples=num_samples)
        print("\n📤 Step 3: Upload...")
        upload_to_hub.remote(repo_id=repo_id)
        print("\n✅ All done!")

    elif action == "list":
        print("📂 Listing checkpoints on volume...")
        list_checkpoints.remote()

    else:
        print(f"Unknown action: {action}")
        print("Valid actions: train, benchmark, visualize, upload, download-viz, all, list")
