"""
Benchmark evaluation script for click prediction model.

Evaluates on Salesforce/grounding_dataset and computes:
- Inside-box accuracy (click lands within target bbox)
- Distance to center (normalized L2)
- Accuracy at different thresholds

Usage:
    python scripts/benchmark.py \
        --model_path outputs/click_dit \
        --num_samples 1000 \
        --output_json benchmark_results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List
import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import load_model, predict


def evaluate_sample(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    bbox: List[int],  # [x1, y1, x2, y2]
    device: str = "cuda",
) -> Dict:
    """Evaluate a single sample."""
    w, h = image.size
    x1, y1, x2, y2 = bbox

    # Save temp image
    temp_path = "/tmp/bench_temp.png"
    image.save(temp_path)

    # Predict
    pred_x, pred_y = predict(model, processor, temp_path, prompt, device)

    # Convert to pixels
    px = pred_x * w
    py = pred_y * h

    # Target center (normalized)
    target_cx = (x1 + x2) / 2 / w
    target_cy = (y1 + y2) / 2 / h

    # Distance to center (normalized)
    dist = np.sqrt((pred_x - target_cx) ** 2 + (pred_y - target_cy) ** 2)

    # Inside box check
    inside = x1 <= px <= x2 and y1 <= py <= y2

    return {
        "pred_x": pred_x,
        "pred_y": pred_y,
        "target_cx": target_cx,
        "target_cy": target_cy,
        "distance": dist,
        "inside_box": inside,
        "image_size": (w, h),
    }


def run_benchmark(
    model,
    processor,
    dataset_name: str = "Salesforce/grounding_dataset",
    split: str = "train",
    num_samples: int = 1000,
    device: str = "cuda",
    seed: int = 42,
) -> Dict:
    """Run benchmark on dataset."""

    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split=split)

    # Sample indices
    np.random.seed(seed)
    if num_samples < len(ds):
        indices = np.random.choice(len(ds), num_samples, replace=False)
    else:
        indices = list(range(len(ds)))
        num_samples = len(ds)

    print(f"Evaluating on {num_samples} samples...")

    results = []
    errors = 0

    for idx in tqdm(indices, desc="Evaluating"):
        sample = ds[int(idx)]

        # Get image
        image = sample.get("image")
        if not isinstance(image, Image.Image):
            errors += 1
            continue

        # Get prompt
        conversations = sample.get("conversations", [])
        if conversations:
            prompt = conversations[0].get("value", "").replace("<image>", "").strip()
        else:
            prompt = "Click on the target"

        # Get bbox
        bbox = sample.get("bbox")
        if not bbox or len(bbox) != 4:
            errors += 1
            continue

        try:
            result = evaluate_sample(model, processor, image, prompt, list(bbox), device)
            result["idx"] = int(idx)
            results.append(result)
        except Exception as e:
            errors += 1
            continue

    # Compute metrics
    if not results:
        return {"error": "No valid samples"}

    distances = [r["distance"] for r in results]
    inside_box = [r["inside_box"] for r in results]

    metrics = {
        "num_samples": len(results),
        "num_errors": errors,

        # Inside-box accuracy
        "inside_box_accuracy": np.mean(inside_box),

        # Distance metrics
        "mean_distance": np.mean(distances),
        "median_distance": np.median(distances),
        "std_distance": np.std(distances),
        "min_distance": np.min(distances),
        "max_distance": np.max(distances),

        # Accuracy at thresholds (% of predictions within X normalized distance)
        "acc_at_0.05": np.mean([d < 0.05 for d in distances]),  # Within 5%
        "acc_at_0.10": np.mean([d < 0.10 for d in distances]),  # Within 10%
        "acc_at_0.15": np.mean([d < 0.15 for d in distances]),  # Within 15%
        "acc_at_0.20": np.mean([d < 0.20 for d in distances]),  # Within 20%
    }

    return {
        "metrics": metrics,
        "samples": results[:100],  # Save first 100 for inspection
    }


def print_results(results: Dict):
    """Pretty print benchmark results."""
    metrics = results.get("metrics", {})

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\nSamples evaluated: {metrics.get('num_samples', 0)}")
    print(f"Errors: {metrics.get('num_errors', 0)}")

    print(f"\n{'Metric':<30} {'Value':>15}")
    print("-" * 45)

    print(f"{'Inside-box Accuracy':<30} {metrics.get('inside_box_accuracy', 0)*100:>14.1f}%")
    print(f"{'Mean Distance to Center':<30} {metrics.get('mean_distance', 0):>15.4f}")
    print(f"{'Median Distance to Center':<30} {metrics.get('median_distance', 0):>15.4f}")

    print(f"\n{'Accuracy @ Threshold':<30}")
    print("-" * 45)
    print(f"{'  Within 5% of image':<30} {metrics.get('acc_at_0.05', 0)*100:>14.1f}%")
    print(f"{'  Within 10% of image':<30} {metrics.get('acc_at_0.10', 0)*100:>14.1f}%")
    print(f"{'  Within 15% of image':<30} {metrics.get('acc_at_0.15', 0)*100:>14.1f}%")
    print(f"{'  Within 20% of image':<30} {metrics.get('acc_at_0.20', 0)*100:>14.1f}%")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark click prediction model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--dataset_name", type=str, default="Salesforce/grounding_dataset")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, default="benchmark_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model, processor = load_model(args.model_path, args.device)

    results = run_benchmark(
        model, processor,
        dataset_name=args.dataset_name,
        split=args.split,
        num_samples=args.num_samples,
        device=args.device,
        seed=args.seed,
    )

    # Print results
    print_results(results)

    # Save to JSON
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()
