"""
Visualize click predictions - clean demo images.

Shows:
- Screenshot with prompt
- Ground Truth (green dot)
- Model Prediction (red dot)
- Simple legend

Usage:
    python scripts/visualize_predictions.py \
        --model_path outputs/click_dit \
        --from_dataset \
        --num_samples 10
"""

import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import load_model, predict


def draw_prediction(
    image: Image.Image,
    pred_x: float,
    pred_y: float,
    prompt: str,
    bbox: tuple = None,  # (x1, y1, x2, y2) in pixels
) -> Image.Image:
    """
    Draw clean prediction visualization.

    Args:
        image: PIL Image
        pred_x, pred_y: Predicted click in normalized [0,1] coords
        prompt: Text prompt used
        bbox: Ground truth bounding box (x1, y1, x2, y2) in pixels

    Returns:
        Annotated PIL Image
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Convert normalized coords to pixels
    px = int(pred_x * w)
    py = int(pred_y * h)

    # Try to load a nice font
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        try:
            font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            font_medium = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            font_large = ImageFont.load_default()
            font_medium = font_large

    # Draw ground truth (green dot) if bbox provided
    if bbox:
        x1, y1, x2, y2 = bbox
        target_cx = (x1 + x2) // 2
        target_cy = (y1 + y2) // 2

        # Green dot for ground truth - larger and more visible
        dot_size = 10
        draw.ellipse(
            [target_cx - dot_size, target_cy - dot_size,
             target_cx + dot_size, target_cy + dot_size],
            fill="#00FF00",
            outline="#006600",
            width=2
        )

    # Draw prediction (red dot) - larger and more visible
    dot_size = 10
    draw.ellipse(
        [px - dot_size, py - dot_size, px + dot_size, py + dot_size],
        fill="#FF0000",
        outline="#660000",
        width=2
    )

    # Draw header bar with prompt
    header_height = 50
    draw.rectangle([0, 0, w, header_height], fill=(0, 0, 0, 230))

    # Truncate prompt if too long
    display_prompt = prompt if len(prompt) < 60 else prompt[:57] + "..."
    draw.text((15, 12), f'"{display_prompt}"', fill="#FFFFFF", font=font_large)

    # Draw legend at bottom
    legend_height = 35
    draw.rectangle([0, h - legend_height, w, h], fill=(0, 0, 0, 230))

    # Legend items
    legend_y = h - legend_height + 8

    # Green dot + label
    draw.ellipse([15, legend_y + 2, 27, legend_y + 14], fill="#00FF00")
    draw.text((35, legend_y - 2), "Ground Truth", fill="#00FF00", font=font_medium)

    # Red dot + label
    draw.ellipse([180, legend_y + 2, 192, legend_y + 14], fill="#FF0000")
    draw.text((200, legend_y - 2), "Prediction", fill="#FF0000", font=font_medium)

    return img


def visualize_from_dataset(
    model,
    processor,
    dataset_name: str = "Salesforce/grounding_dataset",
    num_samples: int = 10,
    output_dir: str = "visualizations",
    device: str = "cuda",
):
    """
    Visualize predictions on samples from the dataset.
    Only selects samples with clear, descriptive prompts.
    """
    from datasets import load_dataset
    import os
    import random

    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    ds = load_dataset(dataset_name, split="train")

    # Find valid samples with bbox and image
    good_samples = []
    for idx in range(min(5000, len(ds))):  # Search first 5000
        sample = ds[idx]

        # Must have bbox
        bbox = sample.get("bbox", None)
        if not bbox:
            continue

        # Must have valid image
        image = sample.get("image")
        if not isinstance(image, Image.Image):
            continue

        good_samples.append(idx)

    # Sample from good ones
    if len(good_samples) < num_samples:
        print(f"Warning: Only found {len(good_samples)} good samples")
    indices = random.sample(good_samples, min(num_samples, len(good_samples)))

    saved_count = 0
    for idx in indices:
        sample = ds[idx]
        image = sample["image"]
        w, h = image.size

        # Get prompt - try multiple fields
        prompt = None
        conversations = sample.get("conversations", [])
        if conversations:
            prompt = conversations[0].get("value", "").replace("<image>", "").strip()

        # Try other common fields
        if not prompt:
            prompt = sample.get("prompt", "")
        if not prompt:
            prompt = sample.get("text", "")
        if not prompt:
            prompt = sample.get("instruction", "")

        # Fallback - describe what to click based on element type if available
        if not prompt or prompt.lower() in ["click on the target", ""]:
            element_type = sample.get("element_type", sample.get("type", "element"))
            prompt = f"Click on the {element_type}"

        # Get bbox
        bbox = tuple(sample["bbox"])

        # Save image temporarily
        temp_path = f"/tmp/temp_img_{saved_count}.png"
        image.save(temp_path)

        # Predict
        try:
            pred_x, pred_y = predict(model, processor, temp_path, prompt, device)
        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            continue

        # Visualize
        vis_img = draw_prediction(image, pred_x, pred_y, prompt, bbox)

        # Save
        output_path = os.path.join(output_dir, f"sample_{saved_count:03d}.png")
        vis_img.save(output_path)
        print(f"Saved: {output_path}")
        saved_count += 1

    print(f"\nDone! Saved {saved_count} visualizations to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Visualize click predictions")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--image_path", type=str, help="Path to single image")
    parser.add_argument("--prompt", type=str, help="Text prompt for single image")
    parser.add_argument("--bbox", type=str, help="Ground truth bbox: x1,y1,x2,y2")
    parser.add_argument("--output_path", type=str, default="prediction.png", help="Output image path")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    # Dataset visualization
    parser.add_argument("--from_dataset", action="store_true", help="Visualize from dataset")
    parser.add_argument("--dataset_name", type=str, default="Salesforce/grounding_dataset")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Output directory")

    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model, processor = load_model(args.model_path, args.device)

    if args.from_dataset:
        # Visualize from dataset
        visualize_from_dataset(
            model, processor,
            dataset_name=args.dataset_name,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            device=args.device,
        )
    else:
        # Single image
        if not args.image_path or not args.prompt:
            parser.error("--image_path and --prompt required for single image mode")

        print(f"Predicting: {args.prompt}")
        pred_x, pred_y = predict(model, processor, args.image_path, args.prompt, args.device)

        # Parse bbox if provided
        bbox = None
        if args.bbox:
            bbox = tuple(map(int, args.bbox.split(",")))

        # Load and visualize
        image = Image.open(args.image_path).convert("RGB")
        vis_img = draw_prediction(image, pred_x, pred_y, args.prompt, bbox)
        vis_img.save(args.output_path)

        print(f"\nPrediction: ({pred_x:.4f}, {pred_y:.4f})")
        print(f"Saved visualization to: {args.output_path}")


if __name__ == "__main__":
    main()
