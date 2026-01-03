"""
Inference script for click prediction.

Usage:
    python -m src.inference.predict_click \
        --model_path outputs/click_dit \
        --image_path screenshot.png \
        --prompt "Click on the search button"
"""

import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, AutoConfig
from src.model import Qwen2_5_VLForClickPrediction


def load_model(model_path: str, device: str = "cuda"):
    """Load trained model from checkpoint."""
    import os

    # Try to load config - may need to fall back to base model config
    try:
        config = AutoConfig.from_pretrained(model_path)
    except ValueError:
        # Config doesn't have model_type, load from base model and add our fields
        print("Loading config from base model...")
        config = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    # Add DiT config fields if not present
    if not hasattr(config, 'dit_hidden_size'):
        config.dit_hidden_size = 512
        config.dit_num_layers = 6
        config.dit_num_heads = 8
        config.dit_dropout = 0.1
        config.num_inference_steps = 16

    # Load model
    model = Qwen2_5_VLForClickPrediction.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = model.to(device)
    model.eval()

    # Load processor - try model path first, fall back to base model
    try:
        processor = AutoProcessor.from_pretrained(model_path)
    except Exception:
        print("Loading processor from base model...")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    return model, processor


def predict(
    model,
    processor,
    image_path: str,
    prompt: str,
    device: str = "cuda",
    min_pixels: int = 200704,
    max_pixels: int = 401408,
) -> tuple[float, float]:
    """
    Predict click coordinates for an image + prompt.

    Returns:
        (x, y) normalized coordinates in [0, 1]
    """
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Build message
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process inputs
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        **video_kwargs,
    )

    # Move to device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        click_xy = model.predict(**inputs)

    x, y = click_xy[0].cpu().tolist()
    return x, y


def main():
    parser = argparse.ArgumentParser(description="Predict click coordinates")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image")
    parser.add_argument("--prompt", type=str, required=True, help="Text instruction")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model, processor = load_model(args.model_path, args.device)

    print(f"Predicting click for: {args.prompt}")
    x, y = predict(model, processor, args.image_path, args.prompt, args.device)

    # Load image to get pixel coordinates
    image = Image.open(args.image_path)
    w, h = image.size
    pixel_x = int(x * w)
    pixel_y = int(y * h)

    print(f"\nResult:")
    print(f"  Normalized: ({x:.4f}, {y:.4f})")
    print(f"  Pixels:     ({pixel_x}, {pixel_y}) on {w}x{h} image")


if __name__ == "__main__":
    main()
