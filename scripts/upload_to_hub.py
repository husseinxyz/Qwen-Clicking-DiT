"""
Upload trained model to HuggingFace Hub.

Usage:
    python scripts/upload_to_hub.py \
        --model_path outputs/click_dit \
        --repo_id TESS-Computer/qwen-click-dit \
        --private False
"""

import argparse
import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoConfig, AutoProcessor


def create_model_card(
    repo_id: str,
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    dataset: str = "Salesforce/grounding_dataset",
    metrics: dict = None,
) -> str:
    """Create a model card for the HuggingFace Hub."""

    metrics_section = ""
    if metrics:
        metrics_section = f"""
## Metrics

| Metric | Value |
|--------|-------|
| Inside-box Accuracy | {metrics.get('inside_box_accuracy', 'N/A'):.1%} |
| Mean Distance | {metrics.get('mean_distance', 'N/A'):.4f} |
| Acc @ 5% | {metrics.get('acc_at_0.05', 'N/A'):.1%} |
| Acc @ 10% | {metrics.get('acc_at_0.10', 'N/A'):.1%} |
"""

    return f"""---
license: mit
base_model: {base_model}
datasets:
- {dataset}
tags:
- vision-language
- click-prediction
- gui-grounding
- diffusion-transformer
- flow-matching
pipeline_tag: image-to-text
---

# Qwen-Click-DiT

Vision-Language Model with Diffusion Transformer for GUI Click Prediction.

## Model Description

This model predicts click coordinates given a screenshot and natural language instruction. It uses:
- **Qwen2.5-VL-3B** as a frozen vision-language backbone
- **DiT (Diffusion Transformer)** action head using flow matching

## Quick Start

### Installation

```bash
pip install torch transformers accelerate qwen-vl-utils pillow
git clone https://github.com/husseinxyz/Qwen-Click-DiT.git
cd Qwen-Click-DiT
```

### Inference

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info

# Clone the repo first to get the model class
from src.model import Qwen2_5_VLForClickPrediction

# Load model
model_id = "{repo_id}"
config = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
config.dit_hidden_size = 512
config.dit_num_layers = 6
config.dit_num_heads = 8
config.dit_dropout = 0.1
config.num_inference_steps = 16

model = Qwen2_5_VLForClickPrediction.from_pretrained(
    model_id, config=config, torch_dtype=torch.bfloat16
)
model = model.to("cuda").eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# Prepare input
image = Image.open("screenshot.png").convert("RGB")
prompt = "Click on the search button"

messages = [{{
    "role": "user",
    "content": [
        {{"type": "image", "image": image, "min_pixels": 200704, "max_pixels": 401408}},
        {{"type": "text", "text": prompt}},
    ],
}}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
inputs = processor(text=text, images=image_inputs, videos=video_inputs, return_tensors="pt", **video_kwargs)
inputs = {{k: v.to("cuda") if torch.is_tensor(v) else v for k, v in inputs.items()}}

# Predict click coordinates
with torch.no_grad():
    click_xy = model.predict(**inputs)

x, y = click_xy[0].cpu().tolist()
print(f"Normalized: ({{x:.4f}}, {{y:.4f}})")
print(f"Pixels: ({{int(x * image.width)}}, {{int(y * image.height)}})")
```

See [GitHub repo](https://github.com/husseinxyz/Qwen-Click-DiT) for more examples.
{metrics_section}
## Training

- **Dataset**: [{dataset}](https://huggingface.co/datasets/{dataset})
- **Samples**: 20,000
- **Epochs**: 3
- **Base Model**: [{base_model}](https://huggingface.co/{base_model})

## Architecture

| Component | Value |
|-----------|-------|
| DiT Hidden Size | 512 |
| DiT Layers | 6 |
| DiT Heads | 8 |
| Inference Steps | 16 |

## Citation

```bibtex
@misc{{2026qwenclickdit,
  title = {{Qwen-Click-DiT: Vision-Language Model with Diffusion Transformer for GUI Click Prediction}},
  author = {{}},
  year = {{2026}},
  howpublished = {{\\url{{https://github.com/husseinxyz/Qwen-Click-DiT}}}},
}}
```

## License

MIT
"""


def upload_model(
    model_path: str,
    repo_id: str,
    private: bool = False,
    metrics_path: str = None,
):
    """Upload model to HuggingFace Hub."""

    model_path = Path(model_path)
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")

    print(f"Uploading model from {model_path} to {repo_id}...")

    # Initialize API
    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, private=private, exist_ok=True)
        print(f"Repository created/exists: {repo_id}")
    except Exception as e:
        print(f"Note: {e}")

    # Load metrics if provided
    metrics = None
    if metrics_path and os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
            metrics = data.get("metrics", {})

    # Create model card
    model_card = create_model_card(repo_id, metrics=metrics)
    readme_path = model_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(model_card)
    print("Created model card")

    # Upload entire folder
    print("Uploading files...")
    upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload Qwen-Click-DiT model",
    )

    print(f"\nModel uploaded successfully!")
    print(f"View at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace repo ID (user/model)")
    parser.add_argument("--private", type=bool, default=False, help="Make repo private")
    parser.add_argument("--metrics_path", type=str, default=None, help="Path to benchmark_results.json")
    args = parser.parse_args()

    upload_model(
        model_path=args.model_path,
        repo_id=args.repo_id,
        private=args.private,
        metrics_path=args.metrics_path,
    )


if __name__ == "__main__":
    main()
