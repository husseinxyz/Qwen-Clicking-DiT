"""
Click Prediction Dataset for GUI grounding.

Loads grounding datasets (like GTA1) and returns:
- Image + text prompt as input
- Normalized (x, y) click coordinates as labels
"""

import os
from typing import Dict, List, Tuple, Any
import torch
import transformers
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
from qwen_vl_utils import process_vision_info

from src.params import DataArguments
from src.constants import SYSTEM_MESSAGE
from .data_utils import pad_sequence


def box_to_normalized_click(
    box: List[float],
    image_width: int,
    image_height: int,
    box_format: str = "xyxy",
) -> Tuple[float, float]:
    """
    Convert bounding box to normalized click coordinates.

    Args:
        box: Bounding box coordinates
        image_width: Original image width
        image_height: Original image height
        box_format: "xyxy" or "xywh"

    Returns:
        (cx, cy): Center point in [0, 1] normalized coordinates
    """
    if box_format == "xywh":
        x0, y0, w, h = box
        x1 = x0 + w
        y1 = y0 + h
    elif box_format == "xyxy":
        x0, y0, x1, y1 = box
    else:
        raise ValueError(f"Unsupported box_format: {box_format}")

    # Normalize to [0, 1]
    x0_norm = max(0.0, min(1.0, x0 / image_width))
    y0_norm = max(0.0, min(1.0, y0 / image_height))
    x1_norm = max(0.0, min(1.0, x1 / image_width))
    y1_norm = max(0.0, min(1.0, y1 / image_height))

    # Click = center of box
    cx = (x0_norm + x1_norm) / 2.0
    cy = (y0_norm + y1_norm) / 2.0

    return cx, cy


def extract_box(box: Any) -> List[float]:
    """Extract box coordinates from various formats."""
    if isinstance(box, dict):
        if all(k in box for k in ("x", "y", "width", "height")):
            return [box["x"], box["y"], box["width"], box["height"]]
        if all(k in box for k in ("xmin", "ymin", "xmax", "ymax")):
            return [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]
    if isinstance(box, (list, tuple)) and len(box) > 0:
        if len(box) == 4 and isinstance(box[0], (int, float)):
            return list(box)
        if isinstance(box[0], (list, tuple)) and len(box[0]) == 4:
            return list(box[0])
    raise ValueError("Unsupported box format in dataset")


def get_image_content(image, min_pixel, max_pixel, width=None, height=None):
    """Create image content dict for Qwen VL processor."""
    content = {
        "type": "image",
        "image": image,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel
    }
    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    return content


class ClickDataset(Dataset):
    """
    Dataset for click prediction from GUI grounding data.

    Supports:
    - HuggingFace datasets (GTA1 grounding_dataset format)
    - Local JSON files with image paths
    """

    def __init__(
        self,
        data_path: str,  # HF dataset name or local JSON path
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id: str,
        dataset_split: str = "train",
        max_samples: int = None,
        box_format: str = "xyxy",
    ):
        super().__init__()

        self.processor = processor
        self.data_args = data_args
        self.model_id = model_id
        self.box_format = box_format
        self.compute_dtype = data_args.compute_dtype

        # Resolution settings
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height

        # Load dataset
        if data_path.endswith('.json'):
            # Local JSON file
            import json
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        else:
            # HuggingFace dataset
            ds = load_dataset(data_path, split=dataset_split)
            if max_samples is not None and max_samples < len(ds):
                ds = ds.select(range(max_samples))
                print(f"Using {max_samples} samples from {len(ds)} total")
            self.data = ds

        print(f"Loaded {len(self.data)} samples for click prediction")

    def __len__(self):
        return len(self.data)

    def _extract_text_from_conversations(self, conversations: list) -> str:
        """Extract instruction from GTA1 conversations format."""
        if not conversations or len(conversations) == 0:
            return ""
        human_msg = conversations[0].get("value", "")
        human_msg = human_msg.replace("<image>", "").strip()
        return human_msg

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]

        # Get image
        if "image" in sample:
            image = sample["image"]
            if isinstance(image, str):
                # Path to image
                if self.data_args.image_folder:
                    image = os.path.join(self.data_args.image_folder, image)
                image = Image.open(image).convert("RGB")
            elif hasattr(image, "convert"):
                image = image.convert("RGB")
        else:
            raise ValueError(f"No image found in sample {idx}")

        width, height = image.size

        # Get text instruction
        if "conversations" in sample:
            # GTA1 format
            text = self._extract_text_from_conversations(sample["conversations"])
        elif "prompt" in sample:
            text = sample["prompt"]
        elif "text" in sample:
            text = sample["text"]
        else:
            text = "Click on the target element."

        # Get bounding box and convert to click coordinates
        if "bbox" in sample:
            box = extract_box(sample["bbox"])
            cx, cy = box_to_normalized_click(box, width, height, self.box_format)
        elif "click_xy" in sample:
            # Already normalized coordinates
            cx, cy = sample["click_xy"]
        else:
            raise ValueError(f"No bbox or click_xy found in sample {idx}")

        # Build message content
        contents = []
        contents.append(get_image_content(
            image,
            self.image_min_pixel,
            self.image_max_pixel,
            self.image_resized_w,
            self.image_resized_h
        ))
        contents.append({"type": "text", "text": text})

        # Build messages
        user_prompt = [{"role": "user", "content": contents}]
        if len(SYSTEM_MESSAGE) > 0:
            user_prompt.insert(0, {"role": "system", "content": SYSTEM_MESSAGE})

        # Apply chat template
        formatted_text = self.processor.apply_chat_template(
            user_prompt, tokenize=False, add_generation_prompt=True
        )

        # Process vision info
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            user_prompt, return_video_kwargs=True
        )

        # Process with Qwen processor
        data_dict = self.processor(
            text=formatted_text,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            **video_kwargs
        )

        # Add labels (click coordinates)
        data_dict['labels'] = torch.tensor([cx, cy], dtype=torch.float32)

        # Attention mask
        data_dict['attention_mask'] = (data_dict['input_ids'] > -1000000).to(torch.long)

        # Cast to compute dtype
        for key, value in data_dict.items():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                data_dict[key] = value.to(self.compute_dtype)

        return data_dict


class DataCollatorForClickDataset:
    """Collate examples for click prediction training."""

    def __init__(self, pad_token_id: int, padding_side: str = "left"):
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def __call__(self, examples):
        batch_input_ids = []
        batch_labels = []
        batch_pixel_values = []
        batch_image_thw = []
        batch_second_per_grid_ts = []

        for example in examples:
            keys = example.keys()

            if "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])

            batch_input_ids.append(example["input_ids"].squeeze(0))
            batch_labels.append(example["labels"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

        # Pad input_ids
        input_ids = pad_sequence(
            batch_input_ids, padding_side=self.padding_side, padding_value=self.pad_token_id
        )

        # Stack labels
        labels = torch.stack(batch_labels, dim=0)  # (batch, 2)

        attention_mask = input_ids != self.pad_token_id

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict


def make_click_data_module(
    model_id: str,
    processor: transformers.ProcessorMixin,
    data_args: DataArguments,
    dataset_name: str = "HelloKKMe/grounding_dataset",
    dataset_split: str = "train",
    max_samples: int = None,
    eval_split: str = None,
):
    """Create data module for click prediction training."""

    train_dataset = ClickDataset(
        data_path=dataset_name,
        processor=processor,
        data_args=data_args,
        model_id=model_id,
        dataset_split=dataset_split,
        max_samples=max_samples,
    )

    train_collator = DataCollatorForClickDataset(
        pad_token_id=processor.tokenizer.pad_token_id,
        padding_side="left"
    )

    eval_dataset = None
    eval_collator = None

    if eval_split is not None:
        eval_dataset = ClickDataset(
            data_path=dataset_name,
            processor=processor,
            data_args=data_args,
            model_id=model_id,
            dataset_split=eval_split,
        )
        eval_collator = DataCollatorForClickDataset(
            pad_token_id=processor.tokenizer.pad_token_id,
            padding_side="left"
        )

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_data_collator=train_collator,
        eval_data_collator=eval_collator,
    )
