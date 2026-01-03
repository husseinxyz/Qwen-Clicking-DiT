"""
Training script for click prediction with DiT action head.

Trains only the DiT action head while keeping the VLM backbone frozen.
"""

import os
import torch
from transformers import AutoProcessor, HfArgumentParser, AutoConfig
from src.trainer import QwenCLSTrainer
from src.model import Qwen2_5_VLForClickPrediction
from src.dataset import make_click_data_module
from src.params import DataArguments, ModelArguments, CLSArguments
from train.train_utils import safe_save_model_for_hf_trainer
import pathlib
import numpy as np

local_rank = None


def compute_metrics(pred):
    """Compute metrics for click prediction."""
    preds = pred.predictions
    labels = pred.label_ids

    mse = np.mean((preds - labels) ** 2)
    distances = np.sqrt(np.sum((preds - labels) ** 2, axis=1))
    mean_dist = np.mean(distances)
    acc_5 = np.mean(distances < 0.05)
    acc_10 = np.mean(distances < 0.10)

    return {
        "mse": mse,
        "mean_dist": mean_dist,
        "acc@5%": acc_5,
        "acc@10%": acc_10,
    }


def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)


def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, CLSArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    compute_dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32)
    data_args.compute_dtype = compute_dtype

    # Load model config with DiT settings
    cfg = AutoConfig.from_pretrained(model_args.model_id)
    cfg.dit_hidden_size = training_args.mlp_head_dim if training_args.mlp_head_dim > 0 else 512
    cfg.dit_num_layers = 6
    cfg.dit_num_heads = 8
    cfg.dit_dropout = training_args.mlp_head_dropout if training_args.mlp_head_dropout > 0 else 0.1
    cfg.num_inference_steps = 16

    rank0_print(f"Loading model: {model_args.model_id}")
    rank0_print(f"DiT config: hidden={cfg.dit_hidden_size}, layers={cfg.dit_num_layers}, heads={cfg.dit_num_heads}")

    # Load model
    model = Qwen2_5_VLForClickPrediction.from_pretrained(
        model_args.model_id,
        config=cfg,
        torch_dtype=compute_dtype,
        attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
    )

    model.config.use_cache = False

    # Freeze backbone, train only action head
    for param in model.model.parameters():
        param.requires_grad = False

    for param in model.action_head.parameters():
        param.requires_grad = True

    # Optionally unfreeze merger
    if not training_args.freeze_merger:
        for param in model.model.visual.merger.parameters():
            param.requires_grad = True

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    processor = AutoProcessor.from_pretrained(model_args.model_id)
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    # Create data module
    dataset_name = data_args.data_path if data_args.data_path else "HelloKKMe/grounding_dataset"

    data_module = make_click_data_module(
        model_id=model_args.model_id,
        processor=processor,
        data_args=data_args,
        dataset_name=dataset_name,
        dataset_split="train",
        max_samples=data_args.max_samples,
        eval_split=data_args.eval_path,
    )

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    rank0_print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    trainer = QwenCLSTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        compute_metrics=compute_metrics,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
