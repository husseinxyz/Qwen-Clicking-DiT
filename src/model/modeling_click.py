"""
Click Prediction Model with DiT Action Head using Flow Matching.

Based on NVIDIA's GR00T architecture:
- ActionEncoder: Embeds noisy actions + timestep using sinusoidal encoding
- DiT: Diffusion Transformer with AdaLayerNorm for timestep conditioning
- ActionDecoder: MLP to decode DiT output back to action space
- Flow matching loss: MSE between predicted and target velocity
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List

from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLModel
)
from train.monkey_patch_vision import replace_qwen2_5_vision

replace_qwen2_5_vision()


# ============================================================================
# Flow Matching Components (from GR00T architecture)
# ============================================================================

def swish(x: torch.Tensor) -> torch.Tensor:
    """Swish activation function (SiLU)."""
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        squeeze_output = False
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(1)
            squeeze_output = True

        timesteps = timesteps.float()
        device = timesteps.device

        half_dim = self.dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            math.log(10000.0) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        enc = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)

        if squeeze_output:
            enc = enc.squeeze(1)

        return enc


class TimestepEmbedder(nn.Module):
    """Timestep embedder with sinusoidal encoding + MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.pos_encoding = SinusoidalPositionalEncoding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.pos_encoding(timesteps)
        emb = emb.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(emb)


class ActionEncoder(nn.Module):
    """Action encoder matching GR00T's MultiEmbodimentActionEncoder."""

    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(action_dim, hidden_size)
        self.W2 = nn.Linear(2 * hidden_size, hidden_size)
        self.W3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        squeeze_output = False
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
            squeeze_output = True

        B, T, _ = actions.shape
        timesteps_expanded = timesteps.unsqueeze(1).expand(-1, T)

        a_emb = self.W1(actions)
        tau_emb = self.pos_encoding(timesteps_expanded).to(dtype=a_emb.dtype)

        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x))
        x = self.W3(x)

        return x


class ActionDecoder(nn.Module):
    """Action decoder - two-layer MLP."""

    def __init__(self, hidden_size: int, action_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.layer1(x))
        return self.layer2(hidden)


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization conditioned on timestep embedding."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Diffusion Transformer block with cross-attention to VLM features."""

    def __init__(self, dim: int, num_heads: int, cross_attention_dim: int, dropout: float = 0.0):
        super().__init__()
        self.ada_ln = AdaLayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=cross_attention_dim,
            vdim=cross_attention_dim,
        )
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        temb: torch.Tensor,
        cond_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.ada_ln(x, temb)
        h, _ = self.cross_attn(
            h, cond, cond,
            key_padding_mask=cond_key_padding_mask,
            need_weights=False,
        )
        x = x + h
        h = self.ff(self.ff_norm(x))
        x = x + h
        return x


class DiT(nn.Module):
    """Diffusion Transformer for flow matching."""

    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        cross_attention_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.timestep_embedder = TimestepEmbedder(dim)
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, cross_attention_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj_out_1 = nn.Linear(dim, dim * 2)
        self.proj_out_2 = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        timesteps: torch.Tensor,
        cond_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = self.timestep_embedder(timesteps)

        for block in self.blocks:
            x = block(x, cond, temb, cond_key_padding_mask)

        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=-1)
        x = self.norm_out(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.proj_out_2(x)

        return x


# ============================================================================
# Click Action Head with Flow Matching
# ============================================================================

class ClickActionHead(nn.Module):
    """
    Action head for click prediction using flow matching.
    Based on GR00T's Gr00tN1d6ActionHead architecture.
    """

    def __init__(
        self,
        backbone_dim: int,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        action_dim: int = 2,
        dropout: float = 0.1,
        num_timestep_buckets: int = 1000,
        noise_s: float = 0.999,
        beta_alpha: float = 1.5,
        beta_beta: float = 1.0,
        num_inference_steps: int = 16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.num_timestep_buckets = num_timestep_buckets
        self.noise_s = noise_s
        self.num_inference_steps = num_inference_steps

        # Projection from backbone dim to hidden size
        self.cond_proj = nn.Linear(backbone_dim, hidden_size)

        # GR00T-style components
        self.action_encoder = ActionEncoder(action_dim, hidden_size)
        self.dit = DiT(
            dim=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            cross_attention_dim=hidden_size,
            dropout=dropout,
        )
        self.action_decoder = ActionDecoder(hidden_size, action_dim)

        # Flow matching noise distribution
        self.register_buffer('beta_alpha', torch.tensor(beta_alpha))
        self.register_buffer('beta_beta', torch.tensor(beta_beta))

    def _sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Sample timesteps from Beta distribution."""
        # Beta/Dirichlet sampling requires float32, then convert to target dtype
        beta_dist = torch.distributions.Beta(
            self.beta_alpha.float(),
            self.beta_beta.float()
        )
        sample = beta_dist.sample((batch_size,)).to(device=device, dtype=dtype)
        return (1 - sample) * self.noise_s

    def forward(
        self,
        cond_tokens: torch.Tensor,
        cond_mask: Optional[torch.Tensor],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass with flow matching loss.

        Args:
            cond_tokens: (B, S, backbone_dim) VLM hidden states
            cond_mask: (B, S) attention mask (1 = attend, 0 = ignore)
            actions: (B, 2) target click coordinates in [0, 1]

        Returns:
            loss: flow matching MSE loss
            pred_velocity: (B, 2) predicted velocity
        """
        actions = actions.unsqueeze(1) if actions.dim() == 2 else actions

        # Sample noise and timesteps
        noise = torch.randn_like(actions)
        t = self._sample_time(actions.shape[0], actions.device, actions.dtype)
        t_broadcast = t[:, None, None]

        # Flow matching interpolation
        noisy_actions = (1 - t_broadcast) * noise + t_broadcast * actions
        velocity = actions - noise

        # Discretize timesteps
        t_discretized = (t * self.num_timestep_buckets).long()

        # Project conditioning
        cond_tokens = self.cond_proj(cond_tokens)

        # Encode noisy actions
        action_tokens = self.action_encoder(noisy_actions.squeeze(1), t_discretized)

        # Attention mask
        cond_key_padding_mask = None
        if cond_mask is not None:
            cond_key_padding_mask = ~cond_mask.bool()

        # DiT forward
        dit_output = self.dit(
            action_tokens,
            cond_tokens,
            t_discretized,
            cond_key_padding_mask,
        )

        # Decode
        pred_velocity = self.action_decoder(dit_output)

        # Flow matching loss
        loss = F.mse_loss(pred_velocity, velocity)

        return loss, pred_velocity.squeeze(1)

    @torch.inference_mode()
    def sample(
        self,
        cond_tokens: torch.Tensor,
        cond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate actions using Euler integration."""
        batch_size = cond_tokens.shape[0]
        device = cond_tokens.device
        dtype = cond_tokens.dtype

        # Initialize from noise
        actions = torch.randn(
            size=(batch_size, self.action_dim),
            device=device,
            dtype=dtype,
        )

        # Project conditioning
        cond_tokens = self.cond_proj(cond_tokens)

        # Attention mask
        cond_key_padding_mask = None
        if cond_mask is not None:
            cond_key_padding_mask = ~cond_mask.bool()

        # Euler integration
        dt = 1.0 / self.num_inference_steps

        for step in range(self.num_inference_steps):
            t_cont = step / float(self.num_inference_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            t_tensor = torch.full((batch_size,), t_discretized, device=device)

            action_tokens = self.action_encoder(actions, t_tensor)
            dit_output = self.dit(
                action_tokens,
                cond_tokens,
                t_tensor,
                cond_key_padding_mask,
            )
            pred_velocity = self.action_decoder(dit_output).squeeze(1)
            actions = actions + dt * pred_velocity

        return actions


# ============================================================================
# Output Dataclass
# ============================================================================

@dataclass
class ClickPredictionOutput(ModelOutput):
    """Output for click prediction model."""
    loss: Optional[torch.FloatTensor] = None
    click_xy: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# ============================================================================
# Main Model Class
# ============================================================================

class Qwen2_5_VLForClickPrediction(Qwen2_5_VLPreTrainedModel):
    """
    Qwen2.5-VL model with DiT action head for click prediction using flow matching.

    The model uses:
    - Frozen Qwen2.5-VL backbone for visual-language understanding
    - Trainable DiT action head that predicts click coordinates via flow matching
    """
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)

        # Get head configuration from config
        hidden_size = getattr(config, 'dit_hidden_size', 512)
        num_layers = getattr(config, 'dit_num_layers', 6)
        num_heads = getattr(config, 'dit_num_heads', 8)
        dropout = getattr(config, 'dit_dropout', 0.1)
        num_inference_steps = getattr(config, 'num_inference_steps', 16)

        # Base model
        self.model = Qwen2_5_VLModel(config)

        # DiT action head
        self.action_head = ClickActionHead(
            backbone_dim=config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            action_dim=2,
            dropout=dropout,
            num_inference_steps=num_inference_steps,
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,  # (batch, 2) - target (x, y)
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, ClickPredictionOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Forward through base model
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        # Get hidden states as conditioning for DiT
        hidden_states = outputs.last_hidden_state

        loss = None
        click_xy = None

        if labels is not None:
            # Training mode: compute flow matching loss
            labels = labels.to(hidden_states.device, dtype=hidden_states.dtype)
            loss, pred_velocity = self.action_head(
                cond_tokens=hidden_states,
                cond_mask=attention_mask,
                actions=labels,
            )
            # For evaluation metrics, we could run sampling, but it's expensive
            # So we just return None for click_xy during training
            click_xy = None
        else:
            # Inference mode: sample click coordinates
            click_xy = self.action_head.sample(
                cond_tokens=hidden_states,
                cond_mask=attention_mask,
            )

        return ClickPredictionOutput(
            loss=loss,
            click_xy=click_xy,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.inference_mode()
    def predict(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Convenience method for inference.

        Returns:
            (B, 2) predicted click coordinates in [0, 1]
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=None,
            **kwargs,
        )
        return outputs.click_xy
