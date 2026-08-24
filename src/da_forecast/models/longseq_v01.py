"""Small spatial-temporal Transformer for continuous 15-minute price history."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

import torch
from torch import nn
from torch.nn import functional as F


class _StateConditionedEncoderLayer(nn.Module):
    """Transformer encoder layer with zero-initialized AdaLN conditioning."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, state_dim: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu
        self.state_affine1 = nn.Linear(state_dim, 2 * d_model)
        self.state_affine2 = nn.Linear(state_dim, 2 * d_model)
        nn.init.zeros_(self.state_affine1.weight)
        nn.init.zeros_(self.state_affine1.bias)
        nn.init.zeros_(self.state_affine2.weight)
        nn.init.zeros_(self.state_affine2.bias)

    @staticmethod
    def _condition(normed: torch.Tensor, state: torch.Tensor, affine: nn.Linear) -> torch.Tensor:
        scale, bias = affine(state).chunk(2, dim=-1)
        return normed * (1.0 + scale.unsqueeze(1)) + bias.unsqueeze(1)

    def forward(self, src: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        attended, _ = self.self_attn(src, src, src, need_weights=False)
        src = src + self.dropout1(attended)
        src = self._condition(self.norm1(src), state, self.state_affine1)
        feedforward = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(feedforward)
        return self._condition(self.norm2(src), state, self.state_affine2)


class _StateConditionedEncoder(nn.Module):
    def __init__(self, layer: _StateConditionedEncoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([layer if idx == 0 else _StateConditionedEncoderLayer(
            layer.self_attn.embed_dim,
            layer.self_attn.num_heads,
            layer.linear1.out_features,
            layer.dropout.p,
            layer.state_affine1.in_features,
        ) for idx in range(num_layers)])

    def forward(self, src: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, state)
        return src


@dataclass
class EarlyStopping:
    """Track strict validation-loss improvements for one rolling training fold."""

    patience: int = 5
    best_validation_loss: float = inf
    best_epoch: int = 0
    stale_epochs: int = 0
    last_improved: bool = False

    def update(self, *, epoch: int, validation_loss: float) -> bool:
        """Record a validation loss and return whether training must stop."""
        if validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss
            self.best_epoch = epoch
            self.stale_epochs = 0
            self.last_improved = True
            return False
        self.stale_epochs += 1
        self.last_improved = False
        return self.stale_epochs >= self.patience


def _relative_sinusoidal_position_encoding(
    length: int, dimension: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Encode patch positions relative to the history cutoff, with the newest patch at zero."""
    positions = torch.arange(length, device=device, dtype=torch.float32) - float(length - 1)
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10_000.0, device=device)) / dimension)
    )
    encoding = torch.zeros(length, dimension, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions.unsqueeze(1) * frequencies)
    encoding[:, 1::2] = torch.cos(positions.unsqueeze(1) * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype)


class SpatialTemporalTransformer(nn.Module):
    """Encode continuous price history and decode one D+1 96-slot distribution."""

    def __init__(
        self,
        *,
        station_variable_dim: int,
        history_extra_dim: int,
        target_extra_dim: int,
        n_stations: int,
        d_model: int = 64,
        nhead: int = 4,
        patch_size: int = 4,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.2,
        enable_retrieval: bool = False,
        conditioning: str = "none",
        state_dim: int = 5,
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        if dim_feedforward is None:
            dim_feedforward = d_model * 2
        if dim_feedforward < 1:
            raise ValueError("dim_feedforward must be positive")
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        if conditioning not in {"none", "film", "adaln"}:
            raise ValueError("conditioning must be one of: none, film, adaln")
        if state_dim < 1:
            raise ValueError("state_dim must be positive")
        self.patch_size = patch_size
        self.n_stations = n_stations
        self.enable_retrieval = enable_retrieval
        self.conditioning = conditioning
        self.state_dim = state_dim
        self.station_encoder = nn.Sequential(
            nn.Linear(station_variable_dim, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.station_gate = nn.Linear(d_model, 1)
        history_dim = 1 + history_extra_dim + d_model
        target_dim = target_extra_dim + d_model
        self.history_patch = nn.Linear(patch_size * history_dim, d_model)
        self.target_projection = nn.Linear(target_dim, d_model)
        if conditioning == "adaln":
            encoder_layer = _StateConditionedEncoderLayer(d_model, nhead, dim_feedforward, dropout, state_dim)
            self.history_encoder = _StateConditionedEncoder(encoder_layer, num_layers)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
            )
            self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        if conditioning == "film":
            self.film_norm = nn.LayerNorm(d_model)
            self.film_affine = nn.Linear(state_dim, 2 * d_model)
            nn.init.zeros_(self.film_affine.weight)
            nn.init.zeros_(self.film_affine.bias)
        if enable_retrieval:
            self.retrieval_patch = nn.Linear(patch_size, d_model)
            self.retrieval_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.retrieval_gate = nn.Linear(d_model * 2, 1)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.decoder_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model)
        )
        self.point_head = nn.Linear(d_model, 1)
        self.negative_head = nn.Linear(d_model, 1)
        self.quantile_head = nn.Linear(d_model, 3)

    def _spatial_pool(self, station_weather: torch.Tensor) -> torch.Tensor:
        if station_weather.ndim != 4 or station_weather.shape[2] != self.n_stations:
            raise ValueError("station_weather must have shape [batch, time, n_stations, variables]")
        encoded = self.station_encoder(station_weather)
        weights = torch.softmax(self.station_gate(encoded).squeeze(-1), dim=2)
        return (weights.unsqueeze(-1) * encoded).sum(dim=2)

    def forward(
        self,
        *,
        history_price: torch.Tensor,
        history_extra: torch.Tensor,
        history_station_weather: torch.Tensor,
        target_extra: torch.Tensor,
        target_station_weather: torch.Tensor,
        state_features: torch.Tensor | None = None,
        retrieval_prices: torch.Tensor | None = None,
        retrieval_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if history_price.ndim != 3 or history_price.shape[-1] != 1:
            raise ValueError("history_price must have shape [batch, time, 1]")
        if history_price.shape[1] % self.patch_size:
            raise ValueError("history length must be divisible by patch_size")
        if self.conditioning != "none":
            if state_features is None or state_features.ndim != 2 or state_features.shape[1] != self.state_dim:
                raise ValueError(f"state_features must have shape [batch, {self.state_dim}] for {self.conditioning}")
        history_spatial = self._spatial_pool(history_station_weather)
        target_spatial = self._spatial_pool(target_station_weather)
        history = torch.cat([history_price, history_extra, history_spatial], dim=-1)
        batch, length, channels = history.shape
        patches = history.reshape(batch, length // self.patch_size, self.patch_size * channels)
        history_tokens = self.history_patch(patches)
        history_tokens = history_tokens + _relative_sinusoidal_position_encoding(
            history_tokens.shape[1],
            history_tokens.shape[2],
            device=history_tokens.device,
            dtype=history_tokens.dtype,
        ).unsqueeze(0)
        memory = self.history_encoder(history_tokens, state_features) if self.conditioning == "adaln" else self.history_encoder(history_tokens)
        query = self.target_projection(torch.cat([target_extra, target_spatial], dim=-1))
        attended, _ = self.cross_attention(query, memory, memory, need_weights=False)
        decoded = self.decoder_norm(query + attended)
        if self.conditioning == "film":
            scale, bias = self.film_affine(state_features).chunk(2, dim=-1)
            decoded = self.film_norm(decoded) * (1.0 + scale.unsqueeze(1)) + bias.unsqueeze(1)
        if retrieval_prices is not None:
            if not self.enable_retrieval:
                raise ValueError("retrieval inputs require enable_retrieval=True")
            if retrieval_prices.ndim != 3 or retrieval_prices.shape[-1] != 96:
                raise ValueError("retrieval_prices must have shape [batch, top_k, 96]")
            if retrieval_weights is None or retrieval_weights.shape != retrieval_prices.shape[:2]:
                raise ValueError("retrieval_weights must have shape [batch, top_k]")
            if torch.any(retrieval_weights < 0):
                raise ValueError("retrieval_weights must be non-negative")
            batch_size, top_k, _ = retrieval_prices.shape
            retrieval_tokens = self.retrieval_patch(
                retrieval_prices.reshape(batch_size * top_k, 96 // self.patch_size, self.patch_size)
            ).reshape(batch_size, top_k * (96 // self.patch_size), -1)
            # Similarity weights scale memory values, while the gate decides
            # per target slot whether retrieval is useful at all.
            retrieval_tokens = retrieval_tokens * retrieval_weights.repeat_interleave(96 // self.patch_size, dim=1).unsqueeze(-1)
            retrieval_attended, _ = self.retrieval_attention(query, retrieval_tokens, retrieval_tokens, need_weights=False)
            available = (retrieval_weights.sum(dim=1, keepdim=True) > 0).unsqueeze(-1)
            gate = torch.sigmoid(self.retrieval_gate(torch.cat([decoded, retrieval_attended], dim=-1))) * available
            decoded = decoded + gate * retrieval_attended
        decoded = self.decoder_norm(decoded + self.decoder_ffn(decoded))
        quantiles = torch.sort(self.quantile_head(decoded), dim=-1).values.clamp(-20.0, 20.0)
        point = self.point_head(decoded).squeeze(-1).clamp(-20.0, 20.0)
        negative_logit = self.negative_head(decoded).squeeze(-1).clamp(-20.0, 20.0)
        return {
            "point": point,
            "negative_logit": negative_logit,
            "negative_probability": torch.sigmoid(negative_logit),
            "quantiles": quantiles,
        }


def _pinball_loss(prediction: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    residual = target - prediction
    return torch.maximum(alpha * residual, (alpha - 1.0) * residual)


def _reduce_sample_loss(values: torch.Tensor, sample_weights: torch.Tensor | None) -> torch.Tensor:
    """Average a slot-level loss, optionally emphasizing whole daily samples."""
    if values.ndim < 1:
        return values
    if sample_weights is None:
        return values.mean()
    if sample_weights.ndim != 1 or sample_weights.shape[0] != values.shape[0]:
        raise ValueError("sample_weights must have shape [batch]")
    weights = sample_weights.to(device=values.device, dtype=values.dtype)
    if torch.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("sample_weights must be non-negative and have positive sum")
    per_sample = values.reshape(values.shape[0], -1).mean(dim=1)
    return (per_sample * weights).sum() / weights.sum()


def multitask_loss(
    output: dict[str, torch.Tensor], target: torch.Tensor, *, negative_labels: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Point, quantile and imbalance-aware negative-price training objective."""
    labels = (target < 0).float() if negative_labels is None else negative_labels.float()
    positive = labels.sum()
    negative = labels.numel() - positive
    pos_weight = torch.ones((), device=target.device) if positive == 0 else negative / positive
    point = _reduce_sample_loss(F.smooth_l1_loss(output["point"], target, reduction="none"), sample_weights)
    quantiles = output["quantiles"]
    pinball = sum(
        _reduce_sample_loss(_pinball_loss(quantiles[..., idx], target, alpha), sample_weights)
        for idx, alpha in enumerate((0.1, 0.5, 0.9))
    ) / 3.0
    classification = _reduce_sample_loss(
        F.binary_cross_entropy_with_logits(
            output["negative_logit"], labels, pos_weight=pos_weight, reduction="none"
        ),
        sample_weights,
    )
    return 0.50 * point + 0.35 * pinball + 0.15 * classification
