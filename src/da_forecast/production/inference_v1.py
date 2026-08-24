"""Model-only inference for one already validated causal input bundle."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from da_forecast.models.qilupulse96_v1 import normalized_output_frame
from .bundle_v1 import QiluPulse96ProductionBundle
from .input_builder_v1 import CausalInputBundle


def infer_qilupulse96(bundle: QiluPulse96ProductionBundle, inputs: CausalInputBundle, *, device: str = "cpu"):
    model = bundle.model.to(torch.device(device))
    model.eval()
    tensors = {
        "history_price": torch.from_numpy(inputs.history_price[None, ...]).to(device),
        "history_extra": torch.from_numpy(inputs.history_extra[None, ...]).to(device),
        "history_station_weather": torch.from_numpy(inputs.history_station_weather[None, ...]).to(device),
        "target_extra": torch.from_numpy(inputs.target_extra[None, ...]).to(device),
        "target_station_weather": torch.from_numpy(inputs.target_station_weather[None, ...]).to(device),
        "state_features": torch.from_numpy(inputs.state_features[None, ...]).to(device),
    }
    with torch.no_grad():
        output = model(**tensors)
    return normalized_output_frame(
        point=output["point"].detach().cpu().numpy()[0],
        negative_probability=output["negative_probability"].detach().cpu().numpy()[0],
        quantiles=output["quantiles"].detach().cpu().numpy()[0],
        normalization_center=inputs.normalization_center,
        normalization_scale=inputs.normalization_scale,
        target_date=inputs.target_date,
        as_of=inputs.target_date - pd.Timedelta(days=1) + pd.Timedelta(hours=12),
        parameter_checksum=bundle.parameter_checksum,
    )
