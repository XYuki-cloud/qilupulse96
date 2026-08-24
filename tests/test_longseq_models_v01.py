from __future__ import annotations

import torch

from da_forecast.models.longseq_v01 import EarlyStopping, SpatialTemporalTransformer, multitask_loss


def test_spatial_temporal_transformer_returns_96_valid_distribution_outputs() -> None:
    torch.manual_seed(7)
    model = SpatialTemporalTransformer(
        station_variable_dim=5,
        history_extra_dim=3,
        target_extra_dim=4,
        n_stations=16,
        d_model=32,
        nhead=4,
        patch_size=4,
        num_layers=2,
    )
    output = model(
        history_price=torch.randn(2, 32, 1),
        history_extra=torch.randn(2, 32, 3),
        history_station_weather=torch.randn(2, 32, 16, 5),
        target_extra=torch.randn(2, 96, 4),
        target_station_weather=torch.randn(2, 96, 16, 5),
    )

    assert output["point"].shape == (2, 96)
    assert output["negative_probability"].shape == (2, 96)
    assert output["quantiles"].shape == (2, 96, 3)
    assert torch.all((output["negative_probability"] >= 0) & (output["negative_probability"] <= 1))
    assert torch.all(output["quantiles"][:, :, 0] <= output["quantiles"][:, :, 1])
    assert torch.all(output["quantiles"][:, :, 1] <= output["quantiles"][:, :, 2])


def test_multitask_loss_is_finite_for_negative_and_nonnegative_prices() -> None:
    target = torch.tensor([[-20.0, 30.0]])
    output = {
        "point": torch.tensor([[-10.0, 20.0]]),
        "negative_logit": torch.tensor([[1.0, -1.0]]),
        "quantiles": torch.tensor([[[-30.0, -10.0, 5.0], [10.0, 20.0, 40.0]]]),
    }

    loss = multitask_loss(output, target)

    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_multitask_loss_accepts_per_sample_training_weights() -> None:
    output = {
        "point": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        "negative_logit": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        "quantiles": torch.zeros(2, 2, 3),
    }
    target = torch.tensor([[0.0, 0.0], [100.0, 100.0]])

    uniform = multitask_loss(output, target, sample_weights=torch.tensor([1.0, 1.0]))
    recent_emphasis = multitask_loss(output, target, sample_weights=torch.tensor([0.01, 1.0]))

    assert torch.isfinite(recent_emphasis)
    assert recent_emphasis > uniform


def test_history_patch_order_changes_position_aware_transformer_output() -> None:
    torch.manual_seed(7)
    model = SpatialTemporalTransformer(
        station_variable_dim=3,
        history_extra_dim=2,
        target_extra_dim=2,
        n_stations=4,
        d_model=16,
        nhead=4,
        patch_size=4,
        num_layers=1,
    ).eval()
    inputs = {
        "history_price": torch.randn(1, 32, 1),
        "history_extra": torch.randn(1, 32, 2),
        "history_station_weather": torch.randn(1, 32, 4, 3),
        "target_extra": torch.randn(1, 96, 2),
        "target_station_weather": torch.randn(1, 96, 4, 3),
    }

    original = model(**inputs)["point"]
    patch_order = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0])
    reordered = dict(inputs)
    for name in ("history_price", "history_extra", "history_station_weather"):
        value = inputs[name]
        reordered[name] = value.reshape(1, 8, 4, *value.shape[2:])[:, patch_order].reshape_as(value)

    changed = model(**reordered)["point"]

    assert not torch.allclose(original, changed, atol=1e-5, rtol=0.0)


@torch.no_grad()
def test_transformer_accepts_30_60_and_90_day_history_lengths() -> None:
    model = SpatialTemporalTransformer(
        station_variable_dim=2,
        history_extra_dim=2,
        target_extra_dim=2,
        n_stations=2,
        d_model=8,
        nhead=2,
        patch_size=4,
        num_layers=1,
    ).eval()

    for context_days in (30, 60, 90):
        length = context_days * 96
        output = model(
            history_price=torch.randn(1, length, 1),
            history_extra=torch.randn(1, length, 2),
            history_station_weather=torch.randn(1, length, 2, 2),
            target_extra=torch.randn(1, 96, 2),
            target_station_weather=torch.randn(1, 96, 2, 2),
        )
        assert output["point"].shape == (1, 96)


def test_early_stopping_uses_strict_improvement_and_default_five_epoch_patience() -> None:
    stopping = EarlyStopping()

    assert stopping.update(epoch=1, validation_loss=1.0) is False
    assert stopping.update(epoch=2, validation_loss=0.8) is False
    assert stopping.update(epoch=3, validation_loss=0.8) is False
    assert stopping.update(epoch=4, validation_loss=0.81) is False
    assert stopping.update(epoch=5, validation_loss=0.82) is False
    assert stopping.update(epoch=6, validation_loss=0.83) is False
    assert stopping.update(epoch=7, validation_loss=0.84) is True
    assert stopping.best_epoch == 2
    assert stopping.best_validation_loss == 0.8


def test_capacity_arguments_change_depth_width_and_keep_retrieval_disabled_by_default() -> None:
    base = SpatialTemporalTransformer(
        station_variable_dim=5, history_extra_dim=3, target_extra_dim=4,
        n_stations=16, d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
    )
    deep = SpatialTemporalTransformer(
        station_variable_dim=5, history_extra_dim=3, target_extra_dim=4,
        n_stations=16, d_model=64, nhead=4, num_layers=4, dim_feedforward=256,
    )
    wide = SpatialTemporalTransformer(
        station_variable_dim=5, history_extra_dim=3, target_extra_dim=4,
        n_stations=16, d_model=128, nhead=8, num_layers=2, dim_feedforward=256,
    )
    assert not base.enable_retrieval
    assert not hasattr(base, "retrieval_attention")
    assert sum(parameter.numel() for parameter in deep.parameters()) > sum(parameter.numel() for parameter in base.parameters())
    assert sum(parameter.numel() for parameter in wide.parameters()) > sum(parameter.numel() for parameter in base.parameters())
    assert deep.history_encoder.layers[0].linear1.out_features == 256


def test_d_model_must_be_divisible_by_attention_heads() -> None:
    try:
        SpatialTemporalTransformer(
            station_variable_dim=2, history_extra_dim=2, target_extra_dim=2,
            n_stations=2, d_model=10, nhead=4,
        )
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("invalid d_model/nhead pair was accepted")


def _conditioning_inputs() -> dict[str, torch.Tensor]:
    return {
        "history_price": torch.randn(1, 32, 1),
        "history_extra": torch.randn(1, 32, 2),
        "history_station_weather": torch.randn(1, 32, 4, 3),
        "target_extra": torch.randn(1, 96, 2),
        "target_station_weather": torch.randn(1, 96, 4, 3),
        "state_features": torch.randn(1, 5),
    }


def test_film_and_adaln_require_and_use_state_features() -> None:
    for conditioning in ("film", "adaln"):
        torch.manual_seed(7)
        model = SpatialTemporalTransformer(
            station_variable_dim=3, history_extra_dim=2, target_extra_dim=2,
            n_stations=4, d_model=16, nhead=4, patch_size=4, num_layers=2,
            conditioning=conditioning,
        ).eval()
        with torch.no_grad():
            if conditioning == "film":
                model.film_affine.weight.normal_(0.0, 0.05)
            else:
                for layer in model.history_encoder.layers:
                    layer.state_affine1.weight.normal_(0.0, 0.05)
                    layer.state_affine2.weight.normal_(0.0, 0.05)
        inputs = _conditioning_inputs()
        first = model(**inputs)["point"]
        changed = dict(inputs, state_features=inputs["state_features"] + 1.0)
        second = model(**changed)["point"]
        assert not torch.allclose(first, second, atol=1e-6, rtol=0.0)
        missing = dict(inputs)
        missing.pop("state_features")
        try:
            model(**missing)
        except ValueError as error:
            assert "state_features" in str(error)
        else:
            raise AssertionError(f"{conditioning} accepted missing state features")
