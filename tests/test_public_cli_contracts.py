from __future__ import annotations

from pathlib import Path

import pytest

from build_qilupulse96_production_bundle_v1 import main as build_bundle_main


def test_bundle_builder_requires_an_explicit_preprocessing_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        build_bundle_main(
            [
                "--checkpoint",
                str(tmp_path / "checkpoint.pt"),
                "--output",
                str(tmp_path / "bundle"),
                "--training-data-snapshot-hash",
                "synthetic-source-hash",
                "--calendar-reference-hash",
                "synthetic-calendar-hash",
            ]
        )

    assert caught.value.code == 2
    assert "provide both --market-data-root and --weather-root" in capsys.readouterr().err
