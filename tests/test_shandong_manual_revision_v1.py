import pandas as pd

from da_forecast.production.manual_revision_v1 import ManualRevisionStore


def test_manual_revision_is_append_only_and_imputation_is_causal(tmp_path):
    store = ManualRevisionStore(tmp_path)
    frame = pd.DataFrame({"timestamp": pd.date_range("2026-08-01", periods=2, freq="15min"), "value": [1.0, 2.0]})
    revision = store.save(frame, market_date="2026-08-01", source_kind="realtime", operator_note="人工确认")
    assert revision.data_path.is_file()
    assert revision.manifest_path.is_file()
    values = pd.Series([10.0, 20.0], index=pd.to_datetime(["2026-07-25", "2026-07-18"]))
    result = store.imputation_suggestions(values, missing_timestamps=pd.DatetimeIndex(["2026-08-01"]), cutoff=pd.Timestamp("2026-07-30", tz="Asia/Shanghai"))
    assert result.loc[0, "status"] == "available"
    assert result.loc[0, "suggestion"] == 15.0
