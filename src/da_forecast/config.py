"""Public configuration for the QiluPulse-96 Shandong production subset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TIMEZONE = "Asia/Shanghai"
PRICE_COL = "price_cny_mwh"
DAYAHEAD_DATATYPE = "day_ahead_prices"
REALTIME_DATATYPE = "realtime_prices"
PRICE_RANGE_MIN = -100.0
PRICE_RANGE_MAX = 1500.0

ZONES = ["SD"]
PRIMARY_ZONES = ["SD"]
ZONE_LABELS = {"SD": "Shandong (山东)"}

SHANDONG_WEATHER_STATIONS = [
    ("SD_JINAN", "济南", 36.65, 117.00),
    ("SD_QINGDAO", "青岛", 36.07, 120.38),
    ("SD_YANTAI", "烟台", 37.46, 121.45),
    ("SD_WEIFANG", "潍坊", 36.71, 119.16),
    ("SD_LINYI", "临沂", 35.05, 118.35),
    ("SD_HEZE", "菏泽", 35.23, 115.48),
    ("SD_DEZHOU", "德州", 37.43, 116.36),
    ("SD_DONGYING", "东营", 37.43, 118.67),
    ("SD_JINING", "济宁", 35.41, 116.59),
    ("SD_WEIHAI", "威海", 37.51, 122.12),
]


@dataclass(frozen=True)
class SpatialWeatherStation:
    """One representative point for a Shandong prefecture-level city."""

    code: str
    city: str
    latitude: float
    longitude: float
    altitude_m: float


SHANDONG_SPATIAL_STATIONS = (
    SpatialWeatherStation("SD_JINAN", "济南", 36.65, 117.00, 50.0),
    SpatialWeatherStation("SD_QINGDAO", "青岛", 36.07, 120.38, 30.0),
    SpatialWeatherStation("SD_ZIBO", "淄博", 36.81, 118.05, 35.0),
    SpatialWeatherStation("SD_ZAOZHUANG", "枣庄", 34.81, 117.32, 70.0),
    SpatialWeatherStation("SD_DONGYING", "东营", 37.43, 118.67, 10.0),
    SpatialWeatherStation("SD_YANTAI", "烟台", 37.46, 121.45, 20.0),
    SpatialWeatherStation("SD_WEIFANG", "潍坊", 36.71, 119.16, 30.0),
    SpatialWeatherStation("SD_JINING", "济宁", 35.41, 116.59, 35.0),
    SpatialWeatherStation("SD_TAIAN", "泰安", 36.20, 117.09, 130.0),
    SpatialWeatherStation("SD_WEIHAI", "威海", 37.51, 122.12, 15.0),
    SpatialWeatherStation("SD_RIZHAO", "日照", 35.42, 119.52, 20.0),
    SpatialWeatherStation("SD_LINYI", "临沂", 35.05, 118.35, 70.0),
    SpatialWeatherStation("SD_DEZHOU", "德州", 37.43, 116.36, 22.0),
    SpatialWeatherStation("SD_LIAOCHENG", "聊城", 36.46, 115.98, 35.0),
    SpatialWeatherStation("SD_BINZHOU", "滨州", 37.38, 117.97, 15.0),
    SpatialWeatherStation("SD_HEZE", "菏泽", 35.23, 115.48, 50.0),
)

DEFAULT_THRESHOLD_CNY = 5.0
DEFAULT_MAX_DAILY_TRADES = 12
MAX_IMPUTATION_PCT = 5.0
MIN_COMPLETENESS_PCT = 90.0
API_MAX_RETRIES = 3
API_BACKOFF_SECONDS = [2, 4, 8]
