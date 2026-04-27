"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


import sys

def is_bundled() -> bool:
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')

def get_resource_path(relative_path: str) -> Path:
    if is_bundled():
        return Path(sys._MEIPASS) / relative_path
    return Path(__file__).resolve().parent.parent / relative_path

def get_user_data_dir(subfolder: str = "") -> Path:
    base = Path.home() / "Library" / "Application Support" / "Stock Predictor"
    if subfolder:
        base = base / subfolder
    base.mkdir(parents=True, exist_ok=True)
    return base

def get_logs_dir() -> Path:
    base = Path.home() / "Library" / "Logs" / "Stock Predictor"
    base.mkdir(parents=True, exist_ok=True)
    return base

PROJECT_ROOT = get_resource_path("")
BASE_DIR = PROJECT_ROOT / "stock_predictor"

if is_bundled():
    ARTIFACT_DIR = get_user_data_dir("artifacts")
    REPORT_DIR = get_user_data_dir("reports")
    MODEL_DIR = get_user_data_dir("models")
    LOG_DIR = get_logs_dir()
else:
    ARTIFACT_DIR = BASE_DIR / "artifacts"
    REPORT_DIR = BASE_DIR.parent / "reports"
    MODEL_DIR = BASE_DIR / "models"
    LOG_DIR = ARTIFACT_DIR

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MISSING_SIGNAL_VALUE = 50.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()

SMALL_CAP_UNIVERSE: List[str] = [
    "PLTR", "SOFI", "RKLB", "IONQ", "UPST", "AFRM", "HOOD", "CAVA", "APP", "ASTS",
    "CRDO", "SMCI", "LUNR", "HIMS", "RBLX", "DLO", "NU", "CELH", "IOT", "DUOL",
    "TTAN", "SOUN", "JOBY", "ACHR", "AI", "BBAI", "GCT", "MARA", "RIOT", "CLSK",
    "FUBO", "OPEN", "RUN", "CHWY", "GME", "LCID", "RIVN", "S", "U", "DOCN",
    "MQ", "WULF", "ENVX", "ARRY", "ONON", "BILL", "MNDY", "FRSH", "CVNA", "YOU",
]

MOST_SHORTED_UNIVERSE: List[str] = [
    "CVNA", "UPST", "AI", "BYND", "LCID", "RIVN", "SOUN", "PLUG", "MARA",
    "RIOT", "SMCI", "AFRM", "SOFI", "FUBO", "GME", "OPEN", "APP", "HIMS", "RKLB",
]


SECTOR_UNIVERSE: Dict[str, List[str]] = {
    "Technology": [
        "AAPL",
        "MSFT",
        "NVDA",
        "AVGO",
        "ORCL",
        "AMD",
        "ANET",
        "CRM",
        "PLTR",
        "SNOW",
    ],
    "Communication Services": [
        "META",
        "GOOGL",
        "NFLX",
        "TTD",
        "SPOT",
    ],
    "Consumer Discretionary": [
        "AMZN",
        "TSLA",
        "COST",
        "LULU",
        "HD",
        "NKE",
    ],
    "Consumer Staples": [
        "WMT",
        "PG",
        "KO",
        "PEP",
        "MDLZ",
    ],
    "Financials": [
        "JPM",
        "GS",
        "MS",
        "AXP",
        "SCHW",
        "CME",
    ],
    "Healthcare": [
        "LLY",
        "UNH",
        "JNJ",
        "MRK",
        "ABBV",
        "ISRG",
    ],
    "Industrials": [
        "GE",
        "ETN",
        "CAT",
        "DE",
        "HWM",
        "URI",
    ],
    "Energy": [
        "XOM",
        "CVX",
        "SLB",
        "BKR",
        "LNG",
        "FANG",
    ],
    "Utilities": [
        "NEE",
        "DUK",
        "SO",
        "CEG",
        "VST",
        "NRG",
    ],
    "Real Estate": [
        "PLD",
        "AMT",
        "EQIX",
        "O",
    ],
    "Materials": [
        "LIN",
        "NUE",
        "FCX",
        "MOS",
    ],
    "Aerospace & Defense": [
        "RTX",
        "LMT",
        "NOC",
        "GD",
        "LHX",
    ],
}


SECTOR_ETFS = {
    "Technology": "XLK",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}


@dataclass(slots=True)
class Thresholds:
    final_score_min: float = 60.0
    minimum_display_score: float = 40.0
    defaulted_sentiment_penalty_points: float = 1.5
    defaulted_flow_penalty_points: float = 2.0
    multiple_defaulted_signals_penalty_points: float = 1.5
    defaulted_signal_uncertainty_points: float = 1.5
    options_block_notional: float = 500_000.0
    sentiment_velocity_bullish: float = 3.0
    vix_risk_off: float = 28.0
    rs_min: float = 85.0
    pattern_min: float = 7.0
    min_price: float = 5.0
    min_average_volume: int = 500_000
    max_correlation: float = 0.75
    min_reward_to_risk: float = _env_float("APP_MIN_RR_RATIO", 2.0)
    stop_atr_multiplier: float = 1.5
    minimum_data_quality_score: float = 0.6
    low_confidence_spread: float = 0.20
    high_confidence_spread: float = 0.10
    kill_switch_vix: float = _env_float("APP_VIX_CUTOFF", 30.0)
    short_squeeze_score_boost_threshold: float = 70.0
    short_squeeze_bonus_points: float = 12.0
    congress_bonus_points: float = 8.0
    anomaly_boost_multiplier: float = 1.15
    anomaly_warning_multiplier: float = 0.9
    smart_money_reject_sentiment: float = 0.65
    smart_money_reject_threshold: float = 0.45
    smart_money_strong_threshold: float = 0.65
    pattern_min_win_rate: float = 0.60
    multi_timeframe_partial_penalty: float = 0.80
    risk_on_threshold: float = 58.0
    neutral_threshold: float = _env_float("APP_MIN_SCORE_THRESHOLD", 54.0)
    risk_off_threshold: float = 65.0
    minimum_returned_picks: int = 5
    suggestion_threshold_step: float = 5.0
    cold_start_min_samples: int = 50
    persistent_momentum_bonus_points: float = 5.0
    hot_sector_bonus_points: float = 8.0
    cold_sector_penalty_points: float = 5.0
    pre_earnings_runner_bonus_points: float = 12.0
    confirmed_breakout_bonus_points: float = 10.0
    unusual_early_volume_bonus_points: float = 8.0
    float_rotation_bonus_points: float = 6.0


@dataclass(slots=True)
class BacktestCosts:
    commission_bps: float = _env_float("BACKTEST_COMMISSION_BPS", 0.5)
    slippage_bps: float = _env_float("BACKTEST_SLIPPAGE_BPS", 8.0)
    spread_bps: float = _env_float("BACKTEST_SPREAD_BPS", 6.0)
    max_position_pct: float = _env_float("BACKTEST_MAX_POSITION_PCT", 0.08)
    turnover_penalty_bps: float = _env_float("BACKTEST_TURNOVER_PENALTY_BPS", 2.0)

    @property
    def round_trip_cost(self) -> float:
        return max(0.0, (self.commission_bps + self.slippage_bps + self.spread_bps + self.turnover_penalty_bps) / 10_000.0)


@dataclass(slots=True)
class CacheTTLs:
    market_history: int = 60 * 60 * 6
    intraday_history: int = 60 * 60 * 6
    info: int = 60 * 60 * 6
    news: int = 60 * 60 * 6
    earnings_dates: int = 60 * 60 * 12
    sec_filings: int = 60 * 60 * 6
    reddit_mentions: int = 60 * 30
    x_mentions: int = 60 * 20
    google_trends: int = 60 * 60
    premium_flow: int = 60 * 20
    congress_trades: int = 60 * 60 * 6
    gpt_reasoning: int = 60 * 60 * 6
    supply_chain: int = 60 * 60 * 24
    macro: int = 60 * 30
    universe_lists: int = 60 * 60 * 24 * 7


@dataclass(slots=True)
class FeatureFlags:
    adaptive_model: bool = _env_bool("ADAPTIVE_MODEL", True)
    online_learning: bool = True
    feature_health_decay: bool = True
    uncertainty_quantification: bool = True
    adaptive_backtest: bool = True
    adaptive_weights: bool = True
    multi_timeframe_confirmation: bool = True
    smart_money_divergence: bool = True
    historical_pattern_win_rate: bool = True
    tft_model: bool = True
    gpt_news_reasoning: bool = True
    anomaly_detection: bool = True
    supply_chain_tracker: bool = True
    congress_tracker: bool = True
    weather_commodity_engine: bool = True
    short_squeeze_probability: bool = True
    dashboard_v2: bool = True
    pdf_report_generator: bool = True
    bot_integration: bool = True
    parallel_processing: bool = True
    data_quality_validator: bool = True
    confidence_interval_display: bool = True
    universe_expansion: bool = True
    momentum_watchlist: bool = True
    sector_etf_signal_booster: bool = True
    auto_training_pipeline: bool = True
    feature_importance_display: bool = True
    walk_forward_backtester: bool = True
    earnings_momentum_detector: bool = True
    breakout_confirmation_engine: bool = True
    relative_volume_alert_system: bool = True
    float_rotation_speed: bool = True
    live_performance_tracker: bool = True
    confidence_score_explainer: bool = True
    pick_quality_tiers: bool = True
    smart_cache_warmup: bool = True
    health_command: bool = True
    two_stage_scan: bool = True
    vectorized_prefilter: bool = True
    regime_specific_model: bool = True
    native_boosters: bool = _env_bool("NATIVE_BOOSTERS", True)
    lightgbm_ensemble: bool = _env_bool("LIGHTGBM_ENSEMBLE", True)
    spy_relative_target: bool = True
    signal_attribution_tracker: bool = True
    threshold_auto_calibrator: bool = True
    startup_banner: bool = True
    config_validation: bool = True
    defensive_stage1_rebalance: bool = True
    inference_alignment: bool = True
    hyperparameter_search: bool = _env_bool("HYPERPARAMETER_SEARCH", True)


@dataclass(slots=True)
class SignalWeights:
    ml: float = 0.30
    technical: float = 0.25
    rs: float = 0.15
    pattern: float = 0.10
    volume: float = 0.10
    sentiment: float = 0.05
    options: float = 0.05

    def as_dict(self) -> Dict[str, float]:
        return {
            "ml": self.ml,
            "technical": self.technical,
            "rs": self.rs,
            "pattern": self.pattern,
            "volume": self.volume,
            "sentiment": self.sentiment,
            "options": self.options,
        }


@dataclass(slots=True)
class AppConfig:
    timezone: str = os.getenv("TZ", "America/New_York")
    cache_db: Path = ARTIFACT_DIR / "cache.sqlite3"
    predictions_log: Path = LOG_DIR / "predictions.log"
    backtest_db: Path = ARTIFACT_DIR / "backtest.db"
    paper_trade_db: Path = ARTIFACT_DIR / "paper_trades.db"
    signal_attribution_db: Path = ARTIFACT_DIR / "signal_attribution.db"
    latest_scan_path: Path = ARTIFACT_DIR / "latest_scan.json"
    latest_report_csv: Path = ARTIFACT_DIR / "latest_report.csv"
    latest_report_md: Path = ARTIFACT_DIR / "latest_report.md"
    latest_report_html: Path = ARTIFACT_DIR / "latest_report.html"
    weekly_report_html: Path = ARTIFACT_DIR / "weekly_report.html"
    latest_single_analysis_path: Path = ARTIFACT_DIR / "latest_single_analysis.json"
    single_analysis_history_path: Path = ARTIFACT_DIR / "single_analysis_history.json"
    model_monitoring_path: Path = ARTIFACT_DIR / "model_monitoring.json"
    latest_report_pdf: Path = REPORT_DIR / "latest.pdf"
    momentum_watchlist_path: Path = ARTIFACT_DIR / "momentum_watchlist.json"
    feature_importance_png: Path = ARTIFACT_DIR / "feature_importance.png"
    backtest_report_path: Path = ARTIFACT_DIR / "backtest_report.md"
    data_quality_log: Path = LOG_DIR / "data_quality.log"
    report_dir: Path = REPORT_DIR
    checkpoint_dir: Path = CHECKPOINT_DIR
    model_dir: Path = MODEL_DIR
    xgb_model_path: Path = MODEL_DIR / "xgboost_model.pkl"
    xgb_metadata_path: Path = MODEL_DIR / "xgboost_metadata.json"
    xgb_calibrator_path: Path = MODEL_DIR / "xgboost_calibrator.pkl"
    xgb_short_model_path: Path = MODEL_DIR / "xgboost_short_horizon.pkl"
    xgb_short_calibrator_path: Path = MODEL_DIR / "xgboost_short_horizon_calibrator.pkl"
    lgbm_model_path: Path = MODEL_DIR / "lightgbm_model.pkl"
    lgbm_metadata_path: Path = MODEL_DIR / "lightgbm_metadata.json"
    lgbm_calibrator_path: Path = MODEL_DIR / "lightgbm_calibrator.pkl"
    xgb_risk_on_path: Path = MODEL_DIR / "xgboost_risk_on.pkl"
    xgb_risk_on_metadata_path: Path = MODEL_DIR / "xgboost_risk_on_metadata.json"
    xgb_neutral_path: Path = MODEL_DIR / "xgboost_neutral.pkl"
    xgb_neutral_metadata_path: Path = MODEL_DIR / "xgboost_neutral_metadata.json"
    xgb_risk_off_path: Path = MODEL_DIR / "xgboost_risk_off.pkl"
    xgb_risk_off_metadata_path: Path = MODEL_DIR / "xgboost_risk_off_metadata.json"
    adaptive_metadata_path: Path = MODEL_DIR / "adaptive_model_metadata.json"
    online_learner_path: Path = MODEL_DIR / "online_learner.pkl"
    feature_health_path: Path = MODEL_DIR / "feature_health.json"
    regime_bull_quiet_path: Path = MODEL_DIR / "regime_bull_quiet.pkl"
    regime_bull_volatile_path: Path = MODEL_DIR / "regime_bull_volatile.pkl"
    regime_neutral_path: Path = MODEL_DIR / "regime_neutral.pkl"
    regime_bear_volatile_path: Path = MODEL_DIR / "regime_bear_volatile.pkl"
    regime_crisis_path: Path = MODEL_DIR / "regime_crisis.pkl"
    default_missing_signal_value: float = DEFAULT_MISSING_SIGNAL_VALUE
    model_trained: bool = False
    default_universe: str = os.getenv("DEFAULT_UNIVERSE", "full")
    reddit_subreddits: List[str] = field(
        default_factory=lambda: ["wallstreetbets", "stocks", "investing"]
    )
    sector_universe: Dict[str, List[str]] = field(default_factory=lambda: SECTOR_UNIVERSE)
    sector_etfs: Dict[str, str] = field(default_factory=lambda: SECTOR_ETFS)
    small_cap_universe: List[str] = field(default_factory=lambda: SMALL_CAP_UNIVERSE)
    most_shorted_universe: List[str] = field(default_factory=lambda: MOST_SHORTED_UNIVERSE)
    benchmark_ticker: str = "SPY"
    breadth_ticker: str = "^SPXA50R"
    vix_ticker: str = "^VIX"
    put_call_ticker: str = ""
    dxy_ticker: str = "DX-Y.NYB"
    hyg_ticker: str = "HYG"
    lqd_ticker: str = "LQD"
    tlt_ticker: str = "TLT"
    ten_year_ticker: str = "^TNX"
    two_year_ticker: str = "^IRX"
    training_lookback_days: int = 1095
    training_history_period: str = "3y"
    training_sample_weekday: int = 0
    training_embargo_days: int = 5
    training_recency_half_life_weeks: int = 26
    training_recency_weight_floor: float = 0.35
    training_profile_recent_folds: int = 4
    training_search_trials: int = _env_int("TRAINING_SEARCH_TRIALS", 36)
    training_search_seed: int = _env_int("TRAINING_SEARCH_SEED", 42)
    training_cv_workers: int = _env_int("TRAINING_CV_WORKERS", 1)
    weekly_profit_target: float = _env_float("WEEKLY_PROFIT_TARGET", 0.06)
    weekly_stop_loss: float = _env_float("WEEKLY_STOP_LOSS", 0.035)
    adaptive_regime_min_samples: int = 180
    adaptive_regime_recent_days: int = 183
    adaptive_uncertainty_models: int = _env_int("ADAPTIVE_UNCERTAINTY_MODELS", 20)
    adaptive_cv_models: int = _env_int("ADAPTIVE_CV_MODELS", 8)
    adaptive_backtest_models: int = _env_int("ADAPTIVE_BACKTEST_MODELS", 4)
    adaptive_uncertainty_warn: float = 0.10
    adaptive_uncertainty_high: float = 0.15
    adaptive_uncertainty_block: float = 0.20
    adaptive_online_blend: float = 0.20
    hourly_interval: str = "60m"
    daily_period: str = "2y"
    intraday_period: str = "365d"
    top_n: int = _env_int("APP_MAX_PICKS", 10)
    stage1_limit: int = 75
    stage1_defensive_slots: int = 16
    stage1_defensive_per_sector: int = 4
    top_sector_count: int = 3
    allow_sector_override_for_diversification: bool = True
    candidate_buffer: int = 30
    max_threads: int = 8
    max_parallel_tickers: int = 15
    enable_finbert_sentiment: bool = _env_bool("ENABLE_FINBERT_SENTIMENT", False)
    finbert_model_name: str = "ProsusAI/finbert"
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    fred_api_key: str | None = os.getenv("FRED_API_KEY")
    reddit_client_id: str | None = os.getenv("REDDIT_CLIENT_ID")
    reddit_client_secret: str | None = os.getenv("REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = os.getenv(
        "REDDIT_USER_AGENT",
        "stock-predictor/1.0",
    )
    x_search_endpoint: str | None = os.getenv("X_SEARCH_ENDPOINT")
    x_search_token: str | None = os.getenv("X_SEARCH_TOKEN")
    unusual_whales_endpoint: str | None = os.getenv("UNUSUAL_WHALES_ENDPOINT")
    unusual_whales_token: str | None = os.getenv("UNUSUAL_WHALES_TOKEN")
    tradytics_endpoint: str | None = os.getenv("TRADYTICS_ENDPOINT")
    tradytics_token: str | None = os.getenv("TRADYTICS_TOKEN")
    alpha_vantage_api_key: str | None = os.getenv("ALPHA_VANTAGE_API_KEY")
    alpha_vantage_base_url: str = os.getenv("ALPHA_VANTAGE_BASE_URL", "https://www.alphavantage.co/query")
    alpaca_api_key: str | None = os.getenv("ALPACA_API_KEY")
    alpaca_api_secret: str | None = os.getenv("ALPACA_API_SECRET")
    alpaca_data_base_url: str = os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets")
    polygon_api_key: str | None = os.getenv("POLYGON_API_KEY")
    quiver_endpoint: str | None = os.getenv("QUIVER_ENDPOINT")
    quiver_token: str | None = os.getenv("QUIVER_TOKEN")
    capitol_trades_endpoint: str | None = os.getenv("CAPITOL_TRADES_ENDPOINT")
    capitol_trades_token: str | None = os.getenv("CAPITOL_TRADES_TOKEN")
    weather_api_endpoint: str | None = os.getenv("WEATHER_API_ENDPOINT")
    weather_api_token: str | None = os.getenv("WEATHER_API_TOKEN")
    discord_webhook_url: str | None = os.getenv("DISCORD_WEBHOOK_URL")
    telegram_bot_token: str | None = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str | None = os.getenv("TELEGRAM_CHAT_ID")
    twilio_account_sid: str | None = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_auth_token: str | None = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_from_number: str | None = os.getenv("TWILIO_FROM_NUMBER")
    smtp_host: str | None = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_username: str | None = os.getenv("SMTP_USERNAME") or os.getenv("SMTP_USER")
    smtp_password: str | None = os.getenv("SMTP_PASSWORD") or os.getenv("SMTP_PASS")
    smtp_from_email: str | None = os.getenv("SMTP_FROM_EMAIL") or os.getenv("SMTP_USERNAME") or os.getenv("SMTP_USER")
    alert_email: str | None = os.getenv("ALERT_EMAIL")
    tft_max_epochs: int = int(os.getenv("TFT_MAX_EPOCHS", "3"))
    tft_batch_size: int = int(os.getenv("TFT_BATCH_SIZE", "64"))
    gpt_reasoning_top_n: int = int(os.getenv("GPT_REASONING_TOP_N", "20"))
    progress_bar: bool = _env_bool("PROGRESS_BAR", not _env_bool("STOCK_PREDICTOR_QUIET_RUNTIME", False))
    feature_flags: FeatureFlags = field(default_factory=FeatureFlags)
    cache_ttls: CacheTTLs = field(default_factory=CacheTTLs)
    signal_weights: SignalWeights = field(default_factory=SignalWeights)
    thresholds: Thresholds = field(default_factory=Thresholds)
    backtest_costs: BacktestCosts = field(default_factory=BacktestCosts)

    @property
    def all_tickers(self) -> List[str]:
        tickers = []
        for names in self.sector_universe.values():
            tickers.extend(names)
        return sorted(set(tickers))

    @property
    def adaptive_regime_paths(self) -> Dict[str, Path]:
        return {
            "bull_quiet": self.regime_bull_quiet_path,
            "bull_volatile": self.regime_bull_volatile_path,
            "neutral": self.regime_neutral_path,
            "bear_volatile": self.regime_bear_volatile_path,
            "crisis": self.regime_crisis_path,
        }


def get_config() -> AppConfig:
    """Return the runtime configuration."""
    config = AppConfig()
    base_threshold = _env_float("APP_MIN_SCORE_THRESHOLD", config.thresholds.neutral_threshold)
    config.thresholds.neutral_threshold = base_threshold
    config.thresholds.risk_on_threshold = max(40.0, base_threshold - 4.0)
    config.thresholds.risk_off_threshold = max(base_threshold, base_threshold + 11.0)
    return config
