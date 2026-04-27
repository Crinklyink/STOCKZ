"""Tree-based short-horizon predictor with XGBoost + LightGBM."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from pandas.tseries.offsets import BDay

from stock_predictor.analysis.indicators import add_indicators
from stock_predictor.config import AppConfig, get_config
from stock_predictor.data.cache import SQLiteCache
from stock_predictor.data.fetcher import MarketDataFetcher
from stock_predictor.data.fundamentals import FREDMacroClient, SECCompanyFactsClient
from stock_predictor.models.validation import PurgedSplit, monthly_purged_splits, purge_train_frame
from stock_predictor.utils import clamp

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:  # pragma: no cover - optional dependency
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:  # pragma: no cover - optional dependency
    from imblearn.over_sampling import SMOTE
except Exception:  # pragma: no cover
    SMOTE = None


CORE_FEATURE_COLUMNS = [
    "return_1",
    "return_5",
    "return_20",
    "roc_10",
    "volume_delta",
    "volume_ratio",
    "macd",
    "macd_hist",
    "rsi",
    "atr",
    "mfi",
    "vwap_distance",
    "adx",
    "stoch_k",
    "sma_50_gap",
    "bollinger_position",
    "price_vs_52w_high",
    "weeks_since_earnings",
    "sector_rs_rank",
    "consecutive_up_days",
    "gap_quality",
    "gap_held",
    "atr_percentile",
    "trend_consistency",
    "price_acceleration",
    "volume_trend",
    "micro_spread_proxy",
    "micro_slippage_proxy",
    "micro_turnover_proxy",
    "micro_impact_proxy",
    "resistance_distance",
    "earnings_proximity_score",
    "breadth_percentile",
    # --- New features (v2) ---
    "hl_ratio_trend",      # 5-day rolling (high-low)/close - volatility contraction setup signal
    "vwap_trend",          # 3-day momentum of VWAP distance - mean-reversion speed
    "obv_acceleration",    # 5-day OBV change / avg dollar volume - smart-money flow
    "atr_norm_return",     # 5-day return / ATR - risk-adjusted momentum
    "days_to_next_earnings",  # raw countdown in days (different info from proximity score)
]

# ---------------------------------------------------------------------------
# Future-derived label columns - these are computed from FUTURE prices and
# must NEVER appear in any feature matrix.  The assertion below enforces this
# at import time so it's caught immediately even if CORE_FEATURE_COLUMNS is
# accidentally extended with one of these names.
# ---------------------------------------------------------------------------
FUTURE_LABEL_COLUMNS: frozenset[str] = frozenset({
    "future_return",
    "future_excess_return",
    "future_return_5d",
    "future_return_10d",
    "future_return_3d",
    "future_excess_return_3d",
    "max_favorable_excursion",
    "max_adverse_excursion",
    "hit_target_before_stop",
    "stopped_before_target",
    "ambiguous_barrier_hit",
    "risk_adjusted_return",
    "spy_return_5d",
    "spy_return_3d",
    "short_target",
    "target",
})

FUNDAMENTAL_FEATURE_COLUMNS = [
    "fund_revenue_growth",
    "fund_revenue_growth_ttm",
    "fund_gross_margin",
    "fund_operating_margin",
    "fund_free_cash_flow_margin",
    "fund_debt_to_assets",
    "fund_share_change",
    "fund_margin_trend_ttm",
    "fund_debt_change_ttm",
    "fund_share_dilution_ttm",
]

MACRO_FEATURE_COLUMNS = [
    "macro_rates_level",
    "macro_curve_slope",
    "macro_credit_spread",
    "macro_inflation_expectation",
    "macro_unemployment",
    "macro_policy_rate",
    "macro_regime_expansion_flag",
    "macro_regime_cooling_flag",
    "macro_regime_stress_flag",
    "macro_regime_transition_up",
    "macro_regime_transition_down",
]

FEATURE_COLUMNS = CORE_FEATURE_COLUMNS + FUNDAMENTAL_FEATURE_COLUMNS + MACRO_FEATURE_COLUMNS

# Structural leakage guard: fail fast at import time if any future-derived outcome
# column is ever accidentally included in the feature matrix.
assert not FUTURE_LABEL_COLUMNS.intersection(FEATURE_COLUMNS), (
    f"DATA LEAKAGE: future label column(s) found in FEATURE_COLUMNS: "
    f"{FUTURE_LABEL_COLUMNS.intersection(FEATURE_COLUMNS)}"
)

LABEL_DEFINITION = (
    "Target = strict same-week triple-barrier setup: hit +6% before a -3.5% stop by Friday close. "
    "Rows are sampled once per Monday with purged/embargoed validation."
)


@dataclass(slots=True)
class XGBOutput:
    probability: float
    status: str
    xgb_probability: float | None = None
    lightgbm_probability: float | None = None
    short_probability: float | None = None
    blend_weights: Dict[str, float] = field(default_factory=lambda: {"xgb": 0.6, "lgbm": 0.4})
    model_spread: float = 0.0
    score_uncertainty: float = 4.0
    confidence_label: str = "medium"
    regime: str = "neutral"
    position_size_pct: float = 0.02


@dataclass(slots=True)
class TrainingReport:
    trained: bool
    training_samples: int
    validation_samples: int
    accuracy: float
    precision: float
    recall: float
    auc: float
    positive_ratio: float
    negative_ratio: float
    scale_pos_weight: float
    feature_importance: Dict[str, float]
    trained_at: str
    xgb_auc: float = 0.0
    lightgbm_auc: float = 0.0
    ensemble_auc: float = 0.0
    fold_aucs: List[Dict[str, object]] = field(default_factory=list)
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {"xgb": 0.6, "lgbm": 0.4})
    label_definition: str = LABEL_DEFINITION
    model_family: str = "XGB+LGBM"
    selected_profile: str = "baseline"
    short_horizon_auc: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class XGBoostPredictor:
    """Train and serve a calibrated tree ensemble on daily features."""

    def __init__(self, config: AppConfig | None = None, horizon_days: int = 5, *, load_persisted: bool = True) -> None:
        self.config = config or get_config()
        self.horizon_days = horizon_days
        self.label_window_days = 7
        self.model = None
        self.model_calibrator = None
        self.lightgbm_model = None
        self.lightgbm_calibrator = None
        self.short_horizon_model = None
        self.short_horizon_calibrator = None
        self.active_regime = "neutral"
        self.enabled = True
        self.training_samples = 0
        self.validation_samples = 0
        self.feature_columns = FEATURE_COLUMNS.copy()
        self.short_feature_columns = FEATURE_COLUMNS.copy()
        self.feature_importance: Dict[str, float] = {}
        self.blacklisted_features: set[str] = set()
        self.regime_models: Dict[str, object] = {}
        self.regime_feature_columns: Dict[str, List[str]] = {}
        self.blend_weights: Dict[str, float] = {"xgb": 0.6, "lgbm": 0.4}
        self.selected_profile: str = "baseline"
        self.active_profile_config: Dict[str, object] | None = None
        self.last_report: TrainingReport | None = None
        self._fundamental_feature_cache: Dict[str, pd.DataFrame] | None = None
        self._macro_feature_frame_cache: pd.DataFrame | None = None
        if load_persisted:
            self._load_persisted_model()
            self._load_regime_models()

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def fit(
        self,
        daily_frames: Dict[str, pd.DataFrame],
        *,
        cutoff: pd.Timestamp | None = None,
        save_model: bool = True,
        sector_map: Dict[str, str] | None = None,
        sector_histories: Dict[str, pd.DataFrame] | None = None,
        vix_history: pd.DataFrame | None = None,
        earnings_dates_map: Dict[str, List[Dict[str, object]]] | None = None,
        benchmark_history: pd.DataFrame | None = None,
        breadth_history: pd.Series | pd.DataFrame | None = None,
    ) -> TrainingReport:
        dataset = self.build_training_frame(
            daily_frames,
            cutoff=cutoff,
            sector_map=sector_map,
            sector_histories=sector_histories,
            vix_history=vix_history,
            earnings_dates_map=earnings_dates_map,
            benchmark_history=benchmark_history,
            breadth_history=breadth_history,
        )
        self.training_samples = len(dataset)
        if dataset.empty or len(dataset) < 200 or dataset["target"].nunique() < 2:
            LOGGER.info("Skipping tree model training; not enough rows (%s)", len(dataset))
            self.enabled = False
            report = TrainingReport(
                trained=False,
                training_samples=len(dataset),
                validation_samples=0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                auc=0.0,
                positive_ratio=0.0,
                negative_ratio=0.0,
                scale_pos_weight=1.0,
                feature_importance={},
                trained_at=datetime.now(timezone.utc).isoformat(),
            )
            self.last_report = report
            return report

        dataset = dataset.sort_values("date").reset_index(drop=True)
        selected_profile = self._default_training_profile()
        if self.config.feature_flags.hyperparameter_search:
            selected_profile, cv_result = self._search_training_profile(dataset)
        else:
            cv_result = self._walk_forward_cv(dataset, profile=selected_profile)
        self.selected_profile = str(selected_profile.get("name", "baseline"))
        self.active_profile_config = dict(selected_profile)
        split = max(int(len(dataset) * 0.8), min(200, len(dataset) - 1))
        train = dataset.iloc[:split]
        valid = dataset.iloc[split:]
        if valid.empty:
            valid = dataset.iloc[-min(200, len(dataset)) :]
            train = dataset.iloc[: max(len(dataset) - len(valid), 1)]

        pos_count = int(train["target"].sum())
        neg_count = int(len(train) - pos_count)
        scale_pos_weight = float(neg_count / max(pos_count, 1))

        feature_importance: Dict[str, float] = {}
        xgb_model = None
        xgb_calibrator = None
        lightgbm_model = None
        lightgbm_calibrator = None
        xgb_auc = 0.5
        lightgbm_auc = 0.5
        ensemble_auc = 0.5
        short_horizon_auc = 0.0
        predictions = np.zeros(len(valid), dtype=int)
        xgb_probabilities = np.full(len(valid), 0.5, dtype=float)
        lightgbm_probabilities = np.full(len(valid), 0.5, dtype=float)
        short_horizon_model = None
        short_horizon_calibrator = None
        self.blacklisted_features = set()
        self.feature_importance = {}
        self.feature_columns = FEATURE_COLUMNS.copy()
        self.short_feature_columns = FEATURE_COLUMNS.copy()
        for _ in range(4):
            self.feature_columns = self._select_feature_columns(train, valid, blacklist=self.blacklisted_features)
            xgb_model = self._build_model(
                scale_pos_weight=scale_pos_weight,
                overrides=selected_profile.get("xgb_params"),
            )
            lightgbm_model = self._build_lightgbm_model(
                scale_pos_weight=scale_pos_weight,
                overrides=selected_profile.get("lgbm_params"),
                enabled=bool(selected_profile.get("use_lightgbm", True)),
            )

            xgb_train_x, xgb_train_y, xgb_train_dates = self._balance_training_data(
                train[self.feature_columns],
                train["target"],
                sample_dates=train["date"],
            )
            xgb_fit_kwargs = self._fit_kwargs(
                xgb_model,
                xgb_train_y,
                scale_pos_weight,
                sample_dates=xgb_train_dates,
            )
            xgb_model.fit(xgb_train_x, xgb_train_y, **xgb_fit_kwargs)

            xgb_calibrator = self._calibrate_model(xgb_model, valid[self.feature_columns], valid["target"])
            xgb_probabilities = self._predict_model_probabilities(
                xgb_model,
                valid[self.feature_columns],
                xgb_calibrator,
            )
            xgb_auc = float(roc_auc_score(valid["target"], xgb_probabilities)) if valid["target"].nunique() > 1 else 0.5

            lightgbm_probabilities = np.full(len(valid), 0.5, dtype=float)
            lightgbm_auc = 0.5
            lightgbm_calibrator = None
            if lightgbm_model is not None:
                lightgbm_train_x, lightgbm_train_y, lightgbm_train_dates = self._balance_training_data(
                    train[self.feature_columns],
                    train["target"],
                    sample_dates=train["date"],
                )
                lightgbm_fit_kwargs = self._fit_kwargs(
                    lightgbm_model,
                    lightgbm_train_y,
                    scale_pos_weight,
                    sample_dates=lightgbm_train_dates,
                )
                lightgbm_model.fit(lightgbm_train_x, lightgbm_train_y, **lightgbm_fit_kwargs)
                lightgbm_calibrator = self._calibrate_model(
                    lightgbm_model,
                    valid[self.feature_columns],
                    valid["target"],
                )
                lightgbm_probabilities = self._predict_model_probabilities(
                    lightgbm_model,
                    valid[self.feature_columns],
                    lightgbm_calibrator,
                )
                lightgbm_auc = (
                    float(roc_auc_score(valid["target"], lightgbm_probabilities))
                    if valid["target"].nunique() > 1
                    else 0.5
                )

            self.blend_weights = self._resolve_blend_weights(
                selected_profile,
                float(cv_result.get("xgb_auc", xgb_auc)),
                float(cv_result.get("lightgbm_auc", lightgbm_auc)),
                lightgbm_available=lightgbm_model is not None,
            )
            ensemble_probabilities = (
                self.blend_weights["xgb"] * xgb_probabilities
                + self.blend_weights["lgbm"] * lightgbm_probabilities
            )
            ensemble_auc = (
                float(roc_auc_score(valid["target"], ensemble_probabilities))
                if valid["target"].nunique() > 1
                else 0.5
            )
            predictions = (ensemble_probabilities >= 0.5).astype(int)
            self.validation_samples = len(valid)

            feature_importance = self._combined_feature_importance(
                xgb_model,
                lightgbm_model,
                valid,
                self.blend_weights,
            )
            dominant_feature = next(iter(feature_importance.items()), None)
            if dominant_feature is None or dominant_feature[1] <= 0.15:
                break
            self.blacklisted_features.add(dominant_feature[0])
            LOGGER.info(
                "Blacklisting overly dominant feature %s at %.2f%% importance and retraining",
                dominant_feature[0],
                dominant_feature[1] * 100.0,
            )
        (
            short_horizon_model,
            short_horizon_calibrator,
            short_horizon_auc,
            self.short_feature_columns,
        ) = self._train_short_horizon_branch(train, valid, selected_profile)
        report = TrainingReport(
            trained=True,
            training_samples=len(dataset),
            validation_samples=len(valid),
            accuracy=float(accuracy_score(valid["target"], predictions)),
            precision=float(precision_score(valid["target"], predictions, zero_division=0)),
            recall=float(recall_score(valid["target"], predictions, zero_division=0)),
            auc=float(cv_result.get("ensemble_auc", ensemble_auc)),
            positive_ratio=float(pos_count / max(len(train), 1)),
            negative_ratio=float(neg_count / max(len(train), 1)),
            scale_pos_weight=scale_pos_weight,
            feature_importance=feature_importance,
            trained_at=datetime.now(timezone.utc).isoformat(),
            xgb_auc=float(cv_result.get("xgb_auc", xgb_auc)),
            lightgbm_auc=float(cv_result.get("lightgbm_auc", lightgbm_auc)),
            ensemble_auc=float(cv_result.get("ensemble_auc", ensemble_auc)),
            fold_aucs=list(cv_result.get("rows", [])),
            ensemble_weights=self.blend_weights.copy(),
            selected_profile=self.selected_profile,
            short_horizon_auc=short_horizon_auc,
            model_family="XGB+LGBM" if self.config.feature_flags.native_boosters else "RandomForestFallback",
        )

        self.model = xgb_model
        self.model_calibrator = xgb_calibrator
        self.lightgbm_model = lightgbm_model
        self.lightgbm_calibrator = lightgbm_calibrator
        self.short_horizon_model = short_horizon_model
        self.short_horizon_calibrator = short_horizon_calibrator
        self.feature_importance = feature_importance
        self.last_report = report
        self.training_samples = len(dataset)
        self.enabled = True
        if save_model:
            self._save_model(report)
        self._train_regime_models(
            dataset,
            save_model=save_model,
            xgb_overrides=selected_profile.get("xgb_params"),
        )
        return report

    def build_training_frame(
        self,
        daily_frames: Dict[str, pd.DataFrame],
        *,
        cutoff: pd.Timestamp | None = None,
        max_samples_per_ticker: int | None = None,
        sector_map: Dict[str, str] | None = None,
        sector_histories: Dict[str, pd.DataFrame] | None = None,
        vix_history: pd.DataFrame | None = None,
        earnings_dates_map: Dict[str, List[Dict[str, object]]] | None = None,
        benchmark_history: pd.DataFrame | None = None,
        breadth_history: pd.Series | pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        records: List[dict] = []
        cutoff_ts = pd.to_datetime(cutoff, utc=True) if cutoff is not None else None
        sector_rank_frame = self._build_sector_rank_frame(sector_histories or {})
        regime_series = self._build_regime_series(vix_history)
        breadth_series = self._build_breadth_series(breadth_history)
        benchmark_weekly_targets = self._build_weekly_return_targets(benchmark_history["close"]) if (
            benchmark_history is not None and not benchmark_history.empty and "close" in benchmark_history
        ) else pd.DataFrame()
        benchmark_short_returns = self._build_benchmark_returns_for_horizon(benchmark_history, horizon_days=3)
        fundamental_feature_map = self._fundamental_feature_map(daily_frames.keys())
        macro_feature_frame = self._macro_feature_frame()

        for ticker, frame in daily_frames.items():
            prepared = self._prepare_frame(
                frame,
                sector_rank_series=sector_rank_frame.get((sector_map or {}).get(ticker, "")),
                earnings_dates=(earnings_dates_map or {}).get(ticker, []),
                breadth_series=breadth_series,
            )
            if len(prepared) < 80:
                continue
            if cutoff_ts is not None:
                prepared = prepared.loc[prepared.index < cutoff_ts]
            if len(prepared) < 80:
                continue
            if self.config.training_lookback_days:
                lookback_start = prepared.index.max() - pd.Timedelta(days=self.config.training_lookback_days)
                prepared = prepared.loc[prepared.index >= lookback_start]
            if len(prepared) < 80:
                continue
            weekly_targets = self._build_weekly_return_targets(prepared["close"], prepared.get("high"), prepared.get("low"))
            candidate_indices = self._weekly_sample_indices(prepared.index)
            if max_samples_per_ticker is not None and len(candidate_indices) > max_samples_per_ticker:
                sampled_positions = np.linspace(0, len(candidate_indices) - 1, num=max_samples_per_ticker, dtype=int)
                candidate_indices = [candidate_indices[position] for position in sampled_positions]
            for index in candidate_indices:
                sample_date = prepared.index[index]
                week_end = self._week_end_for_date(sample_date)
                if week_end not in weekly_targets.index:
                    continue
                if not benchmark_weekly_targets.empty and week_end not in benchmark_weekly_targets.index:
                    continue
                row = prepared.iloc[index]
                stock_weekly = weekly_targets.loc[week_end]
                spy_return = (
                    float(benchmark_weekly_targets.loc[week_end, "weekly_return"])
                    if not benchmark_weekly_targets.empty
                    else 0.0
                )
                future_return = float(stock_weekly["weekly_return"])
                future_excess_return = float(future_return - spy_return)
                trade_target = bool(stock_weekly.get("hit_target_before_stop", False))
                risk_adjusted_return = float(stock_weekly.get("risk_adjusted_return", 0.0))
                sample_close = float(row.get("close", 0.0))
                future_close_3d = float(prepared["close"].shift(-3).iloc[index]) if index < len(prepared) - 3 else sample_close
                short_return_3d = (future_close_3d / sample_close - 1.0) if sample_close > 0 else 0.0
                short_spy = float(benchmark_short_returns.reindex([sample_date], method="ffill").iloc[0]) if not benchmark_short_returns.empty else 0.0
                short_excess = short_return_3d - short_spy
                record = {
                    "ticker": ticker,
                    "date": sample_date,
                    "label_start_date": stock_weekly["label_start_date"],
                    "label_end_date": stock_weekly["label_end_date"],
                    "future_return": future_return,
                    "future_excess_return": future_excess_return,
                    "future_return_5d": float(stock_weekly.get("future_return_5d", future_return)),
                    "future_return_10d": float(stock_weekly.get("future_return_10d", future_return)),
                    "future_return_3d": float(short_return_3d),
                    "max_favorable_excursion": float(stock_weekly.get("max_favorable_excursion", future_return)),
                    "max_adverse_excursion": float(stock_weekly.get("max_adverse_excursion", 0.0)),
                    "hit_target_before_stop": int(trade_target),
                    "stopped_before_target": int(stock_weekly.get("stopped_before_target", False)),
                    "ambiguous_barrier_hit": int(stock_weekly.get("ambiguous_barrier_hit", False)),
                    "risk_adjusted_return": risk_adjusted_return,
                    "spy_return_5d": spy_return,
                    "spy_return_3d": short_spy,
                    "future_excess_return_3d": short_excess,
                    "short_target": int(short_excess >= 0.01),
                    "target": int(trade_target),
                    "regime": self._lookup_regime(sample_date, regime_series),
                }
                for feature in CORE_FEATURE_COLUMNS:
                    record[feature] = float(row[feature])
                record.update(self._fundamental_features_for_date(ticker, sample_date, fundamental_feature_map))
                record.update(self._macro_features_for_date(sample_date, macro_feature_frame))
                records.append(record)
        return pd.DataFrame.from_records(records)

    def walk_forward_backtest(
        self,
        daily_frames: Dict[str, pd.DataFrame],
        *,
        benchmark_history: pd.DataFrame | None = None,
        breadth_history: pd.Series | pd.DataFrame | None = None,
        weeks: int = 12,
        profile: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        dataset = self.build_training_frame(
            daily_frames,
            max_samples_per_ticker=None,
            benchmark_history=benchmark_history,
            breadth_history=breadth_history,
        )
        if dataset.empty:
            return {"rows": [], "summary": {}, "markdown": "No training rows were available."}
        dataset["date"] = pd.to_datetime(dataset["date"], utc=True)
        dataset["label_end_date"] = pd.to_datetime(dataset["label_end_date"], utc=True)
        latest = dataset["date"].max().normalize()
        rows = []
        weekly_returns = []
        active_profile = profile or self.active_profile_config or {
            "name": self.selected_profile,
            "blend_weights": self.blend_weights.copy(),
            "use_lightgbm": self.lightgbm_model is not None,
            "xgb_params": {},
            "lgbm_params": {},
        }
        for offset in range(weeks, 0, -1):
            week_start = latest - pd.Timedelta(weeks=offset - 1)
            train = self._purge_train_rows(dataset, test_start=week_start)
            evaluate = dataset.loc[
                (dataset["date"] >= week_start) & (dataset["date"] < week_start + pd.Timedelta(days=7))
            ]
            if len(train) < 200 or evaluate.empty or train["target"].nunique() < 2:
                continue
            feature_columns = [column for column in FEATURE_COLUMNS if column in train.columns and column in evaluate.columns]
            scale_pos_weight = float((train["target"] == 0).sum() / max(int(train["target"].sum()), 1))
            xgb_model = self._build_model(
                scale_pos_weight=scale_pos_weight,
                overrides=active_profile.get("xgb_params"),
            )
            xgb_train_x, xgb_train_y, xgb_train_dates = self._balance_training_data(
                train[feature_columns],
                train["target"],
                sample_dates=train["date"],
            )
            xgb_model.fit(
                xgb_train_x,
                xgb_train_y,
                **self._fit_kwargs(xgb_model, xgb_train_y, scale_pos_weight, sample_dates=xgb_train_dates),
            )
            xgb_probs = self._predict_model_probabilities(xgb_model, evaluate[feature_columns], None)
            xgb_auc = float(roc_auc_score(evaluate["target"], xgb_probs)) if evaluate["target"].nunique() > 1 else 0.5
            lgbm_probs = np.full(len(evaluate), 0.5, dtype=float)
            lightgbm_auc = 0.5
            lightgbm_model = None
            if self.config.feature_flags.lightgbm_ensemble and LGBMClassifier is not None:
                lgbm_model = self._build_lightgbm_model(
                    scale_pos_weight=scale_pos_weight,
                    overrides=active_profile.get("lgbm_params"),
                    enabled=bool(active_profile.get("use_lightgbm", True)),
                )
                if lgbm_model is not None:
                    lgbm_train_x, lgbm_train_y, lgbm_train_dates = self._balance_training_data(
                        train[feature_columns],
                        train["target"],
                        sample_dates=train["date"],
                    )
                    lgbm_model.fit(
                        lgbm_train_x,
                        lgbm_train_y,
                        **self._fit_kwargs(lgbm_model, lgbm_train_y, scale_pos_weight, sample_dates=lgbm_train_dates),
                    )
                    lgbm_probs = self._predict_model_probabilities(lgbm_model, evaluate[feature_columns], None)
                    lightgbm_auc = (
                        float(roc_auc_score(evaluate["target"], lgbm_probs))
                        if evaluate["target"].nunique() > 1
                        else 0.5
                    )
            weights = self._resolve_blend_weights(
                active_profile,
                xgb_auc,
                lightgbm_auc,
                lightgbm_available=lightgbm_model is not None,
            )
            probabilities = weights["xgb"] * xgb_probs + weights["lgbm"] * lgbm_probs
            scored = evaluate.copy()
            scored["probability"] = probabilities
            top = scored.sort_values("probability", ascending=False).head(10)
            if top.empty:
                continue
            gross_return = float(top["future_return"].mean())
            avg_return = float((top["future_return"] - self.config.backtest_costs.round_trip_cost).mean())
            win_rate = float((top["target"] == 1).mean())
            turnover = float(min(1.0, len(top) / 10.0))
            weekly_returns.append(avg_return)
            rows.append(
                {
                    "week": week_start.strftime("%Y-%m-%d"),
                    "picks": int(len(top)),
                    "win_rate": round(win_rate * 100.0, 2),
                    "gross_avg_return": round(gross_return * 100.0, 2),
                    "avg_return": round(avg_return * 100.0, 2),
                    "round_trip_cost_bps": round(self.config.backtest_costs.round_trip_cost * 10_000.0, 2),
                    "turnover": round(turnover * 100.0, 2),
                }
            )
        sharpe = 0.0
        if len(weekly_returns) > 1 and np.std(weekly_returns) > 0:
            sharpe = float(np.mean(weekly_returns) / np.std(weekly_returns) * np.sqrt(52))
        summary = {
            "weeks": len(rows),
            "win_rate": round(np.mean([row["win_rate"] for row in rows]) if rows else 0.0, 2),
            "average_return": round(np.mean([row["avg_return"] for row in rows]) if rows else 0.0, 2),
            "gross_average_return": round(np.mean([row["gross_avg_return"] for row in rows]) if rows else 0.0, 2),
            "round_trip_cost_bps": round(self.config.backtest_costs.round_trip_cost * 10_000.0, 2),
            "sharpe": round(sharpe, 2),
        }
        markdown_lines = [
            "# Walk-Forward Backtest",
            "",
            f"Win Rate: {summary['win_rate']:.2f}%",
            f"Gross Average Return: {summary['gross_average_return']:.2f}%",
            f"Average Return: {summary['average_return']:.2f}%",
            f"Round-Trip Cost: {summary['round_trip_cost_bps']:.2f} bps",
            f"Sharpe Estimate: {summary['sharpe']:.2f}",
            "",
            "| Week | Picks | Win Rate | Gross Avg Return | Net Avg Return | Cost (bps) | Turnover |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            markdown_lines.append(
                f"| {row['week']} | {row['picks']} | {row['win_rate']:.2f}% | "
                f"{row['gross_avg_return']:.2f}% | {row['avg_return']:.2f}% | "
                f"{row['round_trip_cost_bps']:.2f} | {row['turnover']:.2f}% |"
            )
        return {"rows": rows, "summary": summary, "markdown": "\n".join(markdown_lines)}

    def _evaluate_trade_basket(
        self,
        evaluate: pd.DataFrame,
        probabilities: np.ndarray,
        *,
        picks_per_date: int = 1,
    ) -> Dict[str, float]:
        if evaluate.empty or len(probabilities) != len(evaluate):
            return {
                "trade_win_rate": 0.0,
                "trade_stop_rate": 0.0,
                "trade_average_return": 0.0,
                "trade_average_excess_return": 0.0,
                "trade_picks": 0.0,
            }
        scored = evaluate.copy()
        scored["probability"] = probabilities
        baskets: List[pd.DataFrame] = []
        for _, group in scored.groupby("date", sort=True):
            ordered = group.sort_values("probability", ascending=False).head(min(picks_per_date, len(group)))
            if not ordered.empty:
                baskets.append(ordered)
        if not baskets:
            return {
                "trade_win_rate": 0.0,
                "trade_stop_rate": 0.0,
                "trade_average_return": 0.0,
                "trade_average_excess_return": 0.0,
                "trade_picks": 0.0,
            }
        basket = pd.concat(baskets, ignore_index=True)
        return {
            "trade_win_rate": float((basket["target"] == 1).mean()),
            "trade_stop_rate": float((basket["stopped_before_target"] == 1).mean())
            if "stopped_before_target" in basket
            else 0.0,
            "trade_average_return": float(basket["future_return"].mean()),
            "trade_average_excess_return": float(basket["future_excess_return"].mean()),
            "trade_picks": float(len(basket)),
        }

    def predict_proba(
        self,
        frame: pd.DataFrame,
        *,
        sector_rs_rank: float = 50.0,
        weeks_since_earnings: float | None = None,
        earnings_dates: List[Dict[str, object]] | None = None,
        earnings_proximity_score: float | None = None,
        breadth_percentile: float = 0.5,
        regime: str | None = None,
    ) -> XGBOutput:
        active_regime = regime or self.active_regime
        model = self.model
        calibrator = self.model_calibrator
        feature_columns = self.feature_columns
        if not self.enabled or model is None:
            probability = self._heuristic_probability(frame)
            return XGBOutput(probability=probability, status="heuristic", xgb_probability=probability)

        try:
            data = self._prepare_inference_frame(
                frame,
                sector_rank_series=pd.Series(sector_rs_rank, index=frame.index),
                earnings_dates=earnings_dates,
                weeks_since_earnings_override=weeks_since_earnings,
                earnings_proximity_override=earnings_proximity_score,
                breadth_series=pd.Series(breadth_percentile, index=frame.index),
            )
        except TypeError:
            data = self._prepare_inference_frame(frame)
        if data.empty:
            probability = self._heuristic_probability(frame)
            return XGBOutput(probability=probability, status="heuristic", xgb_probability=probability)

        latest = data.iloc[-1]
        regime_model = self.regime_models.get(active_regime)
        regime_columns = self.regime_feature_columns.get(active_regime, self.feature_columns)
        if regime_model is not None and all(feature in latest.index for feature in regime_columns):
            model = regime_model
            calibrator = None
            feature_columns = regime_columns
        if not all(feature in latest.index for feature in feature_columns):
            probability = self._heuristic_probability(frame)
            return XGBOutput(probability=probability, status="heuristic", xgb_probability=probability)

        features = pd.DataFrame(
            [{feature: float(latest[feature]) for feature in feature_columns}],
            columns=feature_columns,
        )
        xgb_probability = float(self._predict_model_probabilities(model, features, calibrator)[0])
        lightgbm_probability = None
        if (
            self.config.feature_flags.native_boosters
            and self.config.feature_flags.lightgbm_ensemble
            and self.lightgbm_model is not None
            and all(feature in latest.index for feature in self.feature_columns)
        ):
            lightgbm_features = pd.DataFrame(
                [{feature: float(latest[feature]) for feature in self.feature_columns}],
                columns=self.feature_columns,
            )
            try:
                lightgbm_probability = float(
                    self._predict_model_probabilities(
                        self.lightgbm_model,
                        lightgbm_features,
                        self.lightgbm_calibrator,
                    )[0]
                )
            except Exception:
                LOGGER.debug("Ignoring LightGBM prediction failure", exc_info=True)
                lightgbm_probability = None
        weights = self.blend_weights.copy() if lightgbm_probability is not None else {"xgb": 1.0, "lgbm": 0.0}
        if lightgbm_probability is not None:
            total = float(weights.get("xgb", 0.0)) + float(weights.get("lgbm", 0.0))
            if total <= 0:
                weights = self._blend_weights_from_aucs(
                    self.last_report.xgb_auc if self.last_report else 0.5,
                    self.last_report.lightgbm_auc if self.last_report else 0.5,
                    lightgbm_available=True,
                )
            else:
                weights = {
                    "xgb": float(weights.get("xgb", 0.0)) / total,
                    "lgbm": float(weights.get("lgbm", 0.0)) / total,
                }
        if lightgbm_probability is None:
            base_probability = xgb_probability
            spread = 0.0
        else:
            base_probability = weights["xgb"] * xgb_probability + weights["lgbm"] * lightgbm_probability
            spread = abs(xgb_probability - lightgbm_probability)
        short_probability = None
        short_weight = 0.0
        if self.short_horizon_model is not None and all(
            feature in latest.index for feature in self.short_feature_columns
        ):
            short_features = pd.DataFrame(
                [{feature: float(latest[feature]) for feature in self.short_feature_columns}],
                columns=self.short_feature_columns,
            )
            short_probability = float(
                self._predict_model_probabilities(
                    self.short_horizon_model,
                    short_features,
                    self.short_horizon_calibrator,
                )[0]
            )
            short_weight = self._short_horizon_blend_weight()
        probability = base_probability
        if short_probability is not None and short_weight > 0:
            probability = (1.0 - short_weight) * base_probability + short_weight * short_probability
            spread = max(spread, abs(base_probability - short_probability))
            weights = {**weights, "short": short_weight}
        confidence_label = "high" if spread <= 0.10 else "medium" if spread <= 0.20 else "low"
        status = active_regime if regime_model is not None and model is regime_model else "trained"
        return XGBOutput(
            probability=clamp(probability, 0.0, 1.0),
            status=status,
            xgb_probability=clamp(xgb_probability, 0.0, 1.0),
            lightgbm_probability=clamp(lightgbm_probability, 0.0, 1.0) if lightgbm_probability is not None else None,
            short_probability=clamp(short_probability, 0.0, 1.0) if short_probability is not None else None,
            blend_weights=weights,
            model_spread=spread,
            confidence_label=confidence_label,
        )

    def _prepare_frame(
        self,
        frame: pd.DataFrame,
        *,
        sector_rank_series: pd.Series | None = None,
        earnings_dates: List[Dict[str, object]] | None = None,
        benchmark_forward_return_series: pd.Series | None = None,
        breadth_series: pd.Series | None = None,
        weeks_since_earnings_override: float | None = None,
        earnings_proximity_override: float | None = None,
    ) -> pd.DataFrame:
        data = add_indicators(frame).copy()
        if data.empty:
            return data

        close = data["close"]
        volume = data["volume"]
        high = data["high"]
        previous_close = close.shift(1)

        # uses T-1 data (safe)
        data["return_1"] = data["return_1"].shift(1)
        # uses T-1 data (safe)
        data["return_5"] = data["return_5"].shift(1)
        # uses T-1 data (safe)
        data["return_20"] = data["return_20"].shift(1)
        # uses T-1 data (safe)
        data["roc_10"] = data["roc_10"].shift(1)
        # uses T-1 data (safe)
        data["volume_delta"] = data["volume_delta"].shift(1)
        # uses T-1 data (safe)
        data["volume_ratio"] = (volume / volume.rolling(20).mean().replace(0, np.nan)).shift(1)
        # uses T-1 data (safe)
        data["macd"] = data["macd"].shift(1)
        # uses T-1 data (safe)
        data["macd_hist"] = data["macd_hist"].shift(1)
        # uses T-1 data (safe)
        data["rsi"] = data["rsi"].shift(1)
        # uses T-1 data (safe)
        data["atr"] = data["atr"].shift(1)
        # uses T-1 data (safe)
        data["mfi"] = data["mfi"].shift(1)
        # uses T-1 data (safe)
        data["vwap_distance"] = data["vwap_distance"].shift(1)
        # uses T-1 data (safe)
        data["adx"] = data["adx"].shift(1)
        # uses T-1 data (safe)
        data["stoch_k"] = data["stoch_k"].shift(1)
        # uses T-1 data (safe)
        data["sma_50_gap"] = (close / data["sma_50"].replace(0, np.nan) - 1.0).shift(1)
        band_width = (data["bollinger_high"] - data["bollinger_low"]).replace(0, np.nan)
        # uses T-1 data (safe)
        data["bollinger_position"] = ((close - data["bollinger_low"]) / band_width).shift(1)
        # uses T-1 data (safe)
        data["price_vs_52w_high"] = (close / close.rolling(252).max().replace(0, np.nan)).shift(1)
        # uses calendar data available at T (safe)
        data["day_of_week"] = data.index.dayofweek.astype(float)
        up_days = (close.diff() > 0).astype(int)
        streak_group = (up_days != up_days.shift()).cumsum()
        # uses T-1 data (safe)
        data["consecutive_up_days"] = up_days.groupby(streak_group).cumsum().where(up_days.eq(1), 0.0).shift(1)
        gap_size = (data["open"] - previous_close) / previous_close.replace(0, np.nan)
        gap_volume_ratio = volume / volume.rolling(20).mean().replace(0, np.nan)
        # uses T-open, T-1 close, and T volume (safe)
        data["gap_quality"] = (gap_size * gap_volume_ratio).clip(lower=-0.1, upper=0.1).shift(1)
        # uses T open/close follow-through and only positive gap setups (safe)
        data["gap_held"] = (
            ((close - data["open"]) / data["open"].abs().clip(lower=0.001))
            .where(gap_size > 0.005, 0.0)
            .shift(1)
        )
        # Compatibility for older persisted models; no longer part of FEATURE_COLUMNS.
        data["gap_up_flag"] = (gap_size > 0.005).astype(float).shift(1)
        # uses T-1 data (safe)
        data["atr_percentile"] = data["atr"].rolling(126, min_periods=20).rank(pct=True).shift(1)
        # uses T-1 data (safe)
        data["trend_consistency"] = (close.pct_change() > 0).astype(float).rolling(10).mean().shift(1)
        momentum_5d = close.pct_change(5)
        momentum_10d = close.pct_change(10)
        # uses T-1 data (safe)
        data["price_acceleration"] = (momentum_5d - (momentum_10d / 2.0)).shift(1)
        # uses T-1 data (safe)
        data["volume_trend"] = (volume.rolling(5).mean() / volume.rolling(20).mean()).shift(1)
        intraday_range = (data["high"] - data["low"]).replace(0, np.nan)
        dollar_volume = (close * volume).replace(0, np.nan)
        # uses T-1 daily spread proxy (safe)
        data["micro_spread_proxy"] = (intraday_range / close.replace(0, np.nan)).shift(1)
        # uses T-1 slippage pressure proxy (safe)
        data["micro_slippage_proxy"] = ((close - data["open"]).abs() / intraday_range).clip(0.0, 2.0).shift(1)
        # uses T-1 turnover intensity proxy (safe)
        data["micro_turnover_proxy"] = (dollar_volume / dollar_volume.rolling(20).mean().replace(0, np.nan)).shift(1)
        # uses T-1 impact proxy: large returns on low liquidity (safe)
        data["micro_impact_proxy"] = (
            close.pct_change(fill_method=None).abs() / dollar_volume.rolling(5).mean().replace(0, np.nan)
        ).mul(1e8).clip(0.0, 5.0).shift(1)
        high_52w = close.rolling(252).max().replace(0, np.nan)
        # uses T-1 data (safe)
        data["resistance_distance"] = ((high_52w - close) / close.replace(0, np.nan)).shift(1)

        # --- New v2 features ---
        # hl_ratio_trend: 5-day rolling mean of intraday range / close - volatility contraction signal
        hl_ratio = intraday_range / close.replace(0, np.nan)
        data["hl_ratio_trend"] = hl_ratio.rolling(5, min_periods=2).mean().shift(1)
        # vwap_trend: 3-day momentum of VWAP distance - data["vwap_distance"] is already T-1 shifted
        # pct_change(3) gives T-4 to T-1 trend; already properly lagged, no extra shift needed
        data["vwap_trend"] = data["vwap_distance"].pct_change(3, fill_method=None).clip(-0.5, 0.5)
        # obv_acceleration: 5-day OBV change normalized by avg dollar volume - smart-money detection
        if "obv" in data.columns:
            obv_5d = data["obv"].diff(5).fillna(0.0)
            obv_scale = dollar_volume.rolling(20, min_periods=5).mean().replace(0, np.nan)
            data["obv_acceleration"] = (obv_5d / obv_scale).clip(-5.0, 5.0).shift(1)
        else:
            data["obv_acceleration"] = 0.0
        # atr_norm_return: risk-adjusted 5-day return - both return_5 and atr are already T-1 shifted
        atr_safe = data["atr"].replace(0, np.nan)
        data["atr_norm_return"] = (data["return_5"] / atr_safe).clip(-10.0, 10.0)
        # days_to_next_earnings: raw countdown (uses published calendar data, safe)
        data["days_to_next_earnings"] = self._days_to_next_earnings(data.index, earnings_dates or [])

        if sector_rank_series is not None and not sector_rank_series.empty:
            # uses T-1 sector tape (safe)
            data["sector_rs_rank"] = sector_rank_series.reindex(data.index, method="ffill").shift(1).fillna(50.0)
        else:
            data["sector_rs_rank"] = 50.0

        if weeks_since_earnings_override is not None:
            # uses published calendar data available at T (safe)
            data["weeks_since_earnings"] = weeks_since_earnings_override
        else:
            # uses published calendar data available at T (safe)
            data["weeks_since_earnings"] = self._weeks_since_earnings(data.index, earnings_dates or [])

        if earnings_proximity_override is not None:
            # uses published calendar data available at T (safe)
            data["earnings_proximity_score"] = earnings_proximity_override
        else:
            # uses published calendar data available at T (safe)
            data["earnings_proximity_score"] = self._earnings_proximity_score(data.index, earnings_dates or [])

        if breadth_series is not None and not breadth_series.empty:
            # uses T-1 market breadth data (safe)
            data["breadth_percentile"] = breadth_series.reindex(data.index, method="ffill").shift(1).fillna(0.5)
        else:
            data["breadth_percentile"] = 0.5
        for feature in FUNDAMENTAL_FEATURE_COLUMNS:
            data[feature] = 0.0
        for feature in MACRO_FEATURE_COLUMNS:
            data[feature] = 0.0

        future_high = self._future_high_return(high, close)
        data["future_return"] = future_high
        if benchmark_forward_return_series is not None and not benchmark_forward_return_series.empty:
            data["spy_return_5d"] = benchmark_forward_return_series.reindex(data.index, method="ffill")
        else:
            data["spy_return_5d"] = 0.0
        data["future_excess_return"] = data["future_return"] - data["spy_return_5d"]
        data["target"] = (data["future_excess_return"] >= 0.03).astype(int)
        return data.replace([np.inf, -np.inf], np.nan).dropna(
            subset=CORE_FEATURE_COLUMNS + ["future_return", "spy_return_5d", "future_excess_return"]
        )

    def _prepare_inference_frame(self, frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if frame.empty or not self.config.feature_flags.inference_alignment:
            return self._prepare_frame(frame, **kwargs)
        augmented = frame.copy()
        synthetic = augmented.iloc[[-1]].copy()
        next_index = pd.DatetimeIndex([pd.Timestamp(augmented.index[-1]) + BDay(1)])
        if getattr(augmented.index, "tz", None) is not None and next_index.tz is None:
            next_index = next_index.tz_localize(augmented.index.tz)
        synthetic.index = next_index
        augmented = pd.concat([augmented, synthetic])
        return self._prepare_frame(augmented, **kwargs)

    def _build_model(self, *, scale_pos_weight: float = 1.0, overrides: Dict[str, object] | None = None):
        xgb_params = {
            "max_depth": 5,
            "n_estimators": 420,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.6,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "scale_pos_weight": scale_pos_weight,
            "tree_method": "hist",
            "max_delta_step": 1,
            "min_child_weight": 4,
            "gamma": 0.05,
            "reg_alpha": 0.3,
            "reg_lambda": 2.0,
            "random_state": 42,
            "n_jobs": 1,
        }
        if overrides:
            xgb_params.update(overrides)
        if self.config.feature_flags.native_boosters and XGBClassifier is not None:
            return XGBClassifier(**xgb_params)
        return RandomForestClassifier(
            n_estimators=int((overrides or {}).get("n_estimators", 120)),
            max_depth=int((overrides or {}).get("max_depth", 6)),
            min_samples_leaf=int((overrides or {}).get("min_samples_leaf", 8)),
            max_features=(overrides or {}).get("max_features", "sqrt"),
            class_weight="balanced_subsample",
            random_state=int((overrides or {}).get("random_state", 42)),
            n_jobs=1,
        )

    def _build_lightgbm_model(
        self,
        *,
        scale_pos_weight: float = 1.0,
        overrides: Dict[str, object] | None = None,
        enabled: bool = True,
    ):
        if (
            enabled is False
            or not self.config.feature_flags.native_boosters
            or not self.config.feature_flags.lightgbm_ensemble
            or LGBMClassifier is None
        ):
            return None
        lgbm_params = {
            "n_estimators": 420,
            "learning_rate": 0.03,
            "num_leaves": 47,
            "subsample": 0.85,
            "colsample_bytree": 0.6,
            "min_child_samples": 30,
            "reg_alpha": 0.3,
            "reg_lambda": 2.0,
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": 1,
        }
        if overrides:
            lgbm_params.update(overrides)
        return LGBMClassifier(**lgbm_params)

    def _predict_model_probabilities(self, model, features: pd.DataFrame, calibrator) -> np.ndarray:
        if calibrator is not None and hasattr(calibrator, "predict_proba"):
            return np.asarray(calibrator.predict_proba(features))[:, 1]
        if hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(features))[:, 1]
        return np.asarray(model.predict(features), dtype=float)

    def _calibrate_model(self, model, features: pd.DataFrame, target: pd.Series):
        if len(features) < 100 or target.nunique() < 2:
            return None
        try:
            # Isotonic regression is more expressive than sigmoid (Platt scaling) for
            # tree-based models whose raw scores have non-monotone probability shapes.
            # Requires >= 200 samples to avoid overfitting the calibration mapping.
            method = "isotonic" if len(features) >= 200 else "sigmoid"
            calibrator = CalibratedClassifierCV(model, cv="prefit", method=method)
            calibrator.fit(features, target)
            return calibrator
        except Exception:
            LOGGER.debug("Probability calibration failed (method=%s)", method, exc_info=True)
            # Second-chance fallback: sigmoid is more robust with sparse validation sets.
            if method == "isotonic":
                try:
                    calibrator = CalibratedClassifierCV(model, cv="prefit", method="sigmoid")
                    calibrator.fit(features, target)
                    return calibrator
                except Exception:
                    LOGGER.debug("Sigmoid calibration fallback also failed", exc_info=True)
            return None

    def _fit_kwargs(
        self,
        model,
        target: pd.Series,
        scale_pos_weight: float,
        *,
        sample_dates: pd.Series | None = None,
    ) -> Dict[str, object]:
        recency_weights = self._recency_sample_weights(sample_dates, len(target))
        if XGBClassifier is not None and isinstance(model, XGBClassifier):
            return {"sample_weight": recency_weights} if recency_weights is not None else {}
        if LGBMClassifier is not None and isinstance(model, LGBMClassifier):
            return {"sample_weight": recency_weights} if recency_weights is not None else {}
        class_weights = np.where(np.asarray(target) == 1, scale_pos_weight, 1.0)
        if recency_weights is not None:
            sample_weight = class_weights * recency_weights
        else:
            sample_weight = class_weights
        return {"sample_weight": sample_weight}

    def _train_short_horizon_branch(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        profile: Dict[str, object],
    ) -> tuple[object | None, object | None, float, List[str]]:
        if "short_target" not in train or "short_target" not in valid:
            return None, None, 0.0, self.feature_columns.copy()
        if train["short_target"].nunique() < 2 or valid["short_target"].nunique() < 2:
            return None, None, 0.0, self.feature_columns.copy()
        feature_columns = [
            column
            for column in self.feature_columns
            if column in train.columns and column in valid.columns
        ]
        if not feature_columns:
            return None, None, 0.0, self.feature_columns.copy()
        pos_count = int(train["short_target"].sum())
        neg_count = int(len(train) - pos_count)
        if pos_count <= 0 or neg_count <= 0:
            return None, None, 0.0, feature_columns
        scale_pos_weight = float(neg_count / max(pos_count, 1))
        try:
            short_model = self._build_model(
                scale_pos_weight=scale_pos_weight,
                overrides=profile.get("xgb_params"),
            )
            short_train_x, short_train_y, short_train_dates = self._balance_training_data(
                train[feature_columns],
                train["short_target"],
                sample_dates=train["date"],
            )
            fit_kwargs = self._fit_kwargs(
                short_model,
                short_train_y,
                scale_pos_weight,
                sample_dates=short_train_dates,
            )
            short_model.fit(short_train_x, short_train_y, **fit_kwargs)
            short_calibrator = self._calibrate_model(
                short_model,
                valid[feature_columns],
                valid["short_target"],
            )
            short_probabilities = self._predict_model_probabilities(
                short_model,
                valid[feature_columns],
                short_calibrator,
            )
            short_auc = (
                float(roc_auc_score(valid["short_target"], short_probabilities))
                if valid["short_target"].nunique() > 1
                else 0.5
            )
            return short_model, short_calibrator, short_auc, feature_columns
        except Exception:
            LOGGER.debug("Short-horizon model branch training failed", exc_info=True)
            return None, None, 0.0, feature_columns

    def _short_horizon_blend_weight(self) -> float:
        if self.last_report is None:
            return 0.10
        short_auc = float(self.last_report.short_horizon_auc)
        if short_auc <= 0.5:
            return 0.0
        return clamp(0.08 + (short_auc - 0.5) * 0.7, 0.08, 0.25)

    def _combined_feature_importance(
        self,
        xgb_model,
        lightgbm_model,
        valid: pd.DataFrame,
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        xgb_importance = self._single_feature_importance(xgb_model, valid)
        lightgbm_importance = (
            self._single_feature_importance(lightgbm_model, valid)
            if lightgbm_model is not None
            else {}
        )
        combined = {}
        for feature in self.feature_columns:
            combined[feature] = (
                weights["xgb"] * xgb_importance.get(feature, 0.0)
                + weights["lgbm"] * lightgbm_importance.get(feature, 0.0)
            )
        total = sum(combined.values()) or 1.0
        normalized = {feature: value / total for feature, value in combined.items() if value > 0}
        self._save_feature_importance_chart(normalized)
        return dict(sorted(normalized.items(), key=lambda item: item[1], reverse=True))

    def _single_feature_importance(self, model, valid: pd.DataFrame) -> Dict[str, float]:
        if model is None:
            return {}
        if hasattr(model, "feature_importances_"):
            raw = np.asarray(getattr(model, "feature_importances_"), dtype=float)
        else:
            result = permutation_importance(
                model,
                valid[self.feature_columns],
                valid["target"],
                n_repeats=5,
                random_state=42,
                n_jobs=1,
            )
            raw = np.asarray(result.importances_mean, dtype=float)
        raw = np.clip(raw, a_min=0.0, a_max=None)
        total = raw.sum() or 1.0
        return {
            feature: float(value / total)
            for feature, value in zip(self.feature_columns, raw, strict=False)
        }

    def _select_feature_columns(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        *,
        blacklist: set[str] | None = None,
    ) -> List[str]:
        blacklist = blacklist or set()
        if self.feature_importance and len(self.feature_importance) >= max(5, int(len(FEATURE_COLUMNS) * 0.5)):
            importances = {k: v for k, v in self.feature_importance.items() if v > 0}
            if importances:
                values = np.array(list(importances.values()), dtype=float)
                # Use a percentile-based cutoff rather than a fixed 0.01 threshold.
                # P20 of the importance distribution preserves the top 80% of signal
                # while dropping consistently near-zero features - more adaptive than
                # a hard floor which would either under- or over-select depending on
                # how many features the model actually uses.
                cutoff = float(np.percentile(values, 20))
                cutoff = max(cutoff, 1e-4)  # never drop everything
                filtered = [
                    feature
                    for feature, value in importances.items()
                    if (
                        value >= cutoff
                        and feature not in blacklist
                        and feature in train.columns
                        and feature in valid.columns
                    )
                ]
                if len(filtered) >= 10:
                    return filtered
        candidate_columns = [
            column
            for column in FEATURE_COLUMNS
            if column in train.columns and column in valid.columns and column not in blacklist
        ]
        return candidate_columns or FEATURE_COLUMNS.copy()

    def _save_model(self, report: TrainingReport) -> None:
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        with self.config.xgb_model_path.open("wb") as handle:
            pickle.dump(self.model, handle)
        if self.model_calibrator is not None:
            with self.config.xgb_calibrator_path.open("wb") as handle:
                pickle.dump(self.model_calibrator, handle)
        if self.lightgbm_model is not None:
            with self.config.lgbm_model_path.open("wb") as handle:
                pickle.dump(self.lightgbm_model, handle)
        if self.lightgbm_calibrator is not None:
            with self.config.lgbm_calibrator_path.open("wb") as handle:
                pickle.dump(self.lightgbm_calibrator, handle)
        if self.short_horizon_model is not None:
            with self.config.xgb_short_model_path.open("wb") as handle:
                pickle.dump(self.short_horizon_model, handle)
        if self.short_horizon_calibrator is not None:
            with self.config.xgb_short_calibrator_path.open("wb") as handle:
                pickle.dump(self.short_horizon_calibrator, handle)
        metadata = report.to_dict()
        metadata["feature_columns"] = self.feature_columns
        metadata["short_feature_columns"] = self.short_feature_columns
        self.config.xgb_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.config.lgbm_metadata_path.write_text(
            json.dumps(
                {
                    "trained_at": report.trained_at,
                    "feature_columns": self.feature_columns,
                    "auc": report.lightgbm_auc,
                    "ensemble_weights": report.ensemble_weights,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load_persisted_model(self) -> None:
        if not self.config.xgb_model_path.exists() or not self.config.xgb_metadata_path.exists():
            return
        try:
            with self.config.xgb_model_path.open("rb") as handle:
                self.model = pickle.load(handle)
            if self.config.xgb_calibrator_path.exists():
                with self.config.xgb_calibrator_path.open("rb") as handle:
                    self.model_calibrator = pickle.load(handle)
            if (
                self.config.feature_flags.native_boosters
                and self.config.feature_flags.lightgbm_ensemble
                and self.config.lgbm_model_path.exists()
            ):
                with self.config.lgbm_model_path.open("rb") as handle:
                    self.lightgbm_model = pickle.load(handle)
            else:
                self.lightgbm_model = None
                self.lightgbm_calibrator = None
            if self.lightgbm_model is not None and self.config.lgbm_calibrator_path.exists():
                with self.config.lgbm_calibrator_path.open("rb") as handle:
                    self.lightgbm_calibrator = pickle.load(handle)
            if self.config.xgb_short_model_path.exists():
                with self.config.xgb_short_model_path.open("rb") as handle:
                    self.short_horizon_model = pickle.load(handle)
            if self.config.xgb_short_calibrator_path.exists():
                with self.config.xgb_short_calibrator_path.open("rb") as handle:
                    self.short_horizon_calibrator = pickle.load(handle)
            metadata = json.loads(self.config.xgb_metadata_path.read_text(encoding="utf-8"))
            self.feature_columns = metadata.get("feature_columns", FEATURE_COLUMNS.copy())
            self.short_feature_columns = metadata.get("short_feature_columns", self.feature_columns.copy())
            self.feature_importance = metadata.get("feature_importance", {})
            self.training_samples = int(metadata.get("training_samples", 0))
            self.validation_samples = int(metadata.get("validation_samples", 0))
            self.blend_weights = {
                "xgb": float(metadata.get("ensemble_weights", {}).get("xgb", 0.6)),
                "lgbm": float(metadata.get("ensemble_weights", {}).get("lgbm", 0.4)),
            }
            self.last_report = TrainingReport(
                trained=bool(metadata.get("trained", True)),
                training_samples=self.training_samples,
                validation_samples=self.validation_samples,
                accuracy=float(metadata.get("accuracy", 0.0)),
                precision=float(metadata.get("precision", 0.0)),
                recall=float(metadata.get("recall", 0.0)),
                auc=float(metadata.get("auc", 0.0)),
                positive_ratio=float(metadata.get("positive_ratio", 0.0)),
                negative_ratio=float(metadata.get("negative_ratio", 0.0)),
                scale_pos_weight=float(metadata.get("scale_pos_weight", 1.0)),
                feature_importance={str(k): float(v) for k, v in metadata.get("feature_importance", {}).items()},
                trained_at=str(metadata.get("trained_at", "")),
                xgb_auc=float(metadata.get("xgb_auc", metadata.get("auc", 0.0))),
                lightgbm_auc=float(metadata.get("lightgbm_auc", 0.0)),
                ensemble_auc=float(metadata.get("ensemble_auc", metadata.get("auc", 0.0))),
                fold_aucs=list(metadata.get("fold_aucs", [])),
                ensemble_weights=self.blend_weights.copy(),
                label_definition=str(metadata.get("label_definition", LABEL_DEFINITION)),
                model_family=str(metadata.get("model_family", "XGB+LGBM")),
                selected_profile=str(metadata.get("selected_profile", "baseline")),
                short_horizon_auc=float(metadata.get("short_horizon_auc", 0.0)),
            )
            self.selected_profile = self.last_report.selected_profile
            self.active_profile_config = {
                "name": self.last_report.selected_profile,
                "blend_weights": self.blend_weights.copy(),
            }
        except Exception:
            LOGGER.debug("Failed to load persisted tree ensemble", exc_info=True)
            self.model = None
            self.short_horizon_model = None
            self.short_horizon_calibrator = None

    def _save_feature_importance_chart(self, importance: Dict[str, float]) -> None:
        if plt is None or not importance:
            return
        ordered = sorted(importance.items(), key=lambda item: item[1], reverse=True)[:10]
        labels = [label for label, _ in reversed(ordered)]
        values = [value * 100.0 for _, value in reversed(ordered)]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(labels, values, color="#1f77b4")
        ax.set_xlabel("Importance (%)")
        ax.set_title("Top 10 Predictive Features")
        fig.tight_layout()
        self.config.feature_importance_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.config.feature_importance_png, dpi=150)
        plt.close(fig)

    def _heuristic_probability(self, frame: pd.DataFrame) -> float:
        data = self._prepare_inference_frame(frame)
        if data.empty:
            return 0.5
        latest = data.iloc[-1]
        score = (
            0.18 * clamp((latest["trend_consistency"] - 0.3) / 0.5, 0.0, 1.0)
            + 0.18 * clamp((latest["price_acceleration"] + 0.03) / 0.10, 0.0, 1.0)
            + 0.16 * clamp((latest["rsi"] - 35.0) / 35.0, 0.0, 1.0)
            + 0.14 * clamp((latest["volume_trend"] - 0.8) / 0.8, 0.0, 1.0)
            + 0.14 * clamp((0.18 - latest["resistance_distance"]) / 0.18, 0.0, 1.0)
            + 0.10 * clamp(latest["earnings_proximity_score"], 0.0, 1.0)
            + 0.10 * clamp(latest["breadth_percentile"], 0.0, 1.0)
        )
        return clamp(float(score), 0.0, 1.0)

    def _balance_training_data(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        sample_dates: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
        positives = int(target.sum())
        negatives = int(len(target) - positives)
        if positives == 0 or negatives == 0:
            return features, target, sample_dates
        positive_ratio = positives / max(len(target), 1)
        minority_ratio = min(positives, negatives) / max(len(target), 1)
        if minority_ratio >= 0.40:
            return features, target, sample_dates
        # SMOTE threshold lowered from 0.15 to 0.20: fires earlier on moderately
        # imbalanced datasets, reducing the need for naive duplication.
        if SMOTE is not None and positive_ratio < 0.20:
            sampler = SMOTE(random_state=42)
            balanced_x, balanced_y = sampler.fit_resample(features, target)
            balanced_dates = None
            if sample_dates is not None and not sample_dates.empty:
                latest_date = pd.to_datetime(sample_dates, utc=True).max()
                synthetic_count = len(balanced_x) - len(features)
                balanced_dates = pd.concat(
                    [
                        pd.Series(pd.to_datetime(sample_dates, utc=True).to_numpy()),
                        pd.Series([latest_date] * max(synthetic_count, 0)),
                    ],
                    ignore_index=True,
                )
            return (
                pd.DataFrame(balanced_x, columns=features.columns),
                pd.Series(balanced_y, name=target.name),
                balanced_dates,
            )
        if positives < negatives:
            positive_frame = features.loc[target == 1]
            needed = negatives - positives
            # Recency-aware bootstrap: when sample_dates is available, oversample
            # preferentially from the most recent 90-day window so synthetic
            # positives resemble the current market regime rather than distributing
            # randomly across all historical positives.
            positive_dates = None
            if sample_dates is not None:
                positive_dates = pd.to_datetime(sample_dates.loc[target == 1], utc=True)
            if positive_dates is not None and not positive_dates.empty:
                latest = positive_dates.max()
                recency_mask = (latest - positive_dates).dt.days <= 90
                recency_pool = positive_frame.loc[recency_mask]
                if len(recency_pool) >= 5:
                    # 70% from recent 90-day window, 30% from full history
                    recent_needed = max(int(needed * 0.70), 1)
                    historical_needed = needed - recent_needed
                    recent_extra = recency_pool.sample(n=recent_needed, replace=True, random_state=42)
                    historical_extra = positive_frame.sample(n=historical_needed, replace=True, random_state=43)
                    extra = pd.concat([recent_extra, historical_extra], axis=0)
                    recent_extra_dates = positive_dates.loc[recency_mask].sample(n=recent_needed, replace=True, random_state=42)
                    historical_extra_dates = positive_dates.sample(n=historical_needed, replace=True, random_state=43)
                    extra_dates = pd.concat([recent_extra_dates, historical_extra_dates]).reset_index(drop=True)
                else:
                    extra = positive_frame.sample(n=needed, replace=True, random_state=42)
                    extra_dates = positive_dates.sample(n=needed, replace=True, random_state=42)
                balanced_x = pd.concat([features, extra], axis=0).reset_index(drop=True)
                balanced_y = pd.Series(
                    np.concatenate([target.to_numpy(), np.ones(len(extra), dtype=int)]),
                    name=target.name,
                )
                balanced_dates = pd.concat(
                    [pd.Series(pd.to_datetime(sample_dates, utc=True).to_numpy()), extra_dates.reset_index(drop=True)],
                    ignore_index=True,
                )
                return balanced_x, balanced_y, balanced_dates
            # Fallback: no date info, sample uniformly from all positives
            extra = positive_frame.sample(n=needed, replace=True, random_state=42)
            balanced_x = pd.concat([features, extra], axis=0).reset_index(drop=True)
            balanced_y = pd.Series(
                np.concatenate([target.to_numpy(), np.ones(len(extra), dtype=int)]),
                name=target.name,
            )
            return balanced_x, balanced_y, sample_dates
        return features, target, sample_dates

    def _recency_sample_weights(self, sample_dates: pd.Series | None, length: int) -> np.ndarray | None:
        if sample_dates is None:
            return None
        dates = pd.to_datetime(sample_dates, utc=True, errors="coerce")
        if len(dates) != length:
            return None
        valid = pd.Series(dates).dropna()
        if valid.empty:
            return None
        latest = valid.max()
        age_days = (latest - pd.Series(dates)).dt.days.fillna(0.0).clip(lower=0.0)
        half_life_days = max(int(self.config.training_recency_half_life_weeks * 7), 7)
        weights = np.exp(-np.log(2.0) * age_days.to_numpy(dtype=float) / float(half_life_days))
        floor = clamp(float(self.config.training_recency_weight_floor), 0.0, 1.0)
        weights = np.clip(weights, floor, 1.0)
        # Apply an additional 25% boost to samples from the most recent 4 weeks; these
        # carry the most current regime information and should be emphasized further.
        recent_boost = np.where(age_days.to_numpy(dtype=float) <= 28.0, 1.25, 1.0)
        return weights * recent_boost

    def _build_sector_rank_frame(self, sector_histories: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        usable = {
            sector: frame["close"]
            for sector, frame in sector_histories.items()
            if not frame.empty and "close" in frame
        }
        if not usable:
            return {}
        close_frame = pd.DataFrame(usable).sort_index().ffill()
        weekly_returns = close_frame / close_frame.shift(5) - 1.0
        rank_frame = weekly_returns.rank(axis=1, pct=True).mul(100.0)
        return {sector: rank_frame[sector].dropna() for sector in rank_frame.columns}

    def _build_regime_series(self, vix_history: pd.DataFrame | None) -> pd.Series:
        if vix_history is None or vix_history.empty:
            return pd.Series(dtype=object)
        close = vix_history["close"].sort_index().ffill()
        labels = np.where(
            close > self.config.thresholds.vix_risk_off,
            "risk_off",
            np.where(close > self.config.thresholds.vix_risk_off - 4, "neutral", "risk_on"),
        )
        return pd.Series(labels, index=close.index)

    def _lookup_regime(self, date: pd.Timestamp, regime_series: pd.Series) -> str:
        if regime_series.empty:
            return "neutral"
        eligible = regime_series.loc[regime_series.index <= date]
        if eligible.empty:
            return "neutral"
        return str(eligible.iloc[-1])

    def _weeks_since_earnings(self, index: pd.Index, earnings_dates: List[Dict[str, object]]) -> pd.Series:
        if not earnings_dates:
            return pd.Series(0.0, index=index)
        dates = sorted(
            [
                pd.to_datetime(
                    row.get("earnings_date") or row.get("index") or row.get("date"),
                    utc=True,
                    errors="coerce",
                )
                for row in earnings_dates
            ]
        )
        valid_dates = [date for date in dates if pd.notna(date)]
        if not valid_dates:
            return pd.Series(0.0, index=index)
        earnings_index = pd.DatetimeIndex(valid_dates)
        result = []
        for current in index:
            prior = earnings_index[earnings_index <= current]
            if len(prior) == 0:
                result.append(0.0)
                continue
            result.append(float((current - prior[-1]).days / 7.0))
        return pd.Series(result, index=index)

    def _days_to_next_earnings(self, index: pd.Index, earnings_dates: List[Dict[str, object]]) -> pd.Series:
        if not earnings_dates:
            return pd.Series(30.0, index=index)
        dates = sorted(
            [
                pd.to_datetime(
                    row.get("earnings_date") or row.get("index") or row.get("date"),
                    utc=True,
                    errors="coerce",
                )
                for row in earnings_dates
            ]
        )
        valid_dates = [date for date in dates if pd.notna(date)]
        if not valid_dates:
            return pd.Series(30.0, index=index)
        earnings_index = pd.DatetimeIndex(valid_dates)
        result = []
        for current in index:
            future = earnings_index[earnings_index >= current]
            if len(future) == 0:
                result.append(30.0)
                continue
            result.append(float(max((future[0] - current).days, 0)))
        return pd.Series(result, index=index)

    def _earnings_proximity_score(self, index: pd.Index, earnings_dates: List[Dict[str, object]]) -> pd.Series:
        days_to_earnings = self._days_to_next_earnings(index, earnings_dates)
        return (1.0 - (days_to_earnings / 30.0)).clip(lower=0.0, upper=1.0)

    def _build_forward_benchmark_returns(self, benchmark_history: pd.DataFrame | None) -> pd.Series:
        return self._build_benchmark_returns_for_horizon(benchmark_history, horizon_days=self.horizon_days)

    def _build_benchmark_returns_for_horizon(
        self,
        benchmark_history: pd.DataFrame | None,
        *,
        horizon_days: int,
    ) -> pd.Series:
        if benchmark_history is None or benchmark_history.empty or "close" not in benchmark_history:
            return pd.Series(dtype=float)
        close = benchmark_history["close"].sort_index().ffill()
        return close.shift(-horizon_days) / close - 1.0

    def _build_breadth_series(self, breadth_history: pd.Series | pd.DataFrame | None) -> pd.Series:
        if breadth_history is None:
            return pd.Series(dtype=float)
        if isinstance(breadth_history, pd.Series):
            return breadth_history.sort_index().ffill().clip(lower=0.0, upper=1.0)
        if breadth_history.empty:
            return pd.Series(dtype=float)
        if "close" in breadth_history:
            return (breadth_history["close"].sort_index().ffill() / 100.0).clip(lower=0.0, upper=1.0)
        if len(breadth_history.columns) == 1:
            return breadth_history.iloc[:, 0].sort_index().ffill().clip(lower=0.0, upper=1.0)
        return pd.Series(dtype=float)

    def _fundamental_feature_map(self, tickers: Iterable[str]) -> Dict[str, pd.DataFrame]:
        if self._fundamental_feature_cache is not None:
            return self._fundamental_feature_cache
        empty_timeline = pd.DataFrame(columns=FUNDAMENTAL_FEATURE_COLUMNS)
        try:
            cache = SQLiteCache(self.config.cache_db)
            fetcher = MarketDataFetcher(self.config, cache)
            sec_client = SECCompanyFactsClient(self.config, cache)
            sec_map = fetcher.fetch_sec_ticker_map(fresh=False)
        except Exception:
            LOGGER.debug("Fundamental map initialization failed", exc_info=True)
            self._fundamental_feature_cache = {ticker: empty_timeline.copy() for ticker in tickers}
            return self._fundamental_feature_cache
        result: Dict[str, pd.DataFrame] = {}
        for ticker in sorted(set(tickers)):
            try:
                record = sec_map.get(str(ticker).upper(), {})
                cik = record.get("cik_str")
                if cik:
                    facts = sec_client.fetch_company_facts(str(cik), fresh=False)
                    timeline = sec_client.build_feature_timeline(str(ticker).upper(), facts)
                    result[str(ticker)] = timeline
                else:
                    result[str(ticker)] = empty_timeline.copy()
            except Exception:
                LOGGER.debug("Fundamental snapshot unavailable for %s", ticker, exc_info=True)
                result[str(ticker)] = empty_timeline.copy()
        self._fundamental_feature_cache = result
        return result

    def _fundamental_features_for_date(
        self,
        ticker: str,
        sample_date: pd.Timestamp,
        feature_map: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        defaults = {feature: 0.0 for feature in FUNDAMENTAL_FEATURE_COLUMNS}
        timeline = feature_map.get(str(ticker))
        if timeline is None or timeline.empty:
            return defaults
        sample_ts = pd.Timestamp(sample_date)
        if sample_ts.tzinfo is None:
            sample_ts = sample_ts.tz_localize("UTC")
        else:
            sample_ts = sample_ts.tz_convert("UTC")
        # Use a reporting lag buffer to avoid leaking newly published statements.
        lagged_cutoff = sample_ts - pd.Timedelta(days=45)
        eligible = timeline.loc[timeline.index <= lagged_cutoff]
        if eligible.empty:
            return defaults
        latest = eligible.iloc[-1]
        for feature in FUNDAMENTAL_FEATURE_COLUMNS:
            defaults[feature] = float(latest.get(feature, 0.0))
        return defaults

    def _macro_feature_frame(self) -> pd.DataFrame:
        if self._macro_feature_frame_cache is not None:
            return self._macro_feature_frame_cache
        columns = MACRO_FEATURE_COLUMNS
        empty = pd.DataFrame(columns=columns)
        if not self.config.fred_api_key:
            self._macro_feature_frame_cache = empty
            return empty
        try:
            cache = SQLiteCache(self.config.cache_db)
            client = FREDMacroClient(self.config, cache)
            series_map = {
                "macro_rates_level": "DGS10",
                "macro_curve_slope": "T10Y2Y",
                "macro_credit_spread": "BAMLH0A0HYM2",
                "macro_inflation_expectation": "T5YIE",
                "macro_unemployment": "UNRATE",
                "macro_policy_rate": "FEDFUNDS",
            }
            rows: Dict[str, pd.Series] = {}
            for feature, series_id in series_map.items():
                observations = client.fetch_observations(series_id, fresh=False)
                values = {
                    pd.to_datetime(item.get("date"), utc=True, errors="coerce"): pd.to_numeric(
                        item.get("value"), errors="coerce"
                    )
                    for item in observations
                    if item.get("date")
                }
                if not values:
                    continue
                series = pd.Series(values).dropna().sort_index()
                if series.empty:
                    continue
                # 5-day normalized momentum-style transform for model stability.
                series = (series - series.rolling(252, min_periods=20).mean()) / series.rolling(252, min_periods=20).std()
                rows[feature] = series.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
            if not rows:
                self._macro_feature_frame_cache = empty
                return empty
            frame = pd.DataFrame(rows).sort_index().ffill().fillna(0.0)
            stress_score = (
                0.45 * frame.get("macro_credit_spread", 0.0)
                + 0.30 * (-frame.get("macro_curve_slope", 0.0))
                + 0.15 * frame.get("macro_unemployment", 0.0)
                + 0.10 * frame.get("macro_rates_level", 0.0)
            )
            cooling_score = (
                0.40 * (-frame.get("macro_curve_slope", 0.0))
                + 0.20 * frame.get("macro_policy_rate", 0.0)
                + 0.20 * frame.get("macro_inflation_expectation", 0.0)
                + 0.20 * frame.get("macro_rates_level", 0.0)
            )
            regime_code = pd.Series(0, index=frame.index, dtype=float)
            regime_code = regime_code.where(stress_score <= 0.85, 2.0)
            regime_code = regime_code.where(~((stress_score <= 0.85) & (cooling_score > 0.55)), 1.0)
            frame["macro_regime_expansion_flag"] = (regime_code == 0.0).astype(float)
            frame["macro_regime_cooling_flag"] = (regime_code == 1.0).astype(float)
            frame["macro_regime_stress_flag"] = (regime_code == 2.0).astype(float)
            regime_delta = regime_code.diff().fillna(0.0)
            frame["macro_regime_transition_up"] = regime_delta.clip(lower=0.0, upper=2.0)
            frame["macro_regime_transition_down"] = (-regime_delta).clip(lower=0.0, upper=2.0)
            for feature in columns:
                if feature not in frame:
                    frame[feature] = 0.0
            self._macro_feature_frame_cache = frame[columns]
            return self._macro_feature_frame_cache
        except Exception:
            LOGGER.debug("Macro feature frame unavailable", exc_info=True)
            self._macro_feature_frame_cache = empty
            return empty

    def _macro_features_for_date(self, sample_date: pd.Timestamp, macro_frame: pd.DataFrame) -> Dict[str, float]:
        defaults = {feature: 0.0 for feature in MACRO_FEATURE_COLUMNS}
        if macro_frame.empty:
            return defaults
        sample_ts = pd.Timestamp(sample_date)
        if sample_ts.tzinfo is None:
            sample_ts = sample_ts.tz_localize("UTC")
        else:
            sample_ts = sample_ts.tz_convert("UTC")
        eligible = macro_frame.loc[macro_frame.index <= sample_ts]
        if eligible.empty:
            return defaults
        latest = eligible.iloc[-1]
        for feature in MACRO_FEATURE_COLUMNS:
            defaults[feature] = float(latest.get(feature, 0.0))
        return defaults

    def _future_high_return(self, high: pd.Series, close: pd.Series) -> pd.Series:
        # Stack next N days of highs and compute the max reachable price.
        # Clip at 50% to prevent extreme gap-up events (earnings blowouts, acquisitions)
        # from dominating the label distribution; a 300% return in one week shouldn't
        # carry 100x more label weight than a typical 5% breakout.
        future_high = pd.concat([high.shift(-offset) for offset in range(1, self.label_window_days + 1)], axis=1).max(axis=1)
        raw_return = future_high / close.replace(0, np.nan) - 1.0
        return raw_return.clip(upper=0.50)

    def _walk_forward_cv(self, dataset: pd.DataFrame, *, profile: Dict[str, object] | None = None) -> Dict[str, object]:
        frame = dataset.copy()
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        frame["label_end_date"] = pd.to_datetime(frame["label_end_date"], utc=True)
        rows: List[Dict[str, object]] = []
        month_count = int(frame["date"].dt.tz_localize(None).dt.to_period("M").nunique())
        warmup_months = min(12, max(4, month_count - 2))
        splits = list(
            monthly_purged_splits(
                frame,
                min_train_rows=200,
                warmup_months=warmup_months,
                embargo_days=self.config.training_embargo_days,
            )
        )
        if not splits:
            # Fallback for compact datasets: keep temporal order with an explicit gap.
            fallback = TimeSeriesSplit(
                n_splits=min(5, max(2, len(frame) // 600 if len(frame) >= 1200 else 2)),
                gap=max(int(self.config.training_embargo_days), 1),
            )
            splits = [
                PurgedSplit(
                    fold=idx + 1,
                    train_index=frame.index[train_idx],
                    test_index=frame.index[test_idx],
                    train_rows=len(train_idx),
                    test_rows=len(test_idx),
                    purged_rows=0,
                )
                for idx, (train_idx, test_idx) in enumerate(fallback.split(frame))
            ]
        if not splits:
            return {"rows": rows, "xgb_auc": 0.5, "lightgbm_auc": 0.5, "ensemble_auc": 0.5}
        frame["month"] = frame["date"].dt.tz_localize(None).dt.to_period("M").astype(str)
        def _evaluate_split(split: object) -> Dict[str, object] | None:
            test = frame.loc[split.test_index]
            if test.empty:
                return None
            train = frame.loc[split.train_index]
            if len(train) < 200 or test.empty or train["target"].nunique() < 2 or test["target"].nunique() < 2:
                return None
            feature_columns = [column for column in FEATURE_COLUMNS if column in train.columns and column in test.columns]
            if not feature_columns:
                return None
            pos_count = int(train["target"].sum())
            neg_count = int(len(train) - pos_count)
            scale_pos_weight = float(neg_count / max(pos_count, 1))

            local_profile = profile or self._default_training_profile()
            xgb_model = self._build_model(
                scale_pos_weight=scale_pos_weight,
                overrides={**(local_profile.get("xgb_params") or {}), "n_jobs": 1},
            )
            xgb_train_x, xgb_train_y, xgb_train_dates = self._balance_training_data(
                train[feature_columns],
                train["target"],
                sample_dates=train["date"],
            )
            xgb_model.fit(
                xgb_train_x,
                xgb_train_y,
                **self._fit_kwargs(xgb_model, xgb_train_y, scale_pos_weight, sample_dates=xgb_train_dates),
            )
            xgb_probs = self._predict_model_probabilities(xgb_model, test[feature_columns], None)
            xgb_auc = float(roc_auc_score(test["target"], xgb_probs))

            lightgbm_auc = 0.5
            lightgbm_probs = np.full(len(test), 0.5, dtype=float)
            lightgbm_available = False
            lightgbm_model = self._build_lightgbm_model(
                scale_pos_weight=scale_pos_weight,
                overrides={**(local_profile.get("lgbm_params") or {}), "n_jobs": 1},
                enabled=bool(local_profile.get("use_lightgbm", True)),
            )
            if lightgbm_model is not None:
                lgbm_train_x, lgbm_train_y, lgbm_train_dates = self._balance_training_data(
                    train[feature_columns],
                    train["target"],
                    sample_dates=train["date"],
                )
                lightgbm_model.fit(
                    lgbm_train_x,
                    lgbm_train_y,
                    **self._fit_kwargs(lightgbm_model, lgbm_train_y, scale_pos_weight, sample_dates=lgbm_train_dates),
                )
                lightgbm_probs = self._predict_model_probabilities(lightgbm_model, test[feature_columns], None)
                lightgbm_auc = float(roc_auc_score(test["target"], lightgbm_probs))
                lightgbm_available = True

            weights = self._resolve_blend_weights(local_profile, xgb_auc, lightgbm_auc, lightgbm_available=lightgbm_available)
            ensemble_probs = weights["xgb"] * xgb_probs + weights["lgbm"] * lightgbm_probs
            ensemble_auc = float(roc_auc_score(test["target"], ensemble_probs))
            trade_metrics = self._evaluate_trade_basket(test, ensemble_probs)
            return {
                "fold": split.fold,
                "month": str(test["month"].iloc[0]),
                "purged_rows": split.purged_rows,
                "auc": round(ensemble_auc, 4),
                "xgb_auc": round(xgb_auc, 4),
                "lightgbm_auc": round(lightgbm_auc, 4),
                "trade_win_rate": round(float(trade_metrics["trade_win_rate"]), 4),
                "trade_stop_rate": round(float(trade_metrics["trade_stop_rate"]), 4),
                "trade_average_return": round(float(trade_metrics["trade_average_return"]), 4),
                "trade_average_excess_return": round(float(trade_metrics["trade_average_excess_return"]), 4),
            }

        from concurrent.futures import ThreadPoolExecutor
        workers = max(1, int(getattr(self.config, "training_cv_workers", 1)))
        if workers == 1:
            results = [_evaluate_split(split) for split in splits]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(_evaluate_split, splits))
        
        for res in results:
            if res is not None:
                rows.append(res)
                
        if not rows:
            return {
                "rows": rows,
                "xgb_auc": 0.5,
                "lightgbm_auc": 0.5,
                "ensemble_auc": 0.5,
                "trade_win_rate": 0.0,
                "trade_stop_rate": 0.0,
                "trade_average_return": 0.0,
                "trade_average_excess_return": 0.0,
            }
        return {
            "rows": rows,
            "xgb_auc": float(np.mean([row["xgb_auc"] for row in rows])),
            "lightgbm_auc": float(np.mean([row["lightgbm_auc"] for row in rows])),
            "ensemble_auc": float(np.mean([row["auc"] for row in rows])),
            "trade_win_rate": float(np.mean([row["trade_win_rate"] for row in rows])),
            "trade_stop_rate": float(np.mean([row["trade_stop_rate"] for row in rows])),
            "trade_average_return": float(np.mean([row["trade_average_return"] for row in rows])),
            "trade_average_excess_return": float(np.mean([row["trade_average_excess_return"] for row in rows])),
        }

    def _default_training_profile(self) -> Dict[str, object]:
        return {
            "name": "baseline",
            "xgb_params": {},
            "lgbm_params": {},
            "use_lightgbm": True,
            "blend_weights": None,
        }

    def _candidate_training_profiles(self) -> List[Dict[str, object]]:
        profiles = [
            self._default_training_profile(),
            {
                "name": "xgb_only_recent",
                "xgb_params": {
                    "max_depth": 5,
                    "n_estimators": 520,
                    "learning_rate": 0.025,
                    "subsample": 0.9,
                    "colsample_bytree": 0.55,
                    "min_child_weight": 5,
                    "gamma": 0.08,
                    "reg_alpha": 0.4,
                    "reg_lambda": 2.5,
                },
                "lgbm_params": {},
                "use_lightgbm": False,
                "blend_weights": {"xgb": 1.0, "lgbm": 0.0},
            },
            {
                "name": "regularized_ensemble",
                "xgb_params": {
                    "max_depth": 4,
                    "n_estimators": 560,
                    "learning_rate": 0.022,
                    "subsample": 0.9,
                    "colsample_bytree": 0.5,
                    "min_child_weight": 6,
                    "gamma": 0.10,
                    "reg_alpha": 0.6,
                    "reg_lambda": 3.0,
                },
                "lgbm_params": {
                    "n_estimators": 520,
                    "learning_rate": 0.024,
                    "num_leaves": 31,
                    "subsample": 0.9,
                    "colsample_bytree": 0.55,
                    "min_child_samples": 40,
                    "reg_alpha": 0.4,
                    "reg_lambda": 3.0,
                },
                "use_lightgbm": True,
                "blend_weights": {"xgb": 0.7, "lgbm": 0.3},
            },
            {
                "name": "higher_capacity_ensemble",
                "xgb_params": {
                    "max_depth": 6,
                    "n_estimators": 480,
                    "learning_rate": 0.025,
                    "subsample": 0.82,
                    "colsample_bytree": 0.7,
                    "min_child_weight": 5,
                    "gamma": 0.03,
                    "reg_alpha": 0.25,
                    "reg_lambda": 2.0,
                },
                "lgbm_params": {
                    "n_estimators": 480,
                    "learning_rate": 0.025,
                    "num_leaves": 63,
                    "subsample": 0.82,
                    "colsample_bytree": 0.7,
                    "min_child_samples": 24,
                    "reg_alpha": 0.2,
                    "reg_lambda": 2.0,
                },
                "use_lightgbm": True,
                "blend_weights": {"xgb": 0.65, "lgbm": 0.35},
            },
            {
                "name": "slow_xgb_focus",
                "xgb_params": {
                    "max_depth": 4,
                    "n_estimators": 900,
                    "learning_rate": 0.015,
                    "subsample": 0.88,
                    "colsample_bytree": 0.48,
                    "min_child_weight": 7,
                    "gamma": 0.12,
                    "reg_alpha": 0.8,
                    "reg_lambda": 4.0,
                },
                "lgbm_params": {},
                "use_lightgbm": False,
                "blend_weights": {"xgb": 1.0, "lgbm": 0.0},
            },
            {
                "name": "slow_balanced_ensemble",
                "xgb_params": {
                    "max_depth": 4,
                    "n_estimators": 820,
                    "learning_rate": 0.016,
                    "subsample": 0.88,
                    "colsample_bytree": 0.5,
                    "min_child_weight": 7,
                    "gamma": 0.10,
                    "reg_alpha": 0.7,
                    "reg_lambda": 3.5,
                },
                "lgbm_params": {
                    "n_estimators": 760,
                    "learning_rate": 0.018,
                    "num_leaves": 39,
                    "subsample": 0.88,
                    "colsample_bytree": 0.5,
                    "min_child_samples": 50,
                    "reg_alpha": 0.5,
                    "reg_lambda": 3.5,
                },
                "use_lightgbm": True,
                "blend_weights": {"xgb": 0.75, "lgbm": 0.25},
            },
            {
                "name": "shallow_robust_ensemble",
                "xgb_params": {
                    "max_depth": 3,
                    "n_estimators": 760,
                    "learning_rate": 0.018,
                    "subsample": 0.92,
                    "colsample_bytree": 0.45,
                    "min_child_weight": 8,
                    "gamma": 0.14,
                    "reg_alpha": 0.9,
                    "reg_lambda": 4.5,
                },
                "lgbm_params": {
                    "n_estimators": 680,
                    "learning_rate": 0.02,
                    "num_leaves": 31,
                    "subsample": 0.9,
                    "colsample_bytree": 0.48,
                    "min_child_samples": 55,
                    "reg_alpha": 0.6,
                    "reg_lambda": 4.0,
                },
                "use_lightgbm": True,
                "blend_weights": {"xgb": 0.8, "lgbm": 0.2},
            },
            {
                "name": "shallow_xgb_dominant",
                "xgb_params": {
                    "max_depth": 3,
                    "n_estimators": 820,
                    "learning_rate": 0.017,
                    "subsample": 0.92,
                    "colsample_bytree": 0.44,
                    "min_child_weight": 8,
                    "gamma": 0.15,
                    "reg_alpha": 1.0,
                    "reg_lambda": 4.8,
                },
                "lgbm_params": {
                    "n_estimators": 620,
                    "learning_rate": 0.021,
                    "num_leaves": 31,
                    "subsample": 0.88,
                    "colsample_bytree": 0.46,
                    "min_child_samples": 60,
                    "reg_alpha": 0.7,
                    "reg_lambda": 4.2,
                },
                "use_lightgbm": True,
                "blend_weights": {"xgb": 0.9, "lgbm": 0.1},
            },
            # --- v2 profiles: tuned for the expanded 38-feature core set ---
            {
                "name": "wide_feature_ensemble",
                # Higher colsample_bytree (0.75) samples ~28 of the 38 features per tree,
                # which better exploits the new v2 signals without ignoring them by sampling
                # too few columns.  Pairs with LightGBM using more leaves for richer splits.
                "xgb_params": {
                    "max_depth": 5,
                    "n_estimators": 500,
                    "learning_rate": 0.025,
                    "subsample": 0.85,
                    "colsample_bytree": 0.75,
                    "min_child_weight": 5,
                    "gamma": 0.06,
                    "reg_alpha": 0.35,
                    "reg_lambda": 2.2,
                },
                "lgbm_params": {
                    "n_estimators": 480,
                    "learning_rate": 0.026,
                    "num_leaves": 55,
                    "subsample": 0.85,
                    "colsample_bytree": 0.75,
                    "min_child_samples": 28,
                    "reg_alpha": 0.3,
                    "reg_lambda": 2.2,
                },
                "use_lightgbm": True,
                "blend_weights": {"xgb": 0.65, "lgbm": 0.35},
            },
            {
                "name": "feature_rich_lgbm",
                # LightGBM-dominant profile: more leaves + wider colsample lets it discover
                # non-linear interactions between the new momentum/volatility features.
                "xgb_params": {
                    "max_depth": 5,
                    "n_estimators": 460,
                    "learning_rate": 0.024,
                    "subsample": 0.83,
                    "colsample_bytree": 0.72,
                    "min_child_weight": 5,
                    "gamma": 0.07,
                    "reg_alpha": 0.32,
                    "reg_lambda": 2.3,
                },
                "lgbm_params": {
                    "n_estimators": 560,
                    "learning_rate": 0.022,
                    "num_leaves": 71,
                    "subsample": 0.83,
                    "colsample_bytree": 0.78,
                    "min_child_samples": 22,
                    "reg_alpha": 0.28,
                    "reg_lambda": 1.9,
                },
                "use_lightgbm": True,
                "blend_weights": {"xgb": 0.40, "lgbm": 0.60},
            },
        ]
        profiles.extend(self._random_training_profiles())
        return profiles

    def _random_training_profiles(self) -> List[Dict[str, object]]:
        """Generate diverse parameter candidates for temporal CV search."""

        trials = max(int(getattr(self.config, "training_search_trials", 0)), 0)
        if trials <= 0:
            return []
        rng = np.random.default_rng(int(getattr(self.config, "training_search_seed", 42)))
        candidates: List[Dict[str, object]] = []
        for idx in range(trials):
            use_lightgbm = bool(rng.random() > 0.18) and LGBMClassifier is not None
            xgb_weight = float(rng.uniform(0.58, 0.96)) if use_lightgbm else 1.0
            blend = {"xgb": xgb_weight, "lgbm": 1.0 - xgb_weight} if use_lightgbm else {"xgb": 1.0, "lgbm": 0.0}
            xgb_params = {
                "max_depth": int(rng.integers(3, 8)),
                "n_estimators": int(rng.integers(240, 980)),
                "learning_rate": float(rng.uniform(0.012, 0.06)),
                "subsample": float(rng.uniform(0.70, 0.98)),
                # Extended upper bound: with 38 features, higher colsample_bytree
                # (up to 0.95) allows trees to sample ~36 features - important for
                # discovering interactions among the new v2 signals.
                "colsample_bytree": float(rng.uniform(0.38, 0.95)),
                "min_child_weight": int(rng.integers(2, 12)),
                "gamma": float(rng.uniform(0.0, 0.22)),
                "reg_alpha": float(rng.uniform(0.0, 1.3)),
                "reg_lambda": float(rng.uniform(1.2, 5.2)),
            }
            lgbm_params = {
                "n_estimators": int(rng.integers(240, 980)),
                "learning_rate": float(rng.uniform(0.012, 0.06)),
                # Extended upper bound: more leaves let LightGBM model complex feature
                # interactions from the 64-feature (core+fundamental+macro) total set.
                "num_leaves": int(rng.integers(24, 110)),
                "subsample": float(rng.uniform(0.70, 0.98)),
                "colsample_bytree": float(rng.uniform(0.38, 0.95)),
                "min_child_samples": int(rng.integers(16, 82)),
                "reg_alpha": float(rng.uniform(0.0, 1.3)),
                "reg_lambda": float(rng.uniform(1.2, 5.2)),
            }
            candidates.append(
                {
                    "name": f"random_search_{idx + 1:02d}",
                    "xgb_params": xgb_params,
                    "lgbm_params": lgbm_params,
                    "use_lightgbm": use_lightgbm,
                    "blend_weights": blend,
                }
            )
        return candidates

    def _profile_objective(self, cv_result: Dict[str, object]) -> float:
        rows = list(cv_result.get("rows", []))
        ensemble_auc = float(cv_result.get("ensemble_auc", 0.5))
        if not rows:
            return ensemble_auc
        recent_folds = max(int(self.config.training_profile_recent_folds), 1)
        recent_rows = rows[-recent_folds:]
        recent_auc = float(np.mean([float(row.get("auc", ensemble_auc)) for row in recent_rows]))
        recent_floor = float(min(float(row.get("auc", ensemble_auc)) for row in recent_rows))
        xgb_auc = float(cv_result.get("xgb_auc", ensemble_auc))
        lightgbm_auc = float(cv_result.get("lightgbm_auc", ensemble_auc))
        recent_trade_win_rate = float(np.mean([float(row.get("trade_win_rate", 0.0)) for row in recent_rows]))
        recent_trade_stop_rate = float(np.mean([float(row.get("trade_stop_rate", 0.0)) for row in recent_rows]))
        recent_trade_average_return = float(np.mean([float(row.get("trade_average_return", 0.0)) for row in recent_rows]))
        overall_trade_win_rate = float(cv_result.get("trade_win_rate", 0.0))
        trade_return_score = float(clamp((recent_trade_average_return + 0.02) / 0.05, 0.0, 1.0))
        stop_penalty = clamp(recent_trade_stop_rate * 0.08, 0.0, 0.06)
        # Penalize profiles where XGB and LGBM diverge strongly; high model spread increases
        # uncertainty and causes more picks to be gated out at inference time.
        model_spread_penalty = clamp(abs(xgb_auc - lightgbm_auc) * 0.30, 0.0, 0.05)
        # Favor true temporal AUC first, then trade consistency as a tiebreaker.
        return (
            0.58 * ensemble_auc
            + 0.16 * recent_auc
            + 0.08 * recent_floor
            + 0.08 * xgb_auc
            + 0.06 * recent_trade_win_rate
            + 0.02 * overall_trade_win_rate
            + 0.02 * trade_return_score
            - model_spread_penalty
            - stop_penalty
        )

    def _search_training_profile(self, dataset: pd.DataFrame) -> tuple[Dict[str, object], Dict[str, object]]:
        best_profile = self._default_training_profile()
        best_cv = self._walk_forward_cv(dataset, profile=best_profile)
        best_score = self._profile_objective(best_cv)
        LOGGER.info(
            "Training profile %s objective %.4f (ensemble AUC %.4f)",
            best_profile["name"],
            best_score,
            float(best_cv.get("ensemble_auc", 0.5)),
        )
        ranked = []
        for profile in self._candidate_training_profiles()[1:]:
            cv_result = self._walk_forward_cv(dataset, profile=profile)
            objective = self._profile_objective(cv_result)
            ranked.append(
                (
                    objective,
                    float(cv_result.get("ensemble_auc", 0.5)),
                    str(profile.get("name", "candidate")),
                )
            )
            LOGGER.info(
                "Training profile %s objective %.4f (ensemble AUC %.4f)",
                profile["name"],
                objective,
                float(cv_result.get("ensemble_auc", 0.5)),
            )
            if objective > best_score:
                best_profile = profile
                best_cv = cv_result
                best_score = objective
        if ranked:
            top = sorted(ranked, key=lambda item: (item[0], item[1]), reverse=True)[:5]
            LOGGER.info(
                "Top profile leaderboard: %s",
                " | ".join(f"{name}: obj={obj:.4f}, auc={auc:.4f}" for obj, auc, name in top),
            )
        return best_profile, best_cv

    def _weekly_sample_indices(self, index: pd.Index) -> List[int]:
        sample_indices: List[int] = []
        last_sample_date: pd.Timestamp | None = None
        for position, timestamp in enumerate(index):
            if getattr(timestamp, "dayofweek", None) != self.config.training_sample_weekday:
                continue
            if last_sample_date is not None:
                embargo_boundary = last_sample_date + BDay(self.config.training_embargo_days)
                if timestamp < embargo_boundary:
                    continue
            sample_indices.append(position)
            last_sample_date = timestamp
        return sample_indices

    def _build_weekly_return_targets(
        self,
        close: pd.Series,
        high: pd.Series | None = None,
        low: pd.Series | None = None,
    ) -> pd.DataFrame:
        if close.empty:
            return pd.DataFrame(columns=["weekly_return", "label_start_date", "label_end_date"])
        weekly_close = close.sort_index().resample("W-FRI").last().dropna()
        if len(weekly_close) < 2:
            return pd.DataFrame(columns=["weekly_return", "label_start_date", "label_end_date"])
        targets = pd.DataFrame(index=weekly_close.index)
        targets["weekly_return"] = weekly_close / weekly_close.shift(1) - 1.0
        targets["label_start_date"] = weekly_close.index.to_series().shift(1)
        targets["label_end_date"] = weekly_close.index
        targets["future_return_5d"] = targets["weekly_return"]
        targets["future_return_10d"] = weekly_close / weekly_close.shift(2) - 1.0
        targets["max_favorable_excursion"] = targets["weekly_return"].clip(lower=0.0)
        targets["max_adverse_excursion"] = targets["weekly_return"].clip(upper=0.0)
        profit_target = max(float(getattr(self.config, "weekly_profit_target", 0.06)), 0.001)
        stop_loss = max(float(getattr(self.config, "weekly_stop_loss", 0.035)), 0.001)
        targets["hit_target_before_stop"] = targets["max_favorable_excursion"] >= profit_target
        targets["stopped_before_target"] = targets["max_adverse_excursion"] <= -stop_loss
        targets["ambiguous_barrier_hit"] = False
        if high is not None and low is not None and not high.empty and not low.empty:
            ordered_high = high.sort_index()
            ordered_low = low.sort_index()
            for label_end, row in targets.iterrows():
                label_start = row["label_start_date"]
                entry_price = float(weekly_close.shift(1).get(label_end, 0.0) or 0.0)
                if entry_price <= 0:
                    continue
                window_high = ordered_high.loc[(ordered_high.index > label_start) & (ordered_high.index <= label_end)]
                window_low = ordered_low.loc[(ordered_low.index > label_start) & (ordered_low.index <= label_end)]
                if window_high.empty or window_low.empty:
                    continue
                favorable = float(window_high.max() / entry_price - 1.0)
                adverse = float(window_low.min() / entry_price - 1.0)
                hit_date = self._first_threshold_date(window_high, entry_price * (1.0 + profit_target), above=True)
                stop_date = self._first_threshold_date(window_low, entry_price * (1.0 - stop_loss), above=False)
                ambiguous = hit_date is not None and stop_date is not None and hit_date == stop_date
                hit_before_stop = hit_date is not None and (stop_date is None or hit_date < stop_date)
                stopped_before_target = stop_date is not None and (hit_date is None or stop_date <= hit_date)
                targets.at[label_end, "max_favorable_excursion"] = favorable
                targets.at[label_end, "max_adverse_excursion"] = adverse
                targets.at[label_end, "hit_target_before_stop"] = bool(hit_before_stop)
                targets.at[label_end, "stopped_before_target"] = bool(stopped_before_target)
                targets.at[label_end, "ambiguous_barrier_hit"] = bool(ambiguous)
        targets["risk_adjusted_return"] = targets["weekly_return"] / targets["max_adverse_excursion"].abs().clip(lower=0.01)
        return targets.dropna(subset=["weekly_return", "label_start_date"])

    @staticmethod
    def _first_threshold_date(series: pd.Series, threshold: float, *, above: bool) -> pd.Timestamp | None:
        if series.empty:
            return None
        mask = series >= threshold if above else series <= threshold
        hits = series.loc[mask]
        if hits.empty:
            return None
        return pd.Timestamp(hits.index[0])

    def _week_end_for_date(self, timestamp: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(timestamp)
        week_end = ts.normalize() + pd.offsets.Week(weekday=4)
        if ts.tzinfo is not None and week_end.tzinfo is None:
            week_end = week_end.tz_localize(ts.tzinfo)
        return week_end

    def _purge_train_rows(self, frame: pd.DataFrame, *, test_start: pd.Timestamp) -> pd.DataFrame:
        return purge_train_frame(
            frame,
            test_start=pd.Timestamp(test_start),
            embargo_days=self.config.training_embargo_days,
        )

    def _blend_weights_from_aucs(self, xgb_auc: float, lightgbm_auc: float, *, lightgbm_available: bool) -> Dict[str, float]:
        if not lightgbm_available:
            return {"xgb": 1.0, "lgbm": 0.0}
        # Soft-max blend: weight proportional to squared excess-AUC above 0.5.
        # This gives smooth, continuous blending instead of stepped thresholds,
        # and naturally down-weights a model that is barely above chance.
        xgb_score = max(xgb_auc - 0.5, 0.0) ** 2
        lgbm_score = max(lightgbm_auc - 0.5, 0.0) ** 2
        total = xgb_score + lgbm_score
        if total < 1e-8:
            # Both models are near chance, so fall back to a balanced split.
            return {"xgb": 0.6, "lgbm": 0.4}
        raw_xgb = xgb_score / total
        # Clamp to [0.25, 0.90] so neither model is completely suppressed.
        xgb_weight = clamp(raw_xgb, 0.25, 0.90)
        return {"xgb": xgb_weight, "lgbm": 1.0 - xgb_weight}

    def _resolve_blend_weights(
        self,
        profile: Dict[str, object] | None,
        xgb_auc: float,
        lightgbm_auc: float,
        *,
        lightgbm_available: bool,
    ) -> Dict[str, float]:
        preferred = (profile or {}).get("blend_weights") if isinstance(profile, dict) else None
        if isinstance(preferred, dict):
            xgb_weight = float(preferred.get("xgb", 0.0))
            lgbm_weight = float(preferred.get("lgbm", 0.0))
            total = xgb_weight + lgbm_weight
            if total > 0 and (lightgbm_available or lgbm_weight == 0.0):
                return {"xgb": xgb_weight / total, "lgbm": lgbm_weight / total}
        return self._blend_weights_from_aucs(xgb_auc, lightgbm_auc, lightgbm_available=lightgbm_available)

    def _train_regime_models(
        self,
        dataset: pd.DataFrame,
        *,
        save_model: bool,
        xgb_overrides: Dict[str, object] | None = None,
    ) -> None:
        if not self.config.feature_flags.regime_specific_model or "regime" not in dataset:
            return
        self.regime_models = {}
        self.regime_feature_columns = {}
        for regime in ["risk_on", "neutral", "risk_off"]:
            subset = dataset.loc[dataset["regime"] == regime].copy()
            if len(subset) < 200 or subset["target"].nunique() < 2:
                continue
            subset = subset.sort_values("date").reset_index(drop=True)
            split = max(int(len(subset) * 0.8), min(100, len(subset) - 1))
            train = subset.iloc[:split]
            valid = subset.iloc[split:]
            if valid.empty:
                continue
            feature_columns = self._select_feature_columns(train, valid)
            pos_count = int(train["target"].sum())
            neg_count = int(len(train) - pos_count)
            scale_pos_weight = float(neg_count / max(pos_count, 1))
            model = self._build_model(scale_pos_weight=scale_pos_weight, overrides=xgb_overrides)
            balanced_x, balanced_y, balanced_dates = self._balance_training_data(
                train[feature_columns],
                train["target"],
                sample_dates=train["date"],
            )
            fit_kwargs = self._fit_kwargs(model, balanced_y, scale_pos_weight, sample_dates=balanced_dates)
            model.fit(balanced_x, balanced_y, **fit_kwargs)
            self.regime_models[regime] = model
            self.regime_feature_columns[regime] = feature_columns
            if save_model:
                self._save_regime_model(regime, model, feature_columns)

    def _save_regime_model(self, regime: str, model, feature_columns: List[str]) -> None:
        model_path, metadata_path = self._regime_paths(regime)
        with model_path.open("wb") as handle:
            pickle.dump(model, handle)
        metadata_path.write_text(json.dumps({"feature_columns": feature_columns}, indent=2), encoding="utf-8")

    def _load_regime_models(self) -> None:
        for regime in ["risk_on", "neutral", "risk_off"]:
            model_path, metadata_path = self._regime_paths(regime)
            if not model_path.exists() or not metadata_path.exists():
                continue
            try:
                with model_path.open("rb") as handle:
                    self.regime_models[regime] = pickle.load(handle)
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.regime_feature_columns[regime] = metadata.get("feature_columns", FEATURE_COLUMNS.copy())
            except Exception:
                LOGGER.debug("Failed to load %s regime model", regime, exc_info=True)

    def _regime_paths(self, regime: str) -> tuple[Path, Path]:
        if regime == "risk_on":
            return self.config.xgb_risk_on_path, self.config.xgb_risk_on_metadata_path
        if regime == "risk_off":
            return self.config.xgb_risk_off_path, self.config.xgb_risk_off_metadata_path
        return self.config.xgb_neutral_path, self.config.xgb_neutral_metadata_path
