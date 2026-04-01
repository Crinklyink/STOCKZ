"""Streamlit dashboard V2 for the stock predictor."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from main import run_scan
from stock_predictor.config import get_config
from stock_predictor.output.backtest import BacktestTracker


config = get_config()
backtest = BacktestTracker(config.backtest_db)


def load_latest_payload() -> dict:
    if not config.latest_scan_path.exists():
        return run_scan(fresh=False)
    return json.loads(config.latest_scan_path.read_text(encoding="utf-8"))


def badge_html(label: str) -> str:
    colors = {
        "high": ("#1f7a1f", "#e9f7e9"),
        "medium": ("#8a6d00", "#fff8db"),
        "low": ("#a61b1b", "#fde8e8"),
        "disabled": ("#555555", "#f2f2f2"),
    }
    foreground, background = colors.get(label, ("#555555", "#f2f2f2"))
    return (
        f"<span style='padding:0.2rem 0.55rem;border-radius:999px;"
        f"background:{background};color:{foreground};font-weight:600'>{label.upper()}</span>"
    )


def build_score_band(row: pd.Series) -> str:
    return f"{row['final_score']:.1f} +/- {row.get('score_uncertainty', 0.0):.1f}"


def signal_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "ml_score",
        "technical_score",
        "options_score",
        "sentiment_score",
        "rs_score",
        "institutional_score",
        "news_score",
        "smart_money_score",
        "squeeze_score",
        "gpt_news_score",
        "anomaly_score",
    ]
    return [column for column in candidates if column in frame.columns]


def render_heatmap(frame: pd.DataFrame) -> None:
    signals = signal_columns(frame)
    if not signals:
        st.info("No signal heatmap data available yet.")
        return
    matrix = frame.set_index("ticker")[signals]
    labels = {
        "ml_score": "ML",
        "technical_score": "Tech",
        "options_score": "Options",
        "sentiment_score": "Sentiment",
        "rs_score": "RS",
        "institutional_score": "Institutional",
        "news_score": "News",
        "smart_money_score": "Smart Money",
        "squeeze_score": "Squeeze",
        "gpt_news_score": "GPT News",
        "anomaly_score": "Anomaly",
    }
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.to_numpy(),
            x=[labels.get(column, column) for column in matrix.columns],
            y=matrix.index.tolist(),
            colorscale=[
                [0.0, "#8b0000"],
                [0.25, "#c94c4c"],
                [0.5, "#f3d36b"],
                [0.75, "#52a552"],
                [1.0, "#0c5f25"],
            ],
            zmin=0,
            zmax=100,
            text=matrix.round(1).astype(str).to_numpy(),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)


def render_candlestick(row: pd.Series) -> None:
    historical = pd.DataFrame(row.get("diagnostics", {}).get("price_chart", []))
    if historical.empty:
        st.info("No price chart data available.")
        return
    historical["date"] = pd.to_datetime(historical["date"])
    chart = go.Figure()
    chart.add_trace(
        go.Candlestick(
            x=historical["date"],
            open=historical["open"],
            high=historical["high"],
            low=historical["low"],
            close=historical["close"],
            name=row["ticker"],
        )
    )
    chart.update_layout(height=320, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(chart, use_container_width=True)


def render_detail_metrics(row: pd.Series) -> None:
    metrics = {
        "Pattern win rate": f"{row.get('pattern_win_rate', 0.0):.1f}%",
        "Smart money": f"{row.get('smart_money_score', 0.0):.1f}",
        "Squeeze": f"{row.get('squeeze_score', 0.0):.1f}",
        "Data quality": f"{row.get('data_quality_score', 0.0):.1f}%",
        "GPT news": f"{row.get('gpt_news_score', 0.0):.1f}",
        "Risk/reward": f"{row.get('risk_reward', 0.0):.2f}",
    }
    cols = st.columns(3)
    for index, (label, value) in enumerate(metrics.items()):
        cols[index % 3].metric(label, value)


st.set_page_config(page_title="Elite Stock Predictor", layout="wide")
st.title("Elite 5-Day Stock Predictor")

if st.button("Refresh Full Scan"):
    payload = run_scan(fresh=True)
else:
    payload = load_latest_payload()

selected = pd.DataFrame(payload.get("selected", []))
if selected.empty:
    st.warning("No picks available yet. Run a scan first.")
    st.stop()

macro = payload.get("macro", {})
vix = float(macro.get("vix", 0.0) or 0.0)
if not payload.get("model_trained", True):
    st.warning("Model untrained - using technical signals only.")
if payload.get("qualified_count", 0) == 0:
    st.info("No picks cleared the active threshold. Showing top-ranked near-misses.")
if vix > config.thresholds.kill_switch_vix:
    st.error(
        (
            f"Kill switch warning: VIX is {vix:.2f}, above the configured "
            f"threshold of {config.thresholds.kill_switch_vix:.2f}."
        )
    )

selected["score_band"] = selected.apply(build_score_band, axis=1)
selected["confidence_badge"] = selected["confidence_label"].apply(badge_html)
overview_cols = st.columns(4)
overview_cols[0].metric("Macro regime", str(macro.get("risk_regime", "unknown")).replace("_", " ").title())
overview_cols[1].metric("VIX", f"{vix:.2f}")
overview_cols[2].metric("Runtime", f"{payload.get('runtime_seconds', 0)} sec")
overview_cols[3].metric("Threshold", f"{payload.get('threshold_used', 0):.1f}")

tabs = st.tabs(["Overview", "Signal Heatmap", "Pick Explorer", "Performance"])

with tabs[0]:
    st.subheader("Top 10 Picks")
    table = selected[
        [
            "ticker",
            "company_name",
            "sector",
            "score_band",
            "confidence_label",
            "probability_4pct_5d",
            "current_price",
            "stop_loss",
            "position_size_pct",
            "pattern_name",
        ]
    ].rename(
        columns={
            "company_name": "company",
            "confidence_label": "confidence",
            "probability_4pct_5d": "prob_4pct_5d",
            "current_price": "price",
            "position_size_pct": "position_size",
            "pattern_name": "pattern",
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(payload.get("macro_summary", ""))
    if payload.get("suggested_threshold"):
        suggestion = payload["suggested_threshold"]
        st.caption(
            f"Lower threshold to {suggestion['threshold']:.1f} to include {suggestion['additional_picks']} more picks."
        )

with tabs[1]:
    st.subheader("Live Signal Heatmap")
    render_heatmap(selected)
    st.subheader("Adaptive Weights")
    weights = pd.DataFrame(
        [{"signal": key, "weight": value} for key, value in payload.get("adaptive_weights", {}).items()]
    )
    if not weights.empty:
        st.bar_chart(weights.set_index("signal")["weight"])

with tabs[2]:
    st.subheader("Why These Picks")
    for _, row in selected.iterrows():
        title = (
            f"{row['ticker']} | {row['company_name']} | {row['score_band']} | "
            f"{row['confidence_label'].upper()}"
        )
        with st.expander(title):
            st.markdown(badge_html(row["confidence_label"]), unsafe_allow_html=True)
            st.info(row.get("ai_explanation") or row.get("gpt_news_reason") or "No AI explanation available.")
            render_detail_metrics(row)
            signal_data = pd.DataFrame(
                {
                    "signal": signal_columns(selected),
                    "score": [row.get(column, 0.0) for column in signal_columns(selected)],
                }
            )
            if not signal_data.empty:
                st.bar_chart(signal_data.set_index("signal")["score"])
            render_candlestick(row)
            left, right = st.columns([3, 2])
            with left:
                st.write("Notes")
                for note in row.get("notes", []):
                    st.write(f"- {note}")
            with right:
                st.json(
                    {
                        "targets": row.get("targets", {}),
                        "risk_reward": row.get("risk_reward"),
                        "gpt_news_reason": row.get("gpt_news_reason"),
                        "diagnostics": {
                            "multi_timeframe": row.get("diagnostics", {}).get("multi_timeframe", {}),
                            "smart_money": row.get("diagnostics", {}).get("smart_money", {}),
                            "pattern_history": row.get("diagnostics", {}).get("pattern_history", {}),
                        },
                    }
                )

with tabs[3]:
    st.subheader("Weekly Performance Tracker")
    performance = backtest.performance_frame()
    weekly = backtest.weekly_hit_rate()
    confidence = backtest.confidence_trend()
    if not weekly.empty:
        weekly = weekly.copy()
        weekly["hit_target"] = weekly["hit_target"] * 100
        st.line_chart(weekly.set_index("week")["hit_target"])
    else:
        st.info("Backtest history will appear after at least one evaluation window.")
    if not confidence.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=confidence["week"],
                y=confidence["hit_rate"] * 100,
                mode="lines+markers",
                name="Hit rate %",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=confidence["week"],
                y=confidence["avg_interval_width"],
                mode="lines+markers",
                name="Avg score interval width",
                yaxis="y2",
            )
        )
        fig.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(title="Hit rate %"),
            yaxis2=dict(title="Interval width", overlaying="y", side="right"),
        )
        st.subheader("Confidence Trend")
        st.plotly_chart(fig, use_container_width=True)
    if not performance.empty:
        performance = performance.copy()
        performance["realized_return"] = performance["realized_return"] * 100
        st.subheader("Recent Pick Outcomes")
        st.dataframe(performance.head(50), use_container_width=True, hide_index=True)
