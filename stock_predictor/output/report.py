"""Report generation and export."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

try:  # pragma: no cover - optional dependency
    from tabulate import tabulate
except Exception:  # pragma: no cover
    tabulate = None

from stock_predictor.scoring.composite import CandidateScore


def candidates_to_frame(candidates: Iterable[CandidateScore]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "ticker": candidate.ticker,
                "company": candidate.company_name,
                "sector": candidate.sector,
                "tier": candidate.tier_label,
                "sector_tag": candidate.sector_temperature_tag,
                "score": candidate.final_score,
                "score_low": candidate.score_low,
                "score_high": candidate.score_high,
                "uncertainty": candidate.score_uncertainty,
                "confidence": candidate.confidence_label,
                "prob_4pct_5d": candidate.probability_4pct_5d,
                "pattern": candidate.pattern_name,
                "pattern_win_rate": candidate.pattern_win_rate,
                "confluence": candidate.confluence_count,
                "volume_momentum": candidate.volume_momentum_score,
                "smart_money": candidate.smart_money_score,
                "squeeze": candidate.squeeze_score,
                "gpt_news": candidate.gpt_news_score,
                "price": candidate.current_price,
                "stop_loss": candidate.stop_loss,
                "tp1": candidate.targets["tp1"],
                "tp2": candidate.targets["tp2"],
                "tp3": candidate.targets["tp3"],
                "risk_reward": candidate.risk_reward,
                "position_size_pct": candidate.position_size_pct,
                "kelly_size_pct": candidate.kelly_size_pct,
                "above_score_threshold": candidate.meets_threshold,
                "ai_explanation": candidate.ai_explanation,
                "notes": " | ".join(candidate.notes),
            }
        )
    return pd.DataFrame(rows)


def render_terminal_report(
    candidates: List[CandidateScore],
    *,
    threshold_used: float = 0.0,
    macro_summary: str = "",
    regime_label: str = "unknown",
    model_trained: bool = True,
    qualified_count: int = 0,
    candidate_pool_size: int = 0,
    data_quality_summary: dict | None = None,
    suggestion: dict | None = None,
    summary_header: dict | None = None,
) -> str:
    frame = candidates_to_frame(candidates)
    header_lines = build_scan_summary_header(
        summary_header=summary_header,
        threshold_used=threshold_used,
        regime_label=regime_label,
        macro_summary=macro_summary,
        candidate_pool_size=candidate_pool_size or len(candidates),
        qualified_count=qualified_count,
    )
    if not model_trained:
        header_lines.append("WARNING: Model untrained - using technical signals only")
    if data_quality_summary:
        header_lines.append(
            (
                f"Data quality: {data_quality_summary.get('full_data', 0)} of "
                f"{data_quality_summary.get('total', 0)} tickers had full data; "
                f"{data_quality_summary.get('valid_data', 0)} passed validation"
            )
        )
    if frame.empty:
        return "\n".join(header_lines + ["No ranked candidates were available."])
    frame = frame.copy()
    frame["score_band"] = frame.apply(lambda row: f"{row['score']:.1f} +/- {row['uncertainty']:.1f}", axis=1)
    terminal_columns = [
        "ticker",
        "tier",
        "sector",
        "score_band",
        "confidence",
        "prob_4pct_5d",
        "pattern",
        "pattern_win_rate",
        "price",
        "stop_loss",
        "tp2",
        "risk_reward",
    ]
    display = frame[terminal_columns]
    if tabulate is None:
        table = display.to_string(index=False)
    else:
        table = tabulate(display, headers="keys", tablefmt="github", showindex=False, floatfmt=".2f")
    footer_lines = [f"Above score threshold: {qualified_count}/{candidate_pool_size or len(candidates)}"]
    if qualified_count == 0:
        footer_lines.append("Showing top-ranked watchlist names below the official score threshold.")
    if suggestion:
        footer_lines.append(
            (
                f"Suggestion: lower threshold to {suggestion['threshold']:.1f} to include "
                f"{suggestion['additional_picks']} more picks."
            )
        )
    concentration_warning = build_concentration_warning(candidates)
    if concentration_warning:
        footer_lines.append(concentration_warning)
    score_cards = []
    for candidate in candidates:
        sentiment_star = "*" if candidate.defaulted_signals.get("sentiment") else ""
        flow_star = "*" if candidate.defaulted_signals.get("flow") else ""
        status = "ABOVE SCORE THRESHOLD" if candidate.meets_threshold else "BELOW SCORE THRESHOLD"
        score_cards.append(
            (
                f"{candidate.ticker} | {candidate.tier_label} | {candidate.sector_temperature_tag or 'NEUTRAL SECTOR'} | "
                f"Final: {candidate.final_score:.0f} | Tech: {candidate.technical_score:.0f} | "
                f"RS: {candidate.rs_score:.0f} | ML: {candidate.ml_score:.0f} | Vol: {candidate.volume_momentum_score:.0f} | "
                f"Sentiment: {candidate.sentiment_score:.0f}{sentiment_star} | Flow: {candidate.options_score:.0f}{flow_star} | "
                f"Confluence: {candidate.confluence_count}/5 | "
                f"{status}"
            )
        )
        score_cards.append(generate_pick_explanation(candidate))
    return "\n".join(header_lines + ["", table, "", *footer_lines, "", *score_cards])


def export_reports(
    candidates: List[CandidateScore],
    csv_path: Path,
    md_path: Path,
    html_path: Path,
    *,
    threshold_used: float = 0.0,
    macro_summary: str = "",
    regime_label: str = "unknown",
    model_trained: bool = True,
    qualified_count: int = 0,
    candidate_pool_size: int = 0,
    data_quality_summary: dict | None = None,
    suggestion: dict | None = None,
    summary_header: dict | None = None,
) -> None:
    frame = candidates_to_frame(candidates)
    frame.to_csv(csv_path, index=False)
    md_path.write_text(
        render_terminal_report(
            candidates,
            threshold_used=threshold_used,
            macro_summary=macro_summary,
            regime_label=regime_label,
            model_trained=model_trained,
            qualified_count=qualified_count,
            candidate_pool_size=candidate_pool_size,
            data_quality_summary=data_quality_summary,
            suggestion=suggestion,
            summary_header=summary_header,
        ),
        encoding="utf-8",
    )
    html_path.write_text(frame.to_html(index=False), encoding="utf-8")


def build_scan_summary_header(
    *,
    summary_header: dict | None,
    threshold_used: float,
    regime_label: str,
    macro_summary: str,
    candidate_pool_size: int,
    qualified_count: int,
) -> List[str]:
    if not summary_header:
        lines = [
            f"Macro: {regime_label.upper()} | Threshold: {threshold_used:.1f}",
            macro_summary,
        ]
    else:
        lines = [
            "════════════════════════════════════════════",
            f"📊 WEEKLY STOCK SCAN — {summary_header.get('date', '')}",
            (
                f"Universe: {summary_header.get('universe_total', candidate_pool_size)} tickers | "
                f"Scanned in: {summary_header.get('runtime_seconds', 0):.0f}s"
            ),
            (
                f"Regime: {str(summary_header.get('regime', regime_label)).upper()} | "
                f"VIX: {summary_header.get('vix', 0):.2f} | "
                f"SPY: {summary_header.get('spy_week_return', 0):+.2f}% wk"
            ),
            (
                f"Model: {summary_header.get('model_family', 'XGB+LGBM')} "
                f"({'Trained' if summary_header.get('model_training_samples', 0) else 'Cold'}; "
                f"{summary_header.get('model_training_samples', 0)} samples) | "
                f"AUC: {summary_header.get('model_auc', 0):.3f}"
            ),
            (
                f"Above score threshold: {qualified_count}/{summary_header.get('universe_total', candidate_pool_size)} "
                f"at {summary_header.get('threshold_used', threshold_used):.1f}"
            ),
            (
                f"Top Sector: {summary_header.get('top_sector', 'Unknown')} 🔥 | "
                f"Worst: {summary_header.get('worst_sector', 'Unknown')} ❄️"
            ),
            (
                f"Stage 1: filtered {summary_header.get('universe_total', candidate_pool_size)}→"
                f"{summary_header.get('stage1_survivors', candidate_pool_size)} in "
                f"{summary_header.get('stage1_runtime_seconds', 0):.1f}s | "
                f"Cache: {summary_header.get('cache_warm', 0)}/{summary_header.get('cache_total', 0)} "
                f"warm ({summary_header.get('cache_hit_rate', 0):.1f}%) — "
                f"saved ~{int(summary_header.get('cache_saved_seconds_estimate', 0) // 60)} min of fetch time"
            ),
            "════════════════════════════════════════════",
            macro_summary,
        ]
    if summary_header and summary_header.get("selection_warning"):
        lines.append(str(summary_header["selection_warning"]))
    if summary_header and summary_header.get("regime_memory_text"):
        lines.append(str(summary_header["regime_memory_text"]))
    return lines


def generate_pick_explanation(candidate: CandidateScore) -> str:
    signal_icon = lambda passed: "✅" if passed else "⚪"
    tech_phrase = "Technical momentum is strong" if candidate.technical_score >= 70 else "Technical momentum is constructive"
    volume_phrase = (
        "Volume surged well above average, suggesting institutional participation"
        if candidate.volume_momentum_score >= 80
        else "Volume is supportive but not extreme"
    )
    rsi_phrase = (
        "RSI is pushing higher from a reset zone"
        if candidate.technical_score >= 65
        else "RSI is stable but not yet stretched"
    )
    ml_phrase = f"XGBoost assigns {candidate.probability_4pct_5d:.0f}% probability of a 4%+ move."
    sector_phrase = (
        f"{candidate.sector} is one of the strongest groups this week."
        if "HOT" in candidate.sector_temperature_tag
        else f"{candidate.sector} is trading in line with the market backdrop."
    )
    return "\n".join(
        [
            "┌─────────────────────────────────────────────┐",
            f"│  {candidate.tier_label.split()[0]} {candidate.ticker} — {candidate.company_name[:20]:<20} │",
            f"│  Price: ${candidate.current_price:.2f} | Target: ${candidate.targets['tp2']:.2f} │",
            f"│  Stop: ${candidate.stop_loss:.2f} | R:R: {candidate.risk_reward:.2f}:1               │",
            f"│  Suggested size: {candidate.kelly_size_pct:.1f}% of portfolio         │",
            "├─────────────────────────────────────────────┤",
            "│  WHY THIS PICK:                             │",
            f"│  {sector_phrase[:43]:<43}│",
            f"│  {volume_phrase[:43]:<43}│",
            f"│  {rsi_phrase[:43]:<43}│",
            f"│  {ml_phrase[:43]:<43}│",
            "├─────────────────────────────────────────────┤",
            (
                "│  SIGNALS: "
                f"Tech{signal_icon(candidate.technical_score >= 65)} "
                f"RS{signal_icon(candidate.rs_score >= 70)} "
                f"Vol{signal_icon(candidate.volume_momentum_score >= 65)} "
                f"ML{signal_icon(candidate.ml_score >= 55)} "
                f"Pattern{signal_icon(candidate.pattern_score >= 60)} {candidate.confluence_count}/5 │"
            ),
            "└─────────────────────────────────────────────┘",
        ]
    )


def build_concentration_warning(candidates: List[CandidateScore]) -> str:
    if len(candidates) < 4:
        return ""
    counts = {}
    for candidate in candidates:
        counts[candidate.sector] = counts.get(candidate.sector, 0) + 1
    top_two = sorted(counts.values(), reverse=True)[:2]
    if sum(top_two) >= 7:
        return "⚠️ 7/10 picks are concentrated in the same two sectors — consider reducing position sizes by 50%."
    return ""
