"""Hardcoded supply-chain correlation relationships."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from stock_predictor.utils import clamp


SUPPLY_CHAIN_MAP: Dict[str, List[str]] = {
    "AAPL": ["AVGO", "QCOM", "TXN", "SWKS", "CRUS"],
    "MSFT": ["NVDA", "AMD", "ANET", "AVGO", "ORCL"],
    "NVDA": ["TSM", "ASML", "AMAT", "LRCX", "MU"],
    "AVGO": ["AAPL", "META", "GOOGL", "ORCL", "MSFT"],
    "ORCL": ["NVDA", "AMD", "ANET", "ETN", "VRT"],
    "AMD": ["TSM", "MU", "AMAT", "LRCX", "ASML"],
    "ANET": ["NVDA", "MSFT", "META", "AMZN", "GOOGL"],
    "AMZN": ["NVDA", "ANET", "ORCL", "UPS", "FDX"],
    "META": ["NVDA", "ANET", "AVGO", "AMD", "MSFT"],
    "GOOGL": ["NVDA", "ANET", "AVGO", "AMAT", "LRCX"],
    "TSLA": ["ALB", "SQM", "NUE", "F", "GM"],
    "WMT": ["COST", "PG", "PEP", "KO", "MDLZ"],
    "JPM": ["MS", "GS", "CME", "AXP", "SCHW"],
    "GS": ["JPM", "MS", "AXP", "CME", "SCHW"],
    "LLY": ["UNH", "MRK", "ABBV", "JNJ", "ISRG"],
    "UNH": ["LLY", "JNJ", "MRK", "ABBV", "ISRG"],
    "GE": ["HWM", "RTX", "LMT", "NOC", "ETN"],
    "ETN": ["VRT", "NVDA", "ANET", "ORCL", "GE"],
    "CAT": ["DE", "URI", "FCX", "NUE", "LIN"],
    "DE": ["CAT", "MOS", "NUE", "FCX", "LIN"],
    "HWM": ["GE", "RTX", "LMT", "NOC", "GD"],
    "XOM": ["SLB", "BKR", "CVX", "LNG", "FANG"],
    "CVX": ["XOM", "SLB", "BKR", "LNG", "FANG"],
    "SLB": ["XOM", "CVX", "BKR", "LNG", "FANG"],
    "BKR": ["XOM", "CVX", "SLB", "LNG", "FANG"],
    "LNG": ["XOM", "CVX", "SLB", "BKR", "FANG"],
    "FANG": ["XOM", "CVX", "SLB", "BKR", "LNG"],
    "NEE": ["DUK", "SO", "CEG", "VST", "NRG"],
    "DUK": ["NEE", "SO", "CEG", "VST", "NRG"],
    "SO": ["NEE", "DUK", "CEG", "VST", "NRG"],
    "CEG": ["NEE", "VST", "NRG", "DUK", "SO"],
    "VST": ["CEG", "NRG", "NEE", "DUK", "SO"],
    "NRG": ["VST", "CEG", "NEE", "DUK", "SO"],
    "PLD": ["AMZN", "UPS", "FDX", "WMT", "COST"],
    "AMT": ["T", "VZ", "TMUS", "MSFT", "GOOGL"],
    "EQIX": ["MSFT", "GOOGL", "AMZN", "META", "ORCL"],
    "LIN": ["NUE", "FCX", "CAT", "DE", "MOS"],
    "NUE": ["FCX", "LIN", "CAT", "DE", "HWM"],
    "FCX": ["NUE", "LIN", "CAT", "DE", "MOS"],
    "MOS": ["DE", "NUE", "FCX", "LIN", "CAT"],
    "RTX": ["LMT", "NOC", "GD", "LHX", "HWM"],
    "LMT": ["RTX", "NOC", "GD", "LHX", "HWM"],
    "NOC": ["RTX", "LMT", "GD", "LHX", "HWM"],
    "GD": ["RTX", "LMT", "NOC", "LHX", "HWM"],
    "LHX": ["RTX", "LMT", "NOC", "GD", "HWM"],
}


@dataclass(slots=True)
class SupplyChainSignal:
    score: float
    related_movers: List[str]
    summary: str


def compute_supply_chain_signal(ticker: str, price_frames: Dict[str, pd.DataFrame]) -> SupplyChainSignal:
    """Boost a stock when its linked suppliers/customers are already moving."""

    peers = SUPPLY_CHAIN_MAP.get(ticker, [])
    movers = []
    peer_returns = []
    for peer in peers:
        frame = price_frames.get(peer)
        if frame is None or frame.empty or len(frame) < 6:
            continue
        change = frame["close"].iloc[-1] / frame["close"].iloc[-6] - 1
        peer_returns.append(change)
        if change >= 0.04:
            movers.append(peer)
    avg_return = sum(peer_returns) / max(len(peer_returns), 1)
    score = clamp((avg_return + 0.02) / 0.08, 0.0, 1.0)
    summary = "No significant supply-chain confirmation."
    if movers:
        summary = f"Related movers: {', '.join(movers[:3])}"
    return SupplyChainSignal(score=score, related_movers=movers, summary=summary)

