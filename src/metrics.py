import numpy as np
import pandas as pd


def calculate_metrics(history):
    """
    Calcula métricas: Sharpe, Sortino, Calmar, Max Drawdown y Win Rate.
    """
    hist = np.array(history)
    returns = pd.Series(hist).pct_change().dropna()
    ann_factor = 288 * 365  # Velas de 5 min anualizadas

    total_ret = (hist[-1] / hist[0]) - 1
    # Max Drawdown
    peak = np.maximum.accumulate(hist)
    dd = (hist - peak) / peak
    max_dd = abs(dd.min())

    # Sharpe y Sortino
    vol = returns.std() * np.sqrt(ann_factor)
    sharpe = (returns.mean() * ann_factor) / vol if vol != 0 else 0
    downside_vol = returns[returns < 0].std() * np.sqrt(ann_factor)
    sortino = (returns.mean() * ann_factor) / downside_vol if downside_vol != 0 else 0

    # Calmar Ratio (Métrica Principal )
    calmar = total_ret / max_dd if max_dd > 0 else 0
    win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0

    return {"Calmar": calmar, "Sharpe": sharpe, "Sortino": sortino, "MaxDD": max_dd, "WinRate": win_rate,
            "Return": total_ret}