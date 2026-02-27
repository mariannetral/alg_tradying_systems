import numpy as np
import pandas as pd


def calculate_metrics(equity_series: pd.Series) -> dict:
    if len(equity_series) < 2: return {"Calmar": 0, "MaxDD": 0, "Return": 0}

    days = (equity_series.index[-1] - equity_series.index[0]).total_seconds() / 86400
    total_ret = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1


    ann_ret = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0

    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_dd = abs(drawdown.min())

    calmar = ann_ret / max_dd if max_dd > 0 else 0

    return {"Calmar": calmar, "MaxDD": max_dd, "TotalReturn": total_ret, "AnnReturn": ann_ret}


def get_drawdown_series(equity_series: pd.Series) -> pd.Series:
    peak = equity_series.cummax()
    return (equity_series - peak) / peak