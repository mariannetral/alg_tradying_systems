import pandas as pd
import numpy as np
import ta


def run_single_backtest(data: pd.DataFrame, params: dict) -> list[float]:
    """Ejecuta el backtest con la lógica Long/Short, sin apalancamiento."""
    data = data.copy()
    cash = 1_000_000
    COM = 0.00125  # 0.125% de comisión
    active_pos = None
    strategy_value = []

    # 1. Aplicar Indicadores Técnicos
    data['rsi'] = ta.momentum.RSIIndicator(data.Close, window=params.get('rsi_p', 14)).rsi()
    data['sma'] = ta.trend.sma_indicator(data.Close, window=params.get('sma_p', 20))
    stoch = ta.momentum.StochasticOscillator(data.High, data.Low, data.Close, window=params.get('stoch_p', 14))
    data['stoch'] = stoch.stoch()
    data = data.dropna()

    for i, row in data.iterrows():
        # 2. Confirmación de Señales (2 de 3)
        long_hints = [row.rsi < 35, row.Close > row.sma, row.stoch < 25]
        short_hints = [row.rsi > 65, row.Close < row.sma, row.stoch > 75]

        # 3. Gestión de Riesgo (Cierre de Posiciones)
        if active_pos:
            pnl_pct = (row.Close - active_pos['price']) / active_pos['price']
            if active_pos['type'] == 'SHORT': pnl_pct *= -1

            if pnl_pct >= params['tp'] or pnl_pct <= -params['sl']:
                if active_pos['type'] == 'LONG':
                    cash += (row.Close * active_pos['qty']) * (1 - COM)
                else:
                    # Lógica exacta para SHORT solicitada
                    pnl_cash = (active_pos['price'] - row.Close) * active_pos['qty'] * (1 - COM)
                    cash += pnl_cash
                active_pos = None

        # 4. Apertura de Posiciones (Usando 10% del capital - Sin apalancamiento)
        elif sum(long_hints) >= 2:
            qty = (cash * 0.1) / row.Close
            cost = row.Close * qty * (1 + COM)
            if cash >= cost:
                cash -= cost
                active_pos = {'type': 'LONG', 'price': row.Close, 'qty': qty}
        elif sum(short_hints) >= 2:
            qty = (cash * 0.1) / row.Close
            cash -= (row.Close * qty * COM)  # Solo descontar comisión inicial
            active_pos = {'type': 'SHORT', 'price': row.Close, 'qty': qty}

        # 5. Valor del Portafolio Mark-to-Market
        val = cash
        if active_pos:
            if active_pos['type'] == 'LONG':
                val += (row.Close * active_pos['qty'])
            else:
                val += (active_pos['price'] - row.Close) * active_pos['qty']
        strategy_value.append(val)

    return strategy_value


def calculate_metrics(history: list[float]) -> dict:
    """Calcula las métricas de desempeño."""
    hist = np.array(history)
    if len(hist) < 2: return {"Calmar": 0, "Sharpe": 0, "Sortino": 0, "MaxDD": 0, "WinRate": 0}

    returns = pd.Series(hist).pct_change().dropna()
    total_ret = (hist[-1] / hist[0]) - 1

    peak = np.maximum.accumulate(hist)
    dd = (hist - peak) / peak
    max_dd = abs(dd.min())

    ann_factor = 288 * 365  # Anualización para velas de 5m
    vol = returns.std() * np.sqrt(ann_factor)
    downside_vol = returns[returns < 0].std() * np.sqrt(ann_factor)

    sharpe = (returns.mean() * ann_factor) / vol if vol != 0 else 0
    sortino = (returns.mean() * ann_factor) / downside_vol if downside_vol != 0 else 0
    calmar = total_ret / max_dd if max_dd > 0 else 0
    win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0

    return {"Calmar": calmar, "Sharpe": sharpe, "Sortino": sortino, "MaxDD": max_dd, "WinRate": win_rate,
            "Return": total_ret}


def objective(data: pd.DataFrame, trial) -> float:
    """Función objetivo para Optuna: Maximizar Calmar Ratio."""
    params = {
        'tp': trial.suggest_float("tp", 0.02, 0.12),
        'sl': trial.suggest_float("sl", 0.01, 0.05),
        'rsi_p': trial.suggest_int("rsi_p", 10, 20)
    }
    history = run_single_backtest(data, params)
    metrics = calculate_metrics(history)
    return metrics['Calmar']