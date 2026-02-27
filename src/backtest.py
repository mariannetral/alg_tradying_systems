import pandas as pd
from src.indicators import generate_signals


def run_single_backtest(data: pd.DataFrame, params: dict, initial_cash: float = 1_000_000) -> pd.Series:
    df = generate_signals(data, params)
    cash = initial_cash
    COM = 0.00125  # Comisión del 0.125%
    active_pos = None
    strategy_value = []

    for date, row in df.iterrows():
        signal = row['signal']
        current_price = row['Close']

        # --- 1. GESTIÓN DE RIESGO (CIERRE DE POSICIONES) ---
        if active_pos:
            pnl_pct = (current_price - active_pos['price']) / active_pos['price']
            if active_pos['type'] == 'SHORT':
                pnl_pct *= -1

            # Cierre por Take Profit o Stop Loss
            if pnl_pct >= params['tp'] or pnl_pct <= -params['sl']:
                if active_pos['type'] == 'LONG':
                    cash += (current_price * active_pos['qty']) * (1 - COM)
                else:
                    # FÓRMULA ESTRICTA DEL PROFESOR PARA CIERRE DE SHORT
                    # pnl = (bought_at - current_price) * n_shares * (1 - COM)
                    pnl = (active_pos['price'] - current_price) * active_pos['qty'] * (1 - COM)
                    cash += pnl

                active_pos = None

        # --- 2. EJECUCIÓN (APERTURA DE POSICIONES) ---
        elif signal == 1:  # LONG
            qty = (cash * 0.1) / current_price
            cost = current_price * qty * (1 + COM)
            if cash >= cost:
                cash -= cost
                active_pos = {'type': 'LONG', 'price': current_price, 'qty': qty}

        elif signal == -1:  # SHORT
            qty = (cash * 0.1) / current_price
            # FÓRMULA ESTRICTA DEL PROFESOR PARA APERTURA DE SHORT
            # cost = current_price * n_shares * COM
            cost = current_price * qty * COM
            cash -= cost  # Solo se deduce la comisión de apertura
            active_pos = {'type': 'SHORT', 'price': current_price, 'qty': qty}

        # --- 3. VALORACIÓN DEL PORTAFOLIO (Mark-to-Market) ---
        val = cash
        if active_pos:
            if active_pos['type'] == 'LONG':
                val += (current_price * active_pos['qty'])
            else:
                # estimating the current strategy value (SHORT)
                val += (active_pos['price'] - current_price) * active_pos['qty']

        strategy_value.append(val)

    # Retorna la curva de capital con su índice de fechas original para el Walk-Forward
    return pd.Series(strategy_value, index=df.index)