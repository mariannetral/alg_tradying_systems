import ta

def apply_indicators(df, rsi_p=14, sma_p=20, stoch_p=14):
    """
    Implementa 3 indicadores técnicos para validación de señal (2 de 3).
    """
    df = df.copy()
    # 1. RSI (Momentum)
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=rsi_p).rsi()
    # 2. SMA (Trend)
    df['sma'] = ta.trend.sma_indicator(df['Close'], window=sma_p)
    # 3. Stochastic Oscillator (Momentum/Reversion)
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=stoch_p)
    df['stoch'] = stoch.stoch()
    return df.dropna()


def get_signal(row):
    """
    Confirmación de señal: Al menos 2 de 3 indicadores deben coincidir.
    """
    long_hints = [row['rsi'] < 30, row['Close'] > row['sma'], row['stoch'] < 20]
    short_hints = [row['rsi'] > 70, row['Close'] < row['sma'], row['stoch'] > 80]

    if sum(long_hints) >= 2: return 'LONG'
    if sum(short_hints) >= 2: return 'SHORT'
    return None