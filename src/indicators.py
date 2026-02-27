import pandas as pd
import ta


def generate_signals(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = data.copy()

    # 1. Momentum: RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=params.get('rsi_p', 14)).rsi()
    # 2. Tendencia: EMA
    df['ema'] = ta.trend.ema_indicator(df['Close'], window=params.get('ema_p', 50))
    # 3. Volatilidad: Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Close'], window=params.get('bb_p', 20))
    df['bb_low'] = bb.bollinger_lband()
    df['bb_high'] = bb.bollinger_hband()

    df = df.dropna()

    # Regla 2 de 3
    long_hints = (df['rsi'] < 35).astype(int) + (df['Close'] > df['ema']).astype(int) + (
                df['Close'] < df['bb_low']).astype(int)
    short_hints = (df['rsi'] > 65).astype(int) + (df['Close'] < df['ema']).astype(int) + (
                df['Close'] > df['bb_high']).astype(int)

    df['raw_signal'] = 0
    df.loc[long_hints >= 2, 'raw_signal'] = 1
    df.loc[short_hints >= 2, 'raw_signal'] = -1

    # SOLUCIÓN CRÍTICA: Desplazar la señal para evitar look-ahead bias
    df['signal'] = df['raw_signal'].shift(1)

    return df.dropna()