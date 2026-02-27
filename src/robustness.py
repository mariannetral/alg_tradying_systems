import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.backtest import run_single_backtest
from src.metrics import calculate_metrics


def run_sensitivity_analysis(data_test, best_params):
    results = []
    variations = [0.8, 1.0, 1.2]

    for v_tp in variations:
        for v_sl in variations:
            # Ahora variamos TP, SL e incluimos RSI en el análisis
            test_p = {
                'tp': best_params['tp'] * v_tp,
                'sl': best_params['sl'] * v_sl,
                'rsi_p': int(best_params['rsi_p'] * v_tp),  # Variando dimensión temporal
                'ema_p': best_params['ema_p'],
                'bb_p': best_params['bb_p']
            }
            equity = run_single_backtest(data_test, test_p)
            calmar = calculate_metrics(equity)['Calmar']

            results.append({
                'TP_Var': f"{int(v_tp * 100)}%",
                'SL_Var': f"{int(v_sl * 100)}%",
                'Calmar': calmar
            })

    df_res = pd.DataFrame(results)
    plot_heatmap(df_res)
    return df_res


def plot_heatmap(df):
    pivot = df.pivot(index='TP_Var', columns='SL_Var', values='Calmar')
    order = ['80%', '100%', '120%']
    pivot = pivot.reindex(index=order, columns=order)
    plt.figure(figsize=(8, 6))
    plt.imshow(pivot, cmap='RdYlGn')
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            plt.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontweight='bold')
    plt.title("Sensibilidad: Calmar Ratio (+-20%)")
    plt.xlabel("Variación Stop Loss")
    plt.ylabel("Variación Take Profit")
    plt.colorbar()
    plt.savefig('results/sensitivity_heatmap.png')