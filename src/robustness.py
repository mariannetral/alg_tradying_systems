import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.backtest import run_single_backtest, calculate_metrics


def run_sensitivity_analysis(data_test, best_params):
    """Varía los parámetros óptimos +-20% y grafica el impacto."""
    results = []
    variations = [0.8, 1.0, 1.2]

    print("\nGenerando matriz de sensibilidad (+-20%)...")
    for v_tp in variations:
        for v_sl in variations:
            test_p = {
                'tp': best_params['tp'] * v_tp,
                'sl': best_params['sl'] * v_sl,
                'rsi_p': best_params.get('rsi_p', 14)
            }
            history = run_single_backtest(data_test, test_p)
            calmar = calculate_metrics(history)['Calmar']

            results.append({
                'TP_Var': f"{int(v_tp * 100)}%",
                'SL_Var': f"{int(v_sl * 100)}%",
                'Calmar': calmar
            })

    df_res = pd.DataFrame(results)
    plot_heatmap(df_res)
    return df_res


def plot_heatmap(df):
    """Heatmap del análisis de sensibilidad."""
    pivot = df.pivot(index='TP_Var', columns='SL_Var', values='Calmar')
    order = ['80%', '100%', '120%']
    pivot = pivot.reindex(index=order, columns=order)

    plt.figure(figsize=(8, 6))
    plt.imshow(pivot, cmap='RdYlGn')

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            plt.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontweight='bold', color='black')

    plt.title("Sensibilidad: Impacto en Calmar Ratio (+-20%)")
    plt.xlabel("Variación Stop Loss")
    plt.ylabel("Variación Take Profit")
    plt.colorbar(label='Calmar Ratio')
    plt.tight_layout()
    plt.savefig('results/sensitivity_heatmap.png')