import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import load_data, preprocess
from src.optimization import optimize_backtest
from src.robustness import run_sensitivity_analysis
from src.metrics import calculate_metrics, get_drawdown_series


def main():
    os.makedirs('results', exist_ok=True)
    start_time = time.time()

    raw_train, raw_test = load_data()
    data_train = preprocess(raw_train)
    data_test = preprocess(raw_test)

    # 1. Walk-Forward OOS
    equity_series, params_hist = optimize_backtest(data_train)

    if equity_series.empty:
        print("Error: No se generó curva de equidad.")
        return

    # 2. Generación de Tablas Obligatorias
    monthly = equity_series.resample('ME').last().pct_change().dropna()
    quarterly = equity_series.resample('QE').last().pct_change().dropna()
    annual = equity_series.resample('YE').last().pct_change().dropna()

    monthly.to_csv('results/monthly_returns.csv')
    quarterly.to_csv('results/quarterly_returns.csv')
    annual.to_csv('results/annual_returns.csv')

    # 3. Gráficos Obligatorios
    plt.figure(figsize=(14, 10))

    # Plot 1: Portfolio Value
    plt.subplot(2, 1, 1)
    plt.plot(equity_series, label='Portfolio Value (OOS)')
    plt.title("Evolución de Capital (Timeline Real Acumulado)")
    plt.legend()

    # Plot 2: Drawdown Curve
    plt.subplot(2, 1, 2)
    dd_series = get_drawdown_series(equity_series)
    plt.fill_between(dd_series.index, dd_series, 0, color='red', alpha=0.3)
    plt.title("Curva de Drawdown Submarina")
    plt.tight_layout()
    plt.savefig('results/portfolio_and_drawdown.png')

    # 4. Análisis de Estabilidad de Parámetros (Overfitting control)
    df_params = pd.DataFrame(params_hist)
    df_params.to_csv('results/params_history.csv')
    param_stability = df_params.std() / df_params.mean()  # Coeficiente de Variación
    param_stability.to_csv('results/parameter_stability.csv')

    # 5. Robustez
    run_sensitivity_analysis(data_test, params_hist[-1])

    final_metrics = calculate_metrics(equity_series)
    print(f"\n--- REPORTE FINAL ---")
    print(f"Tiempo Total: {(time.time() - start_time) / 60:.2f} min")
    for k, v in final_metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()