import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import load_data, preprocess
from src.optimization import optimize_backtest
from src.robustness import run_sensitivity_analysis
from src.backtest import calculate_metrics


def main():
    os.makedirs('results', exist_ok=True)
    start_time = time.time()
    print("Algorithmic Trading...")

    # 1. Carga y Preproceso de Datos
    raw_train, raw_test = load_data()
    data_train = preprocess(raw_train)
    data_test = preprocess(raw_test)

    # 2. Walk-Forward Analysis
    equity_curve, params_hist = optimize_backtest(data_train)

    # 3. Métricas y Tablas de Retorno para el Reporte
    equity_series = pd.Series(equity_curve)
    equity_series.index = data_train.index[-len(equity_curve):]  # Alinear fechas

    # Guardar tablas de retornos
    equity_series.resample('ME').last().pct_change().dropna().to_csv('results/monthly_returns.csv')
    equity_series.resample('QE').last().pct_change().dropna().to_csv('results/quarterly_returns.csv')

    final_metrics = calculate_metrics(equity_curve)

    # 4. Análisis de Sensibilidad (Test OOS con últimos parámetros)
    run_sensitivity_analysis(data_test, params_hist[-1])

    # 5. Guardar Gráficas y Tiempos
    total_min = (time.time() - start_time) / 60
    pd.DataFrame(params_hist).to_csv('results/params_history.csv')

    plt.figure(figsize=(12, 6))
    plt.plot(equity_series, label='Valor del Portafolio')
    plt.title(f"Estrategia Walk-Forward (Calmar: {final_metrics['Calmar']:.2f})")
    plt.legend()
    plt.savefig('results/equity_curve.png')

    print("\nREPORTE FINAL")
    print(f"Tiempo Total de Optimización: {total_min:.2f} minutos")
    for k, v in final_metrics.items():
        print(f"{k}: {v:.4f}")
    print("\nGráficos y tablas están en la carpeta 'results/'.")


if __name__ == "__main__":
    main()