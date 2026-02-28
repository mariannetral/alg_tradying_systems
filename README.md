# BTCUSDT Systematic Trading Strategy

Este repositorio contiene la implementación de un sistema de trading algorítmico sistemático para el par BTCUSDT en temporalidad de 5 minutos. El proyecto está diseñado bajo estrictos estándares cuantitativos, enfocándose en la preservación de capital y la robustez estadística mediante un análisis *Walk-Forward* (WFA).

## Metodología del Sistema

El núcleo del motor de trading se basa en los siguientes principios metodológicos:

* **Confirmación Dimensional (2 de 3):** Las señales de entrada requieren la confirmación de al menos dos de tres indicadores que analizan diferentes dimensiones del mercado:
    * *Momentum:* RSI (Relative Strength Index)
    * *Tendencia:* EMA (Exponential Moving Average)
    * *Volatilidad:* Bandas de Bollinger (BB)
* **Ausencia de Look-Ahead Bias:** Las señales de ejecución aplican un desplazamiento temporal (`shift(1)`) para garantizar que las órdenes operen únicamente con información histórica consolidada al cierre de la vela anterior.
* **Asimetría de Costos Realista:** Incorpora un modelo de comisiones estricto ($0.125\%$) deducido directamente del flujo de caja. Implementa fórmulas diferenciadas y exactas para la apertura y cierre de posiciones LONG y SHORT sin apalancamiento.
* **Walk-Forward Acumulativo:** Optimización continua utilizando ventanas de entrenamiento de 1 mes y validación *Out-of-Sample* (OOS) de 1 semana, transfiriendo el capital de forma acumulativa para reflejar el *timeline* real del portafolio.
* **Optimización por Calmar Ratio:** El sistema busca maximizar la relación entre el Retorno Anualizado y el *Maximum Drawdown*, penalizando estrategias con caídas severas de capital.

## Estructura del Repositorio

El proyecto utiliza una arquitectura modular para separar claramente la generación de señales, el motor de ejecución y la evaluación de métricas:

```text
proyecto_trading/
├── data/
│   ├── btc_project_train.csv      # Dataset histórico de entrenamiento (5-min)
│   └── btc_project_test.csv       # Dataset para análisis Out-of-Sample final
├── results/                       # Directorio autogenerado para outputs (gráficas y CSVs)
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Ingesta, indexación temporal y manejo de NaNs
│   ├── indicators.py              # Generación de señales (Trend, Momentum, Volatility)
│   ├── backtest.py                # Motor de ejecución pura y mark-to-market
│   ├── metrics.py                 # Cálculo de métricas (Sharpe, Sortino, Calmar Anualizado, MaxDD)
│   ├── optimization.py            # Orquestador del Walk-Forward Analysis vía Optuna
│   └── robustness.py              # Análisis de Sensibilidad (±20%) y Heatmaps
├── main.py                        # Punto de entrada y generador de reportes visuales
└── requirements.txt               # Dependencias del proyecto