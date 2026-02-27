import optuna
import pandas as pd
from tqdm import tqdm
from src.backtest import run_single_backtest
from src.metrics import calculate_metrics


def objective(trial, data):
    params = {
        'tp': trial.suggest_float("tp", 0.02, 0.12),
        'sl': trial.suggest_float("sl", 0.01, 0.05),
        'rsi_p': trial.suggest_int("rsi_p", 10, 20),
        'ema_p': trial.suggest_int("ema_p", 20, 100),
        'bb_p': trial.suggest_int("bb_p", 15, 30)
    }

    equity = run_single_backtest(data, params, initial_cash=1_000_000)
    return calculate_metrics(equity)['Calmar']


def optimize_backtest(data):
    train_step, test_step, step = 8640, 2016, 2016
    oos_results_list = []
    params_history = []


    current_cash = 1_000_000

    windows = range(0, len(data) - train_step - test_step, step)

    for start in tqdm(windows, desc="Walk-Forward Acumulativo"):
        train_slice = data.iloc[start: start + train_step]

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, train_slice), n_trials=100, n_jobs=-1)

        best_p = study.best_params
        params_history.append(best_p)

        test_slice = data.iloc[start + train_step: start + train_step + test_step]

        oos_equity_series = run_single_backtest(test_slice, best_p, initial_cash=current_cash)

        if not oos_equity_series.empty:
            current_cash = oos_equity_series.iloc[-1]
            oos_results_list.append(oos_equity_series)

    final_equity_curve = pd.concat(oos_results_list) if oos_results_list else pd.Series(dtype=float)
    return final_equity_curve, params_history