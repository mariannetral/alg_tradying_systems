import optuna
from tqdm import tqdm
from src.backtest import objective, run_single_backtest


def optimize_backtest(data):
    """Ejecuta el Walk-Forward Analysis respetando las ventanas temporales."""
    # Ventanas: 1 mes train (~8640), 1 semana test (~2016), paso semanal (~2016)
    train_step, test_step, step = 8640, 2016, 2016
    oos_results = []
    params_history = []

    windows = range(0, len(data) - train_step - test_step, step)

    for start in tqdm(windows, desc="Walk-Forward Analysis"):
        train_slice = data.iloc[start: start + train_step]

        # Limitar a 100 trials por ventana para balancear eficiencia
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(train_slice, trial), n_trials=100, n_jobs=-1)

        best_p = study.best_params
        params_history.append(best_p)

        test_slice = data.iloc[start + train_step: start + train_step + test_step]
        oos_results.extend(run_single_backtest(test_slice, best_p))

    return oos_results, params_history