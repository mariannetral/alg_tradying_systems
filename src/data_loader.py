import pandas as pd


def load_data():
    """Carga los datasets de entrenamiento y prueba."""
    data_train = pd.read_csv("data/btc_project_train.csv")
    data_test = pd.read_csv("data/btc_project_test.csv")
    return data_train, data_test


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesa el dataset, asigna índices y rellena valores nulos."""
    df = data.copy()
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').set_index('Datetime')

    # Rellenar valores nulos de precio hacia adelante para no romper los indicadores
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].ffill()
    return df