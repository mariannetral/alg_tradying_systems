import pandas as pd

def load_data():
    data_train = pd.read_csv("data/btc_project_train.csv")
    data_test = pd.read_csv("data/btc_project_test.csv")
    return data_train, data_test

def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').set_index('Datetime')
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].ffill()
    return df