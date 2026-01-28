import pandas as pd

def load_data(url="https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"):
    """
    Loads the Boston Housing dataset from a URL.
    """
    df = pd.read_csv(url)
    df = df.rename(columns={"medv": "price"})
    return df
