
import pandas as pd


# One-hot encoding

def one_hot_encode(df):
    """
    One-hot-encode all categorical features in DataFrame
    :param df: Pandas DataFrame
    :return: Pandas DataFrame with categorical features encoded
    """

    categorical_cols = df.select_dtypes(['object']).columns

    for col in categorical_cols:
        df[col] = df[col].apply(lambda x: str(x).strip())
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
        df.drop(col, axis=1, inplace=True)

    return df
