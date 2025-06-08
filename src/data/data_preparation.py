import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_data(df, target_column, test_size=0.2, random_state=42, stratify=True):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if isinstance(stratify, np.ndarray) or hasattr(stratify, "__iter__"):
        stratify_arg = stratify
    else:
        stratify_arg = y if stratify else None

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg)
    return x_train, x_test, y_train, y_test


def encode_categorical_columns(df, columns):
    for column in columns:
        if column in df.columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
    return df


def kfold_data_preparation(df: pd.DataFrame, label: str):
    y = df[label]
    x = df.drop(columns=label)
    return x, y
