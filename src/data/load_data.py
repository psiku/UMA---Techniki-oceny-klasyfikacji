from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np


def get_project_root():
    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )


def load_breath_cancer_data():
    project_root = get_project_root()
    path = os.path.join(project_root, "datasets", "breast_cancer", "wdbc.data")

    df = pd.read_csv(path, delimiter=",")

    return df


def load_students_data():
    project_root = get_project_root()

    mat_df_path = os.path.join(project_root, "datasets", "students", "student-mat.csv")
    por_df_path = os.path.join(project_root, "datasets", "students", "student-por.csv")

    mat_df = pd.read_csv(mat_df_path, delimiter=";")
    por_df = pd.read_csv(por_df_path, delimiter=";")

    students_df = pd.concat([mat_df, por_df], ignore_index=True)

    return students_df


def prepare_data(df, target_column, test_size=0.2, random_state=42, stratify=True):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if isinstance(stratify, np.ndarray) or hasattr(stratify, "__iter__"):
        stratify_arg = stratify
    else:
        stratify_arg = y if stratify else None

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_arg)
    return x_train, x_test, y_train, y_test
