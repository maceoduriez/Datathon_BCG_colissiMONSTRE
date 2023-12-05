import pandas as pd
import numpy as np
from icecream import ic
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# import models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


def traiter_donnees(
    df,
):  # bilan de ce qu'on a fait au dessus, sans prendre en compte l'état de l'arc pour l'instant
    """
    prend le dataframe tout sale que tu prends en csv et renvoie le dataframe avec les dates en index pour bien tracer et sans utc et garde que les colonnes utiles
    """

    # tri par date
    df = df.sort_values(by=["Date et heure de comptage"], ascending=True)

    # on garde que les colonnes qui nous intéressent
    colonnes_a_garder = [
        "Libelle",
        "Date et heure de comptage",
        "Taux d'occupation",
        "Etat arc",
        "Débit horaire",
    ]
    df = df.loc[:, colonnes_a_garder]

    # renommer les colonnes pour que ce soit pratique
    df = df.rename(
        columns={
            "Date et heure de comptage": "timestamp",
            "Taux d'occupation": "taux_occupation",
            "Etat arc": "etat_arc",
            "Débit horaire": "debit_horaire",
        }
    )

    # jsplus pk j'ai fait ça
    df.reset_index(drop=True, inplace=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/Paris")
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df.set_index("timestamp", inplace=True)
    return df


def plot_daily_mean(df, column, title):
    daily_mean = df.loc[:, [column]].resample("D").mean()
    fig, ax = plt.subplots()
    ax.plot(daily_mean.index, daily_mean[column], label=column)
    ax.set_xlabel("Date")
    ax.set_title(title)
    ax.legend()
    plt.xticks(rotation=45)
    plt.show()


def impute_missing_values(df, max_iter=5, espilon=0.001):
    performance = {}
    step = +np.inf
    n_iter = 0

    missing_cols = df.columns[df.isna().any()].tolist()
    X = df.copy()

    for col in missing_cols:
        if X[col].dtype == "object":
            X[col].fillna(X[col].mode()[0], inplace=True)
        else:
            X[col].fillna(X[col].mean(), inplace=True)
    # for col with missing values, fill with mean for continuous and most frequent for categorical

    while n_iter < max_iter and step > espilon:
        n_iter += 1
        for col in missing_cols:
            # split into train and test
            X_train = X[df[col].notnull()]
            X_test = X[df[col].isnull()]

            # train and predict
            y_train = X_train[col]
            X_train = X_train.drop(columns=[col])

            y_test = X_test[col]
            X_test = X_test.drop(columns=[col])

            # fit model
            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            # predict
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            performance[col] = mse

            # fill missing values
            X.loc[df[col].isnull(), col] = y_pred

            # compute error
        step = sum(performance.values())
        print(step)

    return X, performance


def encode_categorical(df):
    X = df.copy()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        X[col] = OrdinalEncoder().fit_transform(X[col].values.reshape(-1, 1))
    return X
