# src/pipeline/normalise.py

# (Helper) functions for (inverse) linear scaling used for normalisation

from typing import List
import numpy as np
import pandas as pd


def get_df_minimum(df: pd.DataFrame) -> float:
    """
    Returns minimum of entire dataframe, thus
    the minimum of the minimums of all columns

    :param df: dataframe
    :return: minimum of entire dataframe
    """
    return np.min(df.min())             


def get_df_maximum(df: pd.DataFrame) -> float:
    """
    Returns maximum of entire dataframe, thus
    the maximum of the maximums of all columns

    :param df: dataframe
    :return: maximum of entire dataframe
    """
    return np.max(df.max())


def calc_combined_min_max_params(dfs: list) -> tuple:
    """"
    Returns min and max of two dataframes combined
    
    :param dfs: list of dataframes
    :return: tuple of min and max
    """
    min = np.min([get_df_minimum(df) for df in dfs])
    max = np.max([get_df_maximum(df) for df in dfs])
    return min, max


def normalise_linear(df: pd.DataFrame, min: float, max: float) -> pd.DataFrame:
    """
    Performs linear scaling (minmax) on dataframe:

    x' = (x - x_min) / (x_max - x_min),

    where x is the original value, and x_min and x_max
    are the minimum and maximum values of the associated
    training data, respectively

    :param df: dataframe
    :param min: minimum value of training data
    :param max: maximum value of training data
    :return: normalised dataframe
    """
    return (df - min) / (max - min)


def normalise_linear_inv(df_norm: pd.DataFrame, min: float, max: float) -> pd.DataFrame:
    """
    Performs inverse linear scaling (minmax) on dataframe:

    x = x' * (x_max - x_min) + x_min,

    where x' is the normalised value, and x_min and x_max
    are the minimum and maximum values of the associated
    training data, respectively

    :param df_norm: normalised dataframe
    :param min: minimum value of training data
    :param max: maximum value of training data
    :return: inverse normalised dataframe
    """
    return df_norm * (max - min) + min


def print_pollutant_extremes(
        dfs: List[pd.DataFrame], pollutants: List[str], bool_print = True
    ) -> pd.DataFrame:
    """
    Takes a list of dataframes and a list of pollutants:
    - NO2 minimum
    - NO2 maximum
    - O3 minimum
    - O3 maximum
    - PM10 minimum
    - PM10 maximum
    - PM25 minimum
    - PM25 maximum

    makes a dataframe of them, prints it and returns it

    :param dfs: list of dataframes
    :param pollutants: list of pollutants
    :param bool_print: whether to print the dataframe
    :return: dataframe with minimum and maximum values
    """
    min_max_values = {}

    for i, pollutant in enumerate(pollutants):
        min_train, max_train = dfs[i*2], dfs[i*2+1]
        min_max_values[pollutant] = [min_train, max_train]

    df_minmax = pd.DataFrame(min_max_values, index=['min', 'max']).T
    if bool_print:
        print(df_minmax)

    return df_minmax