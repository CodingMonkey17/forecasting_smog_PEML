# src/modelling/extract.py

# Functions to extract the data from the data/ folder

from typing import List
import pandas as pd
import os


def import_csv(filename: str) -> pd.DataFrame:
    """
    Imports a file from the data/data_combined folder
    
    :param file_name: name of the file to import
    """
    return pd.read_csv(f'../data/data_combined/{filename}',
                       index_col = 'DateTime',
                       sep = ';',
                       decimal = '.')


def get_dataframes(data_type: str, data_category: str, years: List[int]) -> List[pd.DataFrame]:
    """
    Convenience function that based on data_type (= 'train', 'val', 'test')
    and data_category (= 'input' or 'output') returns the associated list of
    dataframes from the data/data_combined folder, only for the specified years.

    :param data_type: 'train', 'val' (= validation), 'test'
    :param data_category: 'u' (= input), 'y' (= output)
    :param years: List of years to import data for
    """
    dataframes = []
    for year in years:
        filename = f'{data_type}_{year}_combined_{data_category}.csv'
        filepath = f'../data/data_combined/{filename}'
        if os.path.exists(filepath):
            dataframes.append(import_csv(filename))
            print(f"Imported {filename}")
        else:
            print(f"Warning: {filename} does not exist.")
    if not dataframes:
        raise ValueError(f"No dataframes found for data_type '{data_type}' and data_category '{data_category}'")
    return dataframes