import os
from pathlib import Path
from config import *



def init_paths(city, years, loss_func, nn_type):
    """
    Initialise paths for the given city, years, loss function and neural network type.
    
    Args:
        city (str): The city for which to initialise paths ('Utrecht', 'Amsterdam', or 'Multi').
        years (list): List of years to include in the path.
        loss_func (str): The loss function used in the model.
        nn_type (str): The type of neural network used in the model.
        
    Returns:
        dict: A dictionary containing initialised paths.
    """
    
    if city == 'Utrecht':
        idx_dict = UTRECHT_IDX
        station_names = ['tuindorp', 'breukelen']
        main_station = 'breukelen'
        results_path = os.path.join(RESULTS_PATH, 'Utrecht')
        model_path = os.path.join(results_path, 'models')
        data_path = DATA_UTRECHT_PATH
    elif city == 'Amsterdam':
        idx_dict = AMSTERDAM_IDX
        station_names = ['oudemeer', 'haarlem']
        main_station = 'haarlem'
        results_path = os.path.join(RESULTS_PATH, 'Amsterdam')
        model_path = os.path.join(results_path, 'models')
        data_path = DATA_AMSTERDAM_PATH
    elif city == 'Multi':
        idx_dict = MULTI_STATION_IDX
        station_names = ['tuindorp', 'breukelen', 'zegveld', 'oudemeer', 'kantershof']
        main_station = 'breukelen'
        results_path = os.path.join(RESULTS_PATH, 'Multi')
        model_path = os.path.join(results_path, 'models')
        data_path = DATA_MULTI_PATH
    else:
        raise ValueError("CITY must be 'Utrecht', 'Amsterdam', or 'Multi'.")
    
    if years == [2017, 2018, 2020, 2021, 2022, 2023]:
        years_str = "allyears"
        minmax_path = os.path.join(data_path, "all_years", "pollutants_minmax_allyears.csv")
        dataset_path = os.path.join(data_path, "all_years")
    elif years == [2017]:
        years_str = "2017"
        minmax_path = os.path.join(data_path, "only_2017", "pollutants_minmax_2017.csv")
        dataset_path = os.path.join(data_path, "only_2017")
    elif years == [2017, 2018, 2020]:
        years_str = "first_3_years"
        minmax_path = os.path.join(data_path, "first_3_years", "pollutants_minmax_2017_2018_2020.csv")
        dataset_path = os.path.join(data_path, "first_3_years")
    else:
        raise ValueError("Invalid years selected")
    
    y_phy_filename = f"y_phy_batchsize16_{loss_func}_{years_str}_{city}"
    model_filename = f'best_{nn_type}_no2_{loss_func}_{years_str}_{city}.pth'
    results_metrics_filename = f'results_{nn_type}_no2_{loss_func}_{years_str}_{city}.csv'
    bestparams_filename = f'best_params_{nn_type}_no2_{loss_func}_{years_str}_{city}.txt'
    plot_filename = f'plot_{nn_type}_no2_{loss_func}_{years_str}_{city}.png'

    return years_str, idx_dict, station_names, main_station, results_path, model_path, dataset_path, minmax_path, y_phy_filename, model_filename, results_metrics_filename, bestparams_filename, plot_filename


# function to check the indexes of the stations in the dataset vs in config
# input: column names of dataset, idx dict
# the function should be able to check for NO2_XX, wind speed, and wind dir
def check_station_indexes(column_names, idx_dict):
    """
    Check the indexes of the stations in the dataset vs in config.
    
    Args:
        column_names (list): Column names of the dataset.
        idx_dict (dict): Dictionary containing station indexes.
        
    Returns:
        bool: True if all indexes match, False otherwise.
    """
    
    for key, value in idx_dict.items():
        if key == 'WIND_DIR_IDX':
            column_name = 'DD'
        elif key == 'WIND_SPEED_IDX':
            column_name = 'FH' 
        elif key.endswith('_IDX'):
            column_name = key[:-4]  # Remove _IDX suffix
        else:
            parts = key.split('_')
            if len(parts) == 2:
                column_name = key  # Keep the original format
            else:
                print(f"Warning: Unexpected key format: {key}")
                column_name = key
        if column_name not in column_names:
            # raise an error
            ValueError(f"Error: {key} not found in dataset column names.")

            return False
        if value != column_names.index(column_name):
            
            # raise an error
            ValueError(f"Error: {key} index mismatch. Expected {value}, found {column_names.index(column_name)}.")
            
            return False
        # print indexes
        print(f"{key} index matches in index: {value}")
    print("All station indexes match.")
    return True
