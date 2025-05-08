# %% [markdown]
# # **Baseline Simple MLP with just MSE**
# 

# %% [markdown]
# ## **Running the models using the 'modelling' package**
# 
# A notebook through which different modelling configurations can be ran, using the ``modelling`` package. It follows the steps of:
# - preparing packages;
# - setting "global" variables;
# - getting the data;
# - defining hyperparameters;
# - running a Optuna hyperparameters optimisation and/or training a model; and
# - evaluation.
# In the modelling package, variations can be made to the models and training functions to experiment. Don't forget to restart the notebook after making changes there.
# 
# ## **IMPORTANT NOTE**: 
# - do preprocessing from ``preprocess.ipynb`` to obtain data in ``data/data_combined``, before starting this notebook
# - make sure the notebook is under ``src`` directory before running!
# - change the global variables defined below for the desired years of data, loss function and NN type
# 
# 

# %%
print("Starting script...")


from modelling.MLP import BasicMLP
from modelling import *


import optuna
import threading
import os
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

# %% [markdown]
# Use GPU when available

# %%
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Device: ", device)

# %% [markdown]
# ### **Set "global" variables**

# %%
Path.cwd()

# %%
import importlib
import config
importlib.reload(config)

# %%
from config import *

# %%
HABROK = bool(0)                  # set to True if using HABROK; it will print
                                # all stdout to a .txt file to log progress


    # %% [markdown]
    # 
    # ## MODIFY THESE GLOBAL VARIABLES FOR YOUR MODEL SCENARIO
    # all other variables are defined in config.py
    # 
    # LOSS_FUNC: choose from 
    # - MSE
    # - LinearShift_MSE
    # - PDE_nmer_const
    # - PDE_nmer_piece
    # - PINN
    # 
    # CITY: choose from
    # - Utrecht
    # - Amsterdam (for testing transferability)
    # - Multi (extended area around utrecht)

    # %%
    # Change this according to the data you want to use
valid_loss_funcs = ["MSE", "LinearShift_MSE", "PDE_nmer_const", "PDE_nmer_piece", "PINN"]
valid_cities = ["Utrecht", "Multi"]

# Mapping for readable year configs
year_configs = {
    "allyears": [2017, 2018, 2020, 2021, 2022, 2023],
    "first3years": [2017, 2018, 2020],
    "2017": [2017]
}

# Iterate over configurations
for city in valid_cities:
    for loss_func in valid_loss_funcs:
        for year_key, years in year_configs.items():
            if year_key in ["first3years", "2017"] and loss_func not in ["MSE", "PINN"]:
                continue  # Skip unsupported combinations

            print(f"\nRunning for {city}, {loss_func}, {year_key}...")

            # Set global variables dynamically
            CITY = city
            LOSS_FUNC = loss_func
            NN_TYPE = "MLP" 
            if year_key == 'allyears':
                YEARS = [2017, 2018, 2020, 2021, 2022, 2023]
                TRAIN_YEARS = [2017, 2018, 2020, 2021, 2022]
                VAL_YEARS = [2021, 2022, 2023]
                TEST_YEARS = [2021, 2022, 2023]
            else:
                YEARS = years
                TRAIN_YEARS = years
                VAL_YEARS = years
                TEST_YEARS = years
               
            # ## Automated Generation of paths and filenames according to data years, loss func, NN type
            # - will be used throughout the whole notebook
            # - check ``config.py`` for global variables defined outside the notebook

    
            years, idx_dict , station_names, main_station, RESULTS_PATH, MODEL_PATH, DATASET_PATH, MINMAX_PATH, Y_PHY_FILENAME,  MODEL_PATH_NAME,RESULTS_METRICS_FILENAME, BESTPARAMS_FILENAME, PLOT_FILENAME  = init_paths(CITY, YEARS, LOSS_FUNC, NN_TYPE)
            print("years: ", years)
            print("idx_dict: ", idx_dict)
            print("station_names: ", station_names)
            print("main_station: ", main_station)
            print("RESULTS_PATH: ", RESULTS_PATH)
            print("MODEL_PATH: ", MODEL_PATH)
            print("MINMAX_PATH: ", MINMAX_PATH)
            print("DATASET_PATH: ", DATASET_PATH)
            print("Y_PHY_FILENAME: ", Y_PHY_FILENAME)
            print("MODEL_PATH_NAME: ", MODEL_PATH_NAME)
            print("RESULTS_METRICS_FILENAME: ", RESULTS_METRICS_FILENAME)
            print("BESTPARAMS_FILENAME: ", BESTPARAMS_FILENAME)
            print("PLOT_FILENAME: ", PLOT_FILENAME)

   
            # ### **Load in data and create PyTorch *Datasets***

    
            # Load in data and create PyTorch Datasets. To tune
            # which exact .csv files get extracted, change the
            # lists in the get_dataframes() definition

            train_input_frames = get_dataframes('train', 'u', YEARS, DATASET_PATH)
            train_output_frames = get_dataframes('train', 'y', YEARS, DATASET_PATH)

            val_input_frames = get_dataframes('val', 'u', YEARS, DATASET_PATH)
            val_output_frames = get_dataframes('val', 'y', YEARS, DATASET_PATH)

            test_input_frames = get_dataframes('test', 'u', YEARS, DATASET_PATH)
            test_output_frames = get_dataframes('test', 'y', YEARS, DATASET_PATH)

            print("Successfully loaded data")

    
            train_dataset = TimeSeriesDataset(
                train_input_frames,  # list of input training dataframes
                train_output_frames, # list of output training dataframes
                len(TRAIN_YEARS),                   # number of dataframes put in for both
                                    # (basically len(train_input_frames) and
                                    # len(train_output_frames) must be equal)
                N_HOURS_U,           # number of hours of input data
                N_HOURS_Y,           # number of hours of output data
                N_HOURS_STEP,        # number of hours between each input/output pair
            )
            val_dataset = TimeSeriesDataset(
                val_input_frames,    # etc.
                val_output_frames,
                len(VAL_YEARS),
                N_HOURS_U,
                N_HOURS_Y,
                N_HOURS_STEP,
            )
            test_dataset = TimeSeriesDataset(
                test_input_frames,
                test_output_frames,
                len(TEST_YEARS),
                N_HOURS_U,
                N_HOURS_Y,
                N_HOURS_STEP,
            )

            del train_input_frames, train_output_frames
            del val_input_frames, val_output_frames
            del test_input_frames, test_output_frames

   
            # ## Confirmation that the dataset has column indexes the same as those in ``config.py``
            # Indexes are used mainly for the physics calculations, in order to accurately extract the information needed

    
            column_names = list(train_dataset.u[0])  # Convert Index to list
            check_station_indexes(column_names, idx_dict)

    
            import random
            def set_seed(seed):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    
            import json
            file_path = f"{RESULTS_PATH}/best_params/{BESTPARAMS_FILENAME}"

   
            # ## Read params from file

    

            with open(file_path, "r") as f:
                best_params = json.load(f)  # Automatically converts it to a dictionary

            print(f"Loading best parms from {file_path}")
            print("Loaded Best Parameters:", best_params)

   
            # ## Training and Saving Model
            # Model saved in ``src/results/models/best_MLP_no2_MSE_allyears.pth``

    
            set_seed(42)
            # Train the model with the best hyperparameters
            best_model_baseline = BasicMLP(
                N_INPUT_UNITS=train_dataset.__n_features_in__(),
                N_HIDDEN_LAYERS=best_params["n_hidden_layers"],
                N_HIDDEN_UNITS=best_params["n_hidden_units"],
                N_OUTPUT_UNITS=train_dataset.__n_features_out__(),
                loss_function="MSE",
            )

            # Create train & validation loaders with the best batch size
            train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False)


   
            # ## Plot Train-Val
            # Plot saved in ``src/results/trainval_plots/trainval_plot_MLP_no2_MSE_allyears.png``
            # 

   
            # ## Test and Save Results
            # Results saved in ``src/results/metrics/results_MLP_no2_MSE_allyears.csv``

    
            best_model_baseline.load_state_dict(torch.load(f"{MODEL_PATH}/{MODEL_PATH_NAME}", map_location = device))
            print(f"Loading best model of {NN_TYPE} {LOSS_FUNC} {years} from {MODEL_PATH}/{MODEL_PATH_NAME}")
            best_model_baseline.eval()

            # Create the DataLoader for the test dataset
            test_loader = DataLoader(test_dataset, batch_size=best_params["batch_size"], shuffle=False)

            # Evaluate the model on the test dataset
            df_minmax = pd.read_csv(MINMAX_PATH, sep=';')
            min_value = df_minmax["min"].values
            max_value = df_minmax["max"].values
            mse, rmse, smape, inference_time_mean, inference_time_std = best_model_baseline.test_model(test_loader, min_value=min_value, max_value=max_value, device=device)



    
            import csv

            # Define the CSV file path
            results_csv_path = f"{RESULTS_PATH}/metrics/{RESULTS_METRICS_FILENAME}"

            # Read original header and row
            with open(results_csv_path, mode="r") as f:
                reader = csv.reader(f)
                original_header = next(reader)
                original_row = next(reader)

            # Convert to dict for easy manipulation
            data = dict(zip(original_header, original_row))

            # Update inference time
            data["Inference Time"] = str(inference_time_mean)

            # Insert Inference Time Std right after Inference Time
            new_header = []
            new_row = []

            for col in original_header:
                new_header.append(col)
                new_row.append(data[col])
                if col == "Inference Time":
                    new_header.append("Inference Time Std")
                    new_row.append(str(inference_time_std))

            # Write updated CSV
            with open(results_csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(new_header)
                writer.writerow(new_row)


            print(f"Results saved as {RESULTS_METRICS_FILENAME} in Results/metrics folder")




 

