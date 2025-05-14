
# # **Transferability with all MLP Utrecht models**
# With Amsterdam data for transferability
# 

 
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

 
# Use GPU when available


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Device: ", device)


# ### **Set "global" variables**


Path.cwd()


import importlib
import config
importlib.reload(config)


from config import *


HABROK = bool(0)                  # set to True if using HABROK; it will print
                                  # all stdout to a .txt file to log progress


print("BASE_DIR: ", BASE_DIR)
print("Results path: ", RESULTS_PATH)




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


# Change this according to the data you want to use
YEARS = [2017, 2018, 2020, 2021, 2022, 2023]
TRAIN_YEARS = [2017, 2018, 2020, 2021, 2022]
VAL_YEARS = [2021, 2022, 2023]
TEST_YEARS = [2021, 2022, 2023]


NN_TYPE = "MLP" 
MODEL_CITY = 'Multi'
TRANSFER = True # set to True if you want to use transferability
TRANSFER_CITY = 'AmsMulti' # Amsterdam for transfer learning using Utrecht model, or AmsMulti for using Multi model


# ## Automated Generation of paths and filenames according to data years, loss func, NN type
# - will be used throughout the whole notebook
# - check ``config.py`` for global variables defined outside the notebook
valid_loss_funcs = ["MSE", "LinearShift_MSE", "PDE_nmer_const", "PDE_nmer_piece", "PINN"]

for loss_func in valid_loss_funcs:
    years, idx_dict, station_names, main_station, original_results_path, results_path, model_path, dataset_path, minmax_path, y_phy_filename, model_filename, results_metrics_filename, bestparams_filename, plot_filename = init_transferability_paths(model_city= MODEL_CITY, transfer_city=TRANSFER_CITY, years=YEARS, loss_func=loss_func, nn_type=NN_TYPE)
    print("years: ", years)
    print("idx_dict: ", idx_dict)
    print("station_names: ", station_names)
    print("original model main_station: ", main_station)
    print("RESULTS_PATH: ", results_path)
    print("original_results_path: ", original_results_path)
    print("MODEL_PATH: ", model_path)
    print("MINMAX_PATH: ", minmax_path)
    print("DATASET_PATH: ", dataset_path)
    print("Y_PHY_FILENAME: ", y_phy_filename)
    print("model_path_NAME: ", model_filename)
    print("RESULTS_METRICS_FILENAME: ", results_metrics_filename)
    print("BESTPARAMS_FILENAME: ", bestparams_filename)
    print("PLOT_FILENAME: ", plot_filename)


    # ### **Load in data and create PyTorch *Datasets***


    # Load in data and create PyTorch Datasets. To tune
    # which exact .csv files get extracted, change the
    # lists in the get_dataframes() definition

    train_input_frames = get_dataframes('train', 'u', YEARS, dataset_path)
    train_output_frames = get_dataframes('train', 'y', YEARS, dataset_path)

    val_input_frames = get_dataframes('val', 'u', YEARS, dataset_path)
    val_output_frames = get_dataframes('val', 'y', YEARS, dataset_path)

    test_input_frames = get_dataframes('test', 'u', YEARS, dataset_path)
    test_output_frames = get_dataframes('test', 'y', YEARS, dataset_path)

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


    # ## Tuning Hyperparameters with Optuna


    import json
    file_path = f"{original_results_path}/best_params/{bestparams_filename}"


    # ## Read params from file



    with open(file_path, "r") as f:
        best_params = json.load(f)  # Automatically converts it to a dictionary

    print(f"Loading best parms from {file_path}")
    print("Loaded Best Parameters:", best_params)


    if loss_func == 'PDE_nmer_const' or loss_func == 'PDE_nmer_piece':
        batch_size = 16
    else:
        batch_size = best_params["batch_size"]

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # ## Test and Save Results
    # Results saved in ``src/results/transferability/results_MLP_no2_MSE_allyears.csv``


    best_model_baseline.load_state_dict(torch.load(f"{model_path}/{model_filename}", map_location = device))
    print(f"Loading best model of {NN_TYPE} {loss_func} {years} from {model_path}/{model_filename}")
    best_model_baseline.eval()

    # Create the DataLoader for the test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model on the test dataset
    df_minmax = pd.read_csv(minmax_path, sep=';')
    min_value = df_minmax["min"].values
    max_value = df_minmax["max"].values
    mse, rmse, smape, inference_time_mean, inference_time_std  = best_model_baseline.test_model(test_loader, min_value=min_value, max_value=max_value, device=device)




    import csv

    # Define the CSV file path
    results_csv_path = f"{results_path}/{results_metrics_filename}"

    # Save metrics in a proper CSV format (header + values in one row)
    with open(results_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(["MSE", "RMSE", "SMAPE", "Inference Time", "Inference Time Std"])
        
        # Write values
        writer.writerow([mse, rmse, smape, inference_time_mean, inference_time_std])

    print(f"Results saved as {results_metrics_filename} in transferability folder")


    # ## Plot Model predictions vs True values
    # Plot saved ``src/transferability/plots/plot_MLP_no2_MSE_allyears.png``


    import torch 
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Load min and max values for denormalization
    df_minmax = pd.read_csv(minmax_path, sep=';')
    min_value = torch.tensor(df_minmax["min"].values, dtype=torch.float32)  # shape: (N_OUTPUT_UNITS,)
    max_value = torch.tensor(df_minmax["max"].values, dtype=torch.float32)  # shape: (N_OUTPUT_UNITS,)

    # Dynamically detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure model is on the right device and in eval mode
    best_model_baseline.to(device)
    best_model_baseline.eval()

    y_preds = []
    y_trues = []

    # Iterate through the test set and collect predictions & ground truth
    with torch.no_grad():
        for batch in test_loader:
            x_test, y_true = batch
            x_test = x_test.to(device)
            y_true = y_true.to(device)

            # Get predictions
            y_pred = best_model_baseline(x_test)

            # Move to CPU and store
            y_preds.append(y_pred.cpu())
            y_trues.append(y_true.cpu())

    # Stack batches
    y_preds = torch.cat(y_preds, dim=0)  # shape: (batch_size, n_hours_y, n_outputs)
    y_trues = torch.cat(y_trues, dim=0)

    # Denormalize
    min_value = min_value.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, n_outputs)
    max_value = max_value.unsqueeze(0).unsqueeze(0)
    y_preds_denorm = y_preds * (max_value - min_value) + min_value
    y_trues_denorm = y_trues * (max_value - min_value) + min_value

    # Convert to numpy for plotting
    y_preds_np = y_preds_denorm.numpy()
    y_trues_np = y_trues_denorm.numpy()

    # Plot 1 feature/channel (e.g., station 0)
    feature_idx = 0
    plt.figure(figsize=(15, 5))
    plt.plot(y_trues_np[:, :, feature_idx].flatten(), label="Ground Truth (NO₂)", linestyle="-", color="blue")
    plt.plot(y_preds_np[:, :, feature_idx].flatten(), label="Predictions", linestyle="-", color="black")

    plt.xlabel("Time Step")
    plt.ylabel("NO₂ Level")
    plt.title(f"Predictions vs. Ground Truth (Denormalized) for Amsterdam with {NN_TYPE} {MODEL_CITY} model and {loss_func}")
    plt.legend()
    plt.tight_layout()
    # Save the plot
    plt.savefig(f"{results_path}/{plot_filename}")
    plt.show()



