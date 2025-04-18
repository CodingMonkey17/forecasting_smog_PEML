# %% [markdown]
# # **PEML MLP Architecture 2 - PINN**
# ### All years data

# %% [markdown]
# ## **Running the models using the 'modelling' package**
# 
# A notebook through which different modelling configurations can be ran, using the ``modelling`` package. It follows the steps of:
# - preparing packages;
# - setting "global" variables;
# - getting the data;
# - defining hyperparameters;
# - running a grid search and/or training a model; and
# - evaluation.
# In the modelling package, variations can be made to the models and training functions to experiment. Don't forget to restart the notebook after making changes there.
# 
# 

# %% [markdown]
# For loading models, go to the ``src/results/models``:
# - Baseline NO2 2017 with MLP and MSE loss: ``best_mlp_no2_baseline_2017.pth``
# 

# %%
print("Starting script...")


from modelling.MLP import BasicMLP
from modelling import *
from modelling.physics import *


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
import pickle

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


print("BASE_DIR: ", BASE_DIR)
print("MODEL_PATH: ", MODEL_PATH)
print("Results path: ", RESULTS_PATH)

torch.manual_seed(34)             # set seed for reproducibility


# %% [markdown]
# 
# ## MODIFY THESE GLOBAL VARIABLES FOR YOUR MODEL SCENARIO
# ## all other variables are defined in config.py

# %%
# Change this according to the data you want to use
# Change this according to the data you want to use
YEARS = [2017, 2018, 2020, 2021, 2022, 2023]
TRAIN_YEARS = [2017, 2018, 2020, 2021, 2022]
VAL_YEARS = [2021, 2022, 2023]
TEST_YEARS = [2021, 2022, 2023]

LOSS_FUNC = "PINN" # PDE numerical solver with equation 1, of constant wind speed and direction
NN_TYPE = "MLP" # choose from "MLP", "RNN", "LSTM", "GRU"
torch.random.manual_seed(34)

# %%
if YEARS == [2017, 2018, 2020, 2021, 2022, 2023]:
    years = "allyears"
    MINMAX_PATH = MINMAX_PATH_ALLYEARS
    DATASET_PATH = DATASET_PATH_ALLYEARS

    
    print("Using all years")
    
elif YEARS == [2017]:
    years = "2017"
    MINMAX_PATH = MINMAX_PATH_2017
    DATASET_PATH = DATASET_PATH_2017
    print("Using 2017")
else:
    raise ValueError("Invalid years selected")

Y_PHY_FILENAME = f"y_phy_batchsize16_{LOSS_FUNC}_{years}"
MODEL_PATH_NAME = f'best_{NN_TYPE}_no2_{LOSS_FUNC}_{years}.pth'
RESULTS_METRICS_FILENAME = f'results_{NN_TYPE}_no2_{LOSS_FUNC}_{years}.csv'
BESTPARAMS_FILENAME = f'best_params_{NN_TYPE}_no2_{LOSS_FUNC}_{years}.txt'
PLOT_FILENAME = f'plot_{NN_TYPE}_no2_{LOSS_FUNC}_{years}.png'


print("Y_PHY_FILENAME", Y_PHY_FILENAME)
print("MINMAX_PATH: ", MINMAX_PATH)
print("DATASET_PATH: ", DATASET_PATH)
print("MODEL_PATH_NAME: ", MODEL_PATH_NAME)
print("RESULTS_METRICS_FILENAME: ", RESULTS_METRICS_FILENAME)
print("BESTPARAMS_FILENAME: ", BESTPARAMS_FILENAME)
print("PLOT_FILENAME: ", PLOT_FILENAME)

# %% [markdown]
# ### **Load in data and create PyTorch *Datasets***

# %%
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

# %%
train_input_frames

# %%
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

# %%
train_dataset.u

# %%
train_dataset.y

# %%
len(train_dataset.pairs[0][0])

# %%
train_dataset.pairs[0][0]

# %%
train_dataset.pairs[0][1]

# %%
# Assuming train_dataset.u[0] is a pandas Index object with column names
column_names = list(train_dataset.u[0])  # Convert Index to list


print("No2 tuindorp idx: ", column_names.index('NO2_TUINDORP'))
print("No2 breukelen idx: ", column_names.index('NO2_BREUKELEN'))
print("wind dir (dd) idx: ", column_names.index('DD'))
print("wind speed (fh) idx: ", column_names.index('FH'))

# check if the indices are the same as whats defined in config.py
assert column_names.index('NO2_TUINDORP')== NO2_TUINDORP_IDX
assert column_names.index('NO2_BREUKELEN') == NO2_BREUKELEN_IDX
assert column_names.index('DD') == WIND_DIR_IDX
assert column_names.index('FH') == WIND_SPEED_IDX
print("Column indices are same as config.py")



# %%
train_dataset.u[0].iloc[:,NO2_TUINDORP_IDX]

# %%
train_dataset.u[0].iloc[:,NO2_BREUKELEN_IDX]

# %%
train_dataset.u[0].iloc[:,WIND_DIR_IDX]

# %%
train_dataset.u[0].iloc[:,WIND_SPEED_IDX]

# %% [markdown]
# ## Tuning Hyperparamters

# # %%
# print("tuning with loss function: ", LOSS_FUNC)
# print("tuning with nn type: ", NN_TYPE)

# # %%

# def objective(trial):
#     # Define hyperparameters to search over
#     n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 5)
#     n_hidden_units = trial.suggest_int("n_hidden_units", 32, 256)
#     lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
#     weight_decay = trial.suggest_loguniform("weight_decay", 1e-8, 1e-3)
#     lambda_phy = trial.suggest_loguniform("lambda_phy", 1e-5, 1e-1)
#     batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])  # Match the original hp['batch_sz']

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
#     # Initialize MLP model
#     model = BasicMLP(
#         N_INPUT_UNITS=train_dataset.__n_features_in__(),
#         N_HIDDEN_LAYERS=n_hidden_layers,
#         N_HIDDEN_UNITS=n_hidden_units,
#         N_OUTPUT_UNITS=train_dataset.__n_features_out__(),
#         loss_function=LOSS_FUNC,
#     )

#     # Train and return validation loss
#     val_loss, _ = model.train_model(train_loader, val_loader, epochs=50, lr=lr, weight_decay=weight_decay, lambda_phy = lambda_phy, device=device)
    
#     return val_loss


# # Run Optuna optimization
# study = optuna.create_study(
#     direction="minimize", 
#     study_name="mlp_hyperparameter_optimization_PINN", 
#     storage="sqlite:///mlp_hyperparameter_optimization_phy_pde.db", 
#     load_if_exists=True,
#     pruner=optuna.pruners.HyperbandPruner(),
#     )
# study.optimize(objective, n_trials=50)

# # Print best hyperparameters
# best_params = study.best_params
# print("Best Hyperparameters:", best_params)

# # %%
# print(f"Best Hyperparameters for {NN_TYPE} with {LOSS_FUNC} for {years}:\n", best_params)

# # %%
# BESTPARAMS_FILENAME

# # %% [markdown]
# # ### Save params to file

# # %%
# import json
# best_params_file_path = f"{RESULTS_PATH}/best_params/{BESTPARAMS_FILENAME}"

# # %%
# with open(best_params_file_path, "w") as f:
#     json.dump(best_params, f, indent=4)  # Pretty format for readability

# print(f"Best Hyperparameters saved to {best_params_file_path}")


# # %% [markdown]
# # ### Read params from file

# # %%
# with open(best_params_file_path, "r") as f:
#     best_params = json.load(f)  # Automatically converts it to a dictionary

# print("Loaded Best Parameters:", best_params)


# # %%
# LOSS_FUNC

# %%
best_params = {'n_hidden_layers': 4, 'n_hidden_units': 191, 'lr': 0.0002099693529342787, 'weight_decay': 3.4334611075297167e-07, 'batch_size': 8, 'lambda_phy': 3.4051270015813734e-05}

# %%
torch.manual_seed(34)
# Train the model with the best hyperparameters
best_model = BasicMLP(
    N_INPUT_UNITS=train_dataset.__n_features_in__(),
    N_HIDDEN_LAYERS=best_params["n_hidden_layers"],
    N_HIDDEN_UNITS=best_params["n_hidden_units"],
    N_OUTPUT_UNITS=train_dataset.__n_features_out__(),
    loss_function=LOSS_FUNC,
)

# Create train & validation loaders with the best batch size
train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False)

# Train the model
_, training_time = best_model.train_model(train_loader, val_loader, epochs=50, lr=best_params["lr"], weight_decay=best_params["weight_decay"], lambda_phy= best_params["lambda_phy"], device=device)

print(f"Training time: {training_time}")
# Save the trained model
# torch.save(best_model.state_dict(), f"{MODEL_PATH}/{MODEL_PATH_NAME}")
# print(f"Model saved as {MODEL_PATH_NAME} in Model folder")

# %%
torch.manual_seed(34)  # Set seed for reproducibility
# best_model.load_state_dict(torch.load(f"{MODEL_PATH}/{MODEL_PATH_NAME}"))
best_model.eval()

# Create the DataLoader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=best_params["batch_size"], shuffle=False)

# Evaluate the model on the test dataset
df_minmax = pd.read_csv(MINMAX_PATH, sep=';')
min_value = df_minmax["min"].values
max_value = df_minmax["max"].values
mse, rmse, smape, inference_time = best_model.test_model(test_loader, min_value=min_value, max_value=max_value, device=device)



# %%
import csv

# Define the CSV file path
results_csv_path = f"{RESULTS_PATH}/metrics/{RESULTS_METRICS_FILENAME}"

# Save metrics in a proper CSV format (header + values in one row)
with open(results_csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    
    # Write header
    writer.writerow(["MSE", "RMSE", "SMAPE", "Inference Time", "Training Time"])
    
    # Write values
    writer.writerow([mse, rmse, smape, inference_time, training_time])

print(f"Results saved as {RESULTS_METRICS_FILENAME} in Results/metrics folder")

# %%
import torch
import matplotlib.pyplot as plt

# Ensure the model is in evaluation mode
best_model.eval()

y_preds = []
y_trues = []

# Iterate through the test set and collect predictions & ground truth
with torch.no_grad():
    for batch in test_loader:
        x_test, y_true = batch  # Get input and ground truth
        x_test = x_test.to("cpu")  # Ensure data is on CPU if needed

        # Get predictions
        y_pred = best_model(x_test)

        # Store results
        y_preds.append(y_pred.cpu())
        y_trues.append(y_true.cpu())

# Convert lists to tensors
y_preds = torch.cat(y_preds, dim=0).numpy()
y_trues = torch.cat(y_trues, dim=0).numpy()

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(y_trues.flatten(), label="Ground Truth (NO₂)", linestyle="-", color="blue")
plt.scatter(range(len(y_preds.flatten())), y_preds.flatten(), label="Predictions", color="black", s=10)

plt.xlabel("Time Step")
plt.ylabel("NO₂ Level")
plt.title("Predictions vs. Ground Truth")
plt.legend()
#save the plot
plt.savefig(f"{RESULTS_PATH}/plots/{PLOT_FILENAME}")
plt.show()



