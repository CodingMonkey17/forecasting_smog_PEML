# %% [markdown]
# # **PEML MLP Architecture 1 - computing physics output and feed in NN with regularisation**
# ## Experiment 2 - Eq 2: y_phy calculated by solving pde Piecewise constant advection equation
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

LOSS_FUNC = "PDE_nmer_const" # PDE numerical solver with equation 1, of constant wind speed and direction
NN_TYPE = "MLP" # choose from "MLP", "RNN", "LSTM", "GRU"
CITY = 'Multi' 
#%%
if CITY == 'Utrecht':
    idx_dict = UTRECHT_IDX
    station_names = ['tuindorp', 'breukelen']
    main_station = 'breuklen'
elif CITY == 'Amsterdam':
    idx_dict = AMSTERDAM_IDX
    station_names = ['oudemeer', 'haarlem']
    main_station = 'haarlem'
elif CITY == 'Multi':
    idx_dict = MULTI_STATION_IDX
    station_names = ['tuindorp', 'breukelen', 'zegveld', 'oudemeer', 'kantershof']
    main_station = 'breukeln'
else:
    raise ValueError("CITY must be 'Utrecht', 'Amsterdam', or 'Multi'.")

# %%
if YEARS == [2017, 2018, 2020, 2021, 2022, 2023]:
    years = "allyears"
    MINMAX_PATH = MINMAX_PATH_ALLYEARS_MULTI
    DATASET_PATH = DATASET_PATH_ALLYEARS_MULTI

    
    print("Using all years")
    
elif YEARS == [2017]:
    years = "2017"
    MINMAX_PATH = MINMAX_PATH_2017_MULTI
    DATASET_PATH = DATASET_PATH_2017_MULTI
    print("Using 2017")
else:
    raise ValueError("Invalid years selected")

Y_PHY_FILENAME = f"y_phy_batchsize16_{LOSS_FUNC}_{years}_Multi"
MODEL_PATH_NAME = f'best_{NN_TYPE}_no2_{LOSS_FUNC}_{years}.pth'
RESULTS_METRICS_FILENAME = f'results_{NN_TYPE}_no2_{LOSS_FUNC}_{years}.csv'
BESTPARAMS_FILENAME = f'best_params_{NN_TYPE}_no2_{LOSS_FUNC}_{years}.txt'
PLOT_FILENAME = f'plot_{NN_TYPE}_no2_{LOSS_FUNC}_{years}.png'


print("Y_PHY_FILENAME", Y_PHY_FILENAME)
print("MINMAX_PATH: ", MINMAX_PATH)
print("DATASET_PATH: ", DATASET_PATH)

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


# %% [markdown]
# ## Computing y_phy 2017 with eq 1 PDE numerical solver

# %% [markdown]
# ### Computing from Habrok since it takes a long time (ran by converting this notebook to script -> interactive)
# ### Also ran this in 4 chunks (10 batches for first 3 chunks, and 11 batches for last chunk) due to OOM on Habrok

# %%
from itertools import islice

def get_dataloader_chunk(dataloader, start, end):
    """Extracts a chunk of the DataLoader from start to end."""
    return list(islice(dataloader, start, end))


# %%
# Create train & validation loaders (following the original code)
temp_batch_size = 16
# Extract different batch chunks manually
temp_train_loader = DataLoader(train_dataset, batch_size=temp_batch_size, shuffle=True)
temp_val_loader = DataLoader(val_dataset, batch_size=temp_batch_size, shuffle=False)
print("Train loader length: ", len(temp_train_loader))

# %%


first_10_batches = get_dataloader_chunk(temp_train_loader, 0, 10)
second_10_batches = get_dataloader_chunk(temp_train_loader, 10, 20)
third_10_batches = get_dataloader_chunk(temp_train_loader, 20, 30)
fourth_10_batches = get_dataloader_chunk(temp_train_loader, 30, 41)
print(f"length of dataloader: {len(temp_train_loader)}")

print(f"First 10 batches count: {len(first_10_batches)}")
print(f"Second 10 batches count: {len(second_10_batches)}")
print(f"Third 10 batches count: {len(third_10_batches)}")
print(f"Fourth batch count: {len(fourth_10_batches)} (should be 11 if dataset has 41 batches)")


batch_number = 1
eq_num = 1
if batch_number == 1:
    chunk_data_to_use = first_10_batches
    print(f"Using first 10 batches")
elif batch_number == 2:
    chunk_data_to_use = second_10_batches
    print(f"Using second 10 batches")
elif batch_number == 3:
    chunk_data_to_use = third_10_batches
    print(f"Using third 10 batches")
elif batch_number == 4:
    chunk_data_to_use = fourth_10_batches
    print(f"Using fourth 10 batches")




phy_path = f"{PHY_OUTPUT_PATH}/{Y_PHY_FILENAME}_{batch_number}.pkl"
print(f"Chunk Number {batch_number} 10 batches phy_path: ", phy_path)
precompute_y_phy_for_all_batches_multi(all_dataset_loader= temp_train_loader, chunk_dataset_loader=chunk_data_to_use, station_idx_dict= idx_dict,
                                         equation_version= eq_num,output_file = phy_path)



# # %%
# all_y_phy = load_all_y_phy(PHY_OUTPUT_PATH, Y_PHY_FILENAME)
# # Save into one combined pickle file
# combined_file = f"{PHY_OUTPUT_PATH}/{Y_PHY_FILENAME}_full.pkl"
# with open(combined_file, "wb") as f:
#     pickle.dump(all_y_phy, f)

# print(f"Combined y_phy saved to: {combined_file}")
# print(f"Total number of batches: {len(all_y_phy)}")


# # %%
# # Path to the full saved file
# combined_file = f"{PHY_OUTPUT_PATH}/{Y_PHY_FILENAME}_full.pkl"

# # Load it
# with open(combined_file, "rb") as f:
#     all_y_phy = pickle.load(f)

# # Check total batches and shape of first batch
# print(f"Total batches: {len(all_y_phy)}")
# print(f"Shape of first batch: {all_y_phy[0].shape}")

# %% [markdown]
# ### Moved the y_phy file computed from habrok to local
# ### Loading the computed y_phy

# # %%
# # Load it back
# combined_phy_path = f"{PHY_OUTPUT_PATH}/{Y_PHY_FILENAME}_full.pkl"
# print(f"Loading y_phy from file {combined_phy_path}")

# with open(combined_phy_path, "rb") as f:
#     all_y_phy_np = pickle.load(f)  # List of tensors
# # Convert each batch to a torch tensor (keep as a list)
# all_y_phy = [torch.from_numpy(batch) for batch in all_y_phy_np]

# print(f"Number of batches in all_y_phy: {len(all_y_phy)}")

# print(f"all_y_phy first batch shape: {all_y_phy[0].shape}")

# # %% [markdown]
# # ### Confirming the computing y_phy has same shape as the y_true 

# # %%
# for i, (data, output) in enumerate(temp_train_loader):  # train_loader yields (input_data, labels)

#     y_phy_batch = all_y_phy[i]  # Get corresponding precomputed physics output
#     # Compare shapes
#     if output.shape == y_phy_batch.shape:
#         print(f"Batch {i} matches shape: {output.shape}, y_phy {y_phy_batch.shape}")
#     else:
#         print(f"Batch {i} shape mismatch: train_loader {output.shape}, y_phy {y_phy_batch.shape}")


