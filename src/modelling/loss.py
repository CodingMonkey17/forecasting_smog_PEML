import torch
import torch.nn as nn
import numpy as np
import math
import config
from config import *
from modelling.physics import *


# Function to calculate RMSE
def rmse(y_pred, y_true):
    return torch.sqrt(nn.MSELoss()(y_pred, y_true))

# Function to calculate SMAPE
def smape(y_pred, y_true):
    numerator = torch.abs(y_pred - y_true)
    denominator = (torch.abs(y_pred) + torch.abs(y_true)) / 2.0
    return 100 * torch.mean(numerator / denominator)


def mse_loss(y_pred, y_true):
    """Mean Squared Error loss."""
    criterion = nn.MSELoss()
    return criterion(y_pred, y_true)


def latlon_to_xy(lat1, lon1, lat2, lon2):
    """Convert lat/lon to approximate x, y in km using the equirectangular projection."""
    R = 6371  # Radius of Earth in km
    x = (lon2 - lon1) * (math.pi / 180) * R * math.cos(math.radians((lat1 + lat2) / 2))
    y = (lat2 - lat1) * (math.pi / 180) * R
    return x, y





def compute_weighted_total_loss(mse_loss = None, phy_loss = None, ic_loss = 0, lambda_phy = 1e-5, lambda_ic = None, u = None):
    '''
    Computes the total loss as the sum of MSE and Physics loss, weighted by lambda_phy.
    - mse_loss: Mean Squared Error loss
    - phy_loss: Physics loss
    - lambda_phy: Weighting factor (for wind dir towards Breukeln) for physics loss, e.g 0.8 for wind dir towards Breukeln
    - u: Input features containing wind direction, wind speed, and pollution history
    '''
    # warning in case anything is None
    if mse_loss is None or phy_loss is None or u is None:
        print('Warning: some of the inputs are None')
    
    # for models without initial condition loss, return only the MSE loss and phy loss
    if lambda_ic is None:
        return mse_loss + lambda_phy * phy_loss
    

    # Total loss is the sum of MSE and Physics loss, weighted by lambda_phy
    total_loss = mse_loss + lambda_phy * phy_loss + lambda_ic * ic_loss
    return total_loss





def get_y_phy_batch(all_y_phy, batch_idx):
    
    return all_y_phy[batch_idx]
    


# Computing loss for tuning, training, testing the model for actual prediction
def compute_loss(y_pred, y_true, u, loss_function, lambda_phy, all_y_phy, batch_idx, train_loader = None, 
                 idx_dict = None, station_names = None, main_station = None, lambda_ic=None):
    """
    Computes loss function based on global variable setting.
    - y_pred: Predicted pollution level
    - y_true: Ground truth pollution level
    - u: Input features containing wind direction, wind speed, and pollution history
    - loss_function: "MSE"/ "LinearShift_MSE"/ "PDE_nmer_const"/ "PDE_nmer_piece"/ "PINN"
    - lambda_phy: Weighting factor (for wind dir towards Breukeln) for physics loss, e.g 0.8 weight for wind dir towards Breukelen
    - all_y_phy: Precomputed y_phy values for physics loss
    - batch_idx: Index of the current batch
    - train_loader: DataLoader for training data (needed for normalising vx vy for physics)

    Returns: Total loss (MSE or MSE + Physics loss)
    """
    # Detect device from y_pred (assumes y_pred and y_true are on same device)
    device = y_pred.device

    # Ensure y_true and u are also on the same device
    y_true = y_true.to(device)
    u = u.to(device)
    basic_mse_loss = mse_loss(y_pred, y_true)

    if loss_function == "MSE":
        # print(basic_mse_loss)
        return basic_mse_loss

    if idx_dict == None:
        return ValueError("No idx dict!")
    
    if station_names == None:
        return ValueError("No station names!")
    
    

    elif loss_function == "LinearShift_MSE":
        y_phy = compute_linear_y_phy(u, time_step = 1, idx_dict= idx_dict, station_names=station_names, main_station=main_station).to(device)
        phy_loss = mse_loss(y_pred, y_phy) # L_phy (y_pred, y_phy) = MSE(y_pred, y_phy)
        total_weighted_loss = compute_weighted_total_loss(basic_mse_loss, phy_loss, lambda_phy, u) # L = L_mse + lambda_phy * L_phy
        return total_weighted_loss
        
    elif loss_function == "PDE_nmer_const" or loss_function == "PDE_nmer_piece":
        # Ensure y_phy is loaded
        if all_y_phy is None:
            print("Error: all_y_phy is None. Please load the y_phy values first.")
            return None
        y_phy = get_y_phy_batch(all_y_phy, batch_idx).to(device)

        # Compute the loss
        phy_loss = mse_loss(y_pred, y_phy)
        total_weighted_loss = compute_weighted_total_loss(basic_mse_loss, phy_loss, lambda_phy, u)

        return total_weighted_loss
    elif loss_function == "PINN":
        # Calculate the physics loss using the PDE
        if train_loader is None:
            print("Error: train_loader is None. Please provide the train_loader.")
            return None
        phy_loss = compute_pinn_phy_loss(y_pred, u, train_loader, station_names=station_names, main_station=main_station, idx_dict=idx_dict).to(device)
        # phy_loss = compute_pinn_phy_loss_graph(y_pred, u, train_loader, station_names=station_names, main_station=main_station, idx_dict=idx_dict, k=k, D=D).to(device)
        # Combine the losses
        ic_loss = compute_initial_condition_loss(y_pred = y_pred, u=u, idx_dict=idx_dict, station_name=main_station).to(device)
        
        total_weighted_loss = compute_weighted_total_loss(basic_mse_loss, phy_loss, ic_loss, lambda_phy, lambda_ic, u)
        return total_weighted_loss
    




    