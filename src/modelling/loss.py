import torch
import torch.nn as nn
import numpy as np
import math
import config
from config import *


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

def compute_linear_y_phy(u, time_step=1):
    """
    Computes y_phy using the same indexing technique as y.
    
    - u: Input tensor containing weather and pollution data (shape: batch_size, time_steps, features)
    - time_step: Time step interval (default 1 hour)
    
    Returns: y_phy of shape (batch_size, input_length, 1)
    """
    batch_size, total_time_steps, num_features = u.shape  # Get dimensions

    # Extract relevant features
    wind_speed = u[:, :, WIND_SPEED_IDX]  # Wind speed (FH), assumed in m/s
    pollution = u[:, :, NO2_TUINDORP_IDX]  # NO2 pollution (pollution at Breukelen)

    # Convert wind speed from m/s to km/h
    wind_speed_kmh = wind_speed * 3.6  
    
     # Convert to (x, y) in km
    x_breukelen, y_breukelen = latlon_to_xy(LAT_TUINDORP, LON_TUINDORP, LAT_BREUKELEN, LON_BREUKELEN)
    x_tuindorp, y_tuindorp = 0, 0  # Set Tuindorp as the origin
    # Compute distance between Tuindorp and Breukelen
    distance = math.sqrt(x_breukelen**2 + y_breukelen**2)

    # Compute travel time in hours: t = d / u
    travel_time = distance / (wind_speed_kmh + 1e-6)  # Avoid division by zero
    # Convert travel time to index shifts (time steps), rounding UP to nearest hour
    time_shifts = torch.ceil(travel_time / time_step).long()  # (batch_size, N_HOURS_U)
    # Ensure valid indexing (clamp time shift to stay within history range)
    time_shifts = torch.clamp(time_shifts, min=0, max=N_HOURS_U - N_HOURS_Y)

    # Compute y_phy by shifting pollution data accordingly
    y_phy = torch.zeros((batch_size, N_HOURS_Y, 1), device=u.device)  # Initialize tensor

    for b in range(batch_size):
        for t in range(N_HOURS_Y):
            shift_t = time_shifts[b, -N_HOURS_Y + t]  # Get time shift for each time step
            src_idx = max(0, N_HOURS_U - N_HOURS_Y - shift_t + t)  # Ensure valid index
            y_phy[b, t, 0] = pollution[b, src_idx]

    return y_phy



def compute_weighted_total_loss(mse_loss = None, phy_loss = None, lambda_phy = 1e-5, u = None):
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

    # Total loss is the sum of MSE and Physics loss, weighted by lambda_phy
    total_loss = mse_loss + lambda_phy * phy_loss
    return total_loss


def get_y_phy_batch(all_y_phy, batch_idx):
    
    return all_y_phy[batch_idx]
    

# Computing loss for tuning, training, testing the model for actual prediction
def compute_loss(y_pred, y_true, u, loss_function, lambda_phy, all_y_phy, batch_idx):
    """
    Computes loss function based on global variable setting.
    - y_pred: Predicted pollution level
    - y_true: Ground truth pollution level
    - u: Input features containing wind direction, wind speed, and pollution history
    - loss_function: "MSE" or "Physics_Linear_MSE"
    - lambda_phy: Weighting factor (for wind dir towards Breukeln) for physics loss, e.g 0.8 weight for wind dir towards Breukelen
    - all_y_phy: Precomputed y_phy values for physics loss
    - batch_idx: Index of the current batch

    Returns: Total loss (MSE or MSE + Physics loss)
    """
    basic_mse_loss = mse_loss(y_pred, y_true)


    if loss_function == "MSE":
        # print(basic_mse_loss)
        return basic_mse_loss

    elif loss_function == "LinearShift_MSE":
        y_phy = compute_linear_y_phy(u, time_step = 1)
        phy_loss = mse_loss(y_pred, y_phy) # L_phy (y_pred, y_phy) = MSE(y_pred, y_phy)
        total_weighted_loss = compute_weighted_total_loss(basic_mse_loss, phy_loss, lambda_phy, u) # L = L_mse + lambda_phy * L_phy
        return total_weighted_loss
        
    elif loss_function == "PDE_nmer_const" or loss_function == "PDE_nmer_piece":
        # Ensure y_phy is loaded
        if all_y_phy is None:
            print("Error: all_y_phy is None. Please load the y_phy values first.")
            return None
        y_phy = get_y_phy_batch(all_y_phy, batch_idx)

        # Compute the loss
        phy_loss = mse_loss(y_pred, y_phy)
        total_weighted_loss = compute_weighted_total_loss(basic_mse_loss, phy_loss, lambda_phy, u)

        return total_weighted_loss
        