import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader

# To do
# fix up the compute y phy, now it computes in shape of input size but not output
# find a way to compute the y_phy in the same time step as the y_pred, but just use u as the source of pollution and weather



# CHANGE THIS ACCORDING TO THE INDEX OF THE FEATURES IN YOUR DATASET (printed in run models nb)
no2_idx = 4 # NO2 index in the dataset
dd_idx = 0 # Wind direction index
fh_idx = 2 # Wind speed index

# Define the distance between Tuindorp and Breukelen in km
D_TUINDORP_BREUKELEN = 11.36  # Example: 10 km (adjust as needed)
N_HOURS_U = 24 * 3               # number of hours to use for input (number of days * 24 hours)
N_HOURS_Y = 24                    # number of hours to predict (1 day * 24 hours)
N_HOURS_STEP = 24                # step size for sliding window

# Function to calculate RMSE
def rmse(y_pred, y_true):
    return torch.sqrt(nn.MSELoss()(y_pred, y_true))

# Function to calculate SMAPE
def smape(y_pred, y_true):
    numerator = torch.abs(y_pred - y_true)
    denominator = (torch.abs(y_pred) + torch.abs(y_true)) / 2.0
    return 100 * torch.mean(numerator / denominator)



def compute_y_phy(u, time_step=1):
    """
    Computes y_phy using the same indexing technique as y.
    
    - u: Input tensor containing weather and pollution data (shape: batch_size, time_steps, features)
    - time_step: Time step interval (default 1 hour)
    
    Returns: y_phy of shape (batch_size, input_length, 1)
    """
    batch_size, total_time_steps, num_features = u.shape  # Get dimensions

    # Extract relevant features
    wind_speed = u[:, :, fh_idx]  # Wind speed (FH), assumed in m/s
    pollution = u[:, :, no2_idx]  # NO2 pollution (pollution at Breukelen)

    # Convert wind speed from m/s to km/h
    wind_speed_kmh = wind_speed * 3.6  

    # Compute travel time in hours: t = d / u
    travel_time = D_TUINDORP_BREUKELEN / (wind_speed_kmh + 1e-6)  # Avoid division by zero

    # print("travel_time:")
    # print(travel_time)
    # print(travel_time.shape)

    # Convert travel time to index shifts (time steps), rounding UP to nearest hour
    time_shifts = torch.ceil(travel_time / time_step).long()  # (batch_size, N_HOURS_U)
    # Ensure valid indexing (clamp time shift to stay within history range)
    time_shifts = torch.clamp(time_shifts, min=0, max=N_HOURS_U - N_HOURS_Y)

    # print("time_shifts:")
    # print(time_shifts)
    # print(time_shifts.shape)
    
# Compute y_phy by shifting pollution data accordingly
    y_phy = torch.zeros((batch_size, N_HOURS_Y, 1), device=u.device)  # Initialize tensor

    for b in range(batch_size):
        for t in range(N_HOURS_Y):
            shift_t = time_shifts[b, -N_HOURS_Y + t]  # Get time shift for each time step
            src_idx = max(0, N_HOURS_U - N_HOURS_Y - shift_t + t)  # Ensure valid index
            y_phy[b, t, 0] = pollution[b, src_idx]

    # print("y_phy:")
    # print(y_phy)
    # print(y_phy.shape)
    return y_phy

def compute_phy_loss(y_pred, u):
    """
    Computes physics-aware loss function based on the advection equation.
    - y_pred: Predicted pollution level
    - y_true: Ground truth pollution level
    - u: Input features containing wind direction, wind speed, and pollution history
    - time_shift: The computed time shift based on wind speed/direction (e.g., 1 hour)
    """
    # Compute y_phy
    time_shift = 1  # Assume 1-hour shift for now (adjust dynamically if needed)
    y_phy = compute_y_phy(u, time_shift)

    # print("y_phy:")
    # print(y_phy)
    # print(y_phy.shape)

    # print("y_pred:")
    # print(y_pred)
    # print(y_pred.shape) # sth wrong here
    # Compute physics loss
    physics_loss = nn.MSELoss()(y_pred, y_phy)

    
    return physics_loss


# Physics-aware loss function
def compute_loss(y_pred, y_true, u, loss_function, lambda_phy=0.1):
    """
    Computes loss function based on global variable setting.
    - y_pred: Predicted pollution level
    - y_true: Ground truth pollution level
    - u: Input features containing wind direction, wind speed, and pollution history
    - loss_function: "MSE" or "Physics_MSE"
    - lambda_phy: Weighting factor for physics loss (higher if wind blows from Tuindorp to Breukelen)

    Returns: Total loss (MSE or MSE + Physics loss)
    """
    mse_loss = nn.MSELoss()(y_pred, y_true)

    if loss_function == "MSE":
        return mse_loss

    elif loss_function == "Physics_MSE":
        phy_loss = compute_phy_loss(y_pred, u)
        # Adjust lambda_phy based on wind direction (higher if wind is favorable)
        wind_direction_normalised = u[:, :, dd_idx]

        # Rescale the normalized wind direction (0 to 1) back to degrees (0° to 360°)
        # formula: scaled = (x - min) / (max - min)
        wind_direction_degrees = wind_direction_normalised * 360
        
        # Adjust lambda_phy based on the wind direction (closer to 270° means favorable wind direction)
        if torch.mean(wind_direction_degrees) > 200:  # Threshold for favorable wind direction (adjust as needed)
            lambda_phy = 0.8  # Increase weight if wind is blowing towards Tuindorp to Breukelen (westward)
        else:
            lambda_phy = 0.2  # Default weight if wind is blowing in other directions

        # Total loss is the sum of MSE and Physics loss, weighted by lambda_phy
        total_loss = mse_loss + lambda_phy * phy_loss
        return total_loss