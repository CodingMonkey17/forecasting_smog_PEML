import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader
from scipy.integrate import solve_ivp
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler

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


def mse_loss(y_pred, y_true):
    """Mean Squared Error loss."""
    criterion = nn.MSELoss()
    return criterion(y_pred, y_true)

def get_wind_direction(u):
    # Adjust lambda_phy based on wind direction (higher if wind is favorable)
    wind_direction_normalised = u[:, :, dd_idx]

    # Rescale the normalized wind direction (0 to 1) back to degrees (0° to 360°)
    wind_direction_degrees = wind_direction_normalised * 360
    return wind_direction_degrees

def compute_linear_y_phy(u, time_step=1):
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



def compute_weighted_total_loss(mse_loss = None, phy_loss = None, lambda_phy = 0.8, u = None):
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

    wind_direction_degrees = get_wind_direction(u)
    
    # # Adjust lambda_phy based on the wind direction (closer to 270° means favorable wind direction)
    # if not torch.mean(wind_direction_degrees) > 200:  # Threshold for favorable wind direction (adjust as needed)
    #     lambda_phy = 1 - lambda_phy  # if not favorable wind direction, then use 1 - lambda_phy
    # Total loss is the sum of MSE and Physics loss, weighted by lambda_phy
    total_loss = mse_loss + lambda_phy * phy_loss
    return total_loss


def compute_pde_numerical_y_phy(u, scaler = None):
    """
    Computes y_phy using the numerical solution of the advection PDE.
    
    Parameters:
    - u: Input tensor of shape (batch_size, N_HOURS_U, features)
    
    Returns:
    - y_phy: Tensor of shape (batch_size, N_HOURS_Y, 1) representing pollution levels in Breukelen.
    """
    batch_size, total_time_steps, num_features = u.shape  # Get batch size and dimensions

    # Extract wind speed and direction from input tensor
    wind_speed = u[:, :, fh_idx]  # Wind speed (FH) in m/s
    wind_direction = u[:, :, dd_idx] * 360  # Convert normalized [0,1] to degrees

    # Convert wind speed from m/s to km/h
    wind_speed_kmh = wind_speed * 3.6  

    # Compute wind velocity components (vx, vy) using wind direction
    vx = wind_speed_kmh * torch.cos(torch.deg2rad(wind_direction))  # Wind component in x
    vy = wind_speed_kmh * torch.sin(torch.deg2rad(wind_direction))  # Wind component in y

    def latlon_to_xy(lat1, lon1, lat2, lon2):
        """Convert lat/lon to approximate x, y in km using the equirectangular projection."""
        R = 6371  # Radius of Earth in km
        x = (lon2 - lon1) * (math.pi / 180) * R * math.cos(math.radians((lat1 + lat2) / 2))
        y = (lat2 - lat1) * (math.pi / 180) * R
        return x, y

    # Define known coordinates
    lat_tuindorp, lon_tuindorp = 52.10503, 5.12448 # Coordinates of Tuindorp based on valentijn thesis (52°06’18.1”N, 5°07’28.1”E) and converted
    lat_breukelen, lon_breukelen = 52.20153, 4.98741 # Positioned at a 30° angle from Tuindorp 

    # Convert to (x, y) in km
    x_breukelen, y_breukelen = latlon_to_xy(lat_tuindorp, lon_tuindorp, lat_breukelen, lon_breukelen)
    x_tuindorp, y_tuindorp = 0, 0  # Set Tuindorp as the origin

    # Time steps from hour 48 to hour 72 (matching physics assumptions)
    t_eval = np.linspace(0, N_HOURS_Y - 1, N_HOURS_Y)

    def advection_pde(t, C, vx, vy):
        """
        Advection equation: dC/dt + vx * dC/dx + vy * dC/dy = 0
        We solve it for a single trajectory per batch iteration.
        """
        dC_dt = -vx * (C / (x_breukelen - x_tuindorp)) - vy * (C / (y_breukelen - y_tuindorp))
        return dC_dt

    # Initialize y_phy output tensor
    y_phy = torch.zeros((batch_size, N_HOURS_Y, 1), device=u.device)

    for b in range(batch_size):
        # Initial condition: Pollution concentration at Tuindorp at the starting time (hour 48)
        C0 = u[b, -N_HOURS_Y, no2_idx].cpu().numpy()  # Initial NO2 concentration

        # Solve PDE numerically using solve_ivp
        sol = solve_ivp(
            advection_pde, 
            (0, N_HOURS_Y - 1),  # Time span
            [C0],  # Initial condition
            t_eval=t_eval,  # Evaluation time points
            args=(vx[b, -N_HOURS_Y].cpu().numpy(), vy[b, -N_HOURS_Y].cpu().numpy())
        )

        # Store the result in y_phy
        y_phy[b, :, 0] = torch.clamp(torch.tensor(sol.y[0], device=u.device), min=0, max=1)

    
    return y_phy, vx, vy, wind_direction


# Computing loss for tuning, training, testing the model for actual prediction
def compute_loss(y_pred, y_true, u, loss_function, lambda_phy = 0.0001):
    """
    Computes loss function based on global variable setting.
    - y_pred: Predicted pollution level
    - y_true: Ground truth pollution level
    - u: Input features containing wind direction, wind speed, and pollution history
    - loss_function: "MSE" or "Physics_Linear_MSE"
    - lambda_phy: Weighting factor (for wind dir towards Breukeln) for physics loss, e.g 0.8 weight for wind dir towards Breukeln

    Returns: Total loss (MSE or MSE + Physics loss)
    """
    basic_mse_loss = mse_loss(y_pred, y_true)

    if loss_function == "MSE":
        return basic_mse_loss

    elif loss_function == "Physics_Linear_MSE":
        y_phy = compute_linear_y_phy(u, time_step = 1)
        phy_loss = mse_loss(y_pred, y_phy) # L_phy (y_pred, y_phy) = MSE(y_pred, y_phy)
        total_weighted_loss = compute_weighted_total_loss(basic_mse_loss, phy_loss, lambda_phy, u) # L = L_mse + lambda_phy * L_phy
        return total_weighted_loss
        
    elif loss_function == "Physics_PDE":
        # after training the y_phy with pde, we can use it to compute the loss
        # Assuming y_train is your ground truth training labels


        y_phy, vx, vy, wind_direction = compute_pde_numerical_y_phy(u=u)
        phy_loss = mse_loss(y_pred, y_phy)
        total_weighted_loss = compute_weighted_total_loss(basic_mse_loss, phy_loss, lambda_phy, u)
        # if phy_loss > 200:
        #     print("------------PHY LOSS IS TOO HIGH------------")
        #     print("Phy loss", phy_loss)
        #     print("MSE loss", basic_mse_loss)  
        #     print("total_weighted_loss", total_weighted_loss) 
        #     print('vx', vx)
        #     print('vy', vy)
        #     print('wind_direction', wind_direction)
        #     print("y_pred", y_pred)
        #     print("y_phy", y_phy)

        print("Phy loss", phy_loss)
        print("MSE loss", basic_mse_loss)
        print("total_weighted_loss", total_weighted_loss)
        return total_weighted_loss, y_phy, y_pred