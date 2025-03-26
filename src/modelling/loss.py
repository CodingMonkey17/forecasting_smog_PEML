import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader
from scipy.integrate import solve_ivp
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from pde import CartesianGrid, ScalarField, VectorField, PDE
from scipy.interpolate import interp1d
import config
from config import *



def latlon_to_xy(lat1, lon1, lat2, lon2):
    """Convert lat/lon to approximate x, y in km using the equirectangular projection."""
    R = 6371  # Radius of Earth in km
    x = (lon2 - lon1) * (math.pi / 180) * R * math.cos(math.radians((lat1 + lat2) / 2))
    y = (lat2 - lat1) * (math.pi / 180) * R
    return x, y


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
    wind_direction_normalised = u[:, :, WIND_DIR_IDX]

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

    wind_direction_degrees = get_wind_direction(u)
    
    # # Adjust lambda_phy based on the wind direction (closer to 270° means favorable wind direction)
    # if not torch.mean(wind_direction_degrees) > 200:  # Threshold for favorable wind direction (adjust as needed)
    #     lambda_phy = 1 - lambda_phy  # if not favorable wind direction, then use 1 - lambda_phy
    # Total loss is the sum of MSE and Physics loss, weighted by lambda_phy
    total_loss = mse_loss + lambda_phy * phy_loss
    return total_loss

# Function to compute pollution concentration at Breukelen
def compute_pollution_at_breukelen(result, grid, x_breukelen, y_breukelen):
    """
    Given the PDE result, extract the pollution concentration at Breukelen's location.
    """
    # Convert Breukelen's physical coordinates (x_breukelen, y_breukelen) to grid indices
    x_idx = np.abs(grid.axes_coords[0] - x_breukelen).argmin()  # Find closest index on x-axis
    y_idx = np.abs(grid.axes_coords[1] - y_breukelen).argmin()  # Find closest index on y-axis
    
    # Extract the pollution concentration at Breukelen's grid point
    concentration_breukelen = result.data[x_idx, y_idx]  # Get value from result grid
    
    return concentration_breukelen

# Define the initial pollution concentration function
def get_linear_interpolate(x, y, C_tuindorp, C_breukelen, x_tuindorp, y_tuindorp, x_breukelen, y_breukelen):
    """
    Defines the initial pollution concentration function using linear interpolation
    based on the distances to both Tuindorp and Breukelen.
    
    Parameters:
    - x, y: Coordinates of the current grid point.
    - C_tuindorp: Pollution level at Tuindorp.
    - C_breukelen: Pollution level at Breukelen.
    
    Returns:
    - Pollution concentration at (x, y) based on distance-weighted interpolation.
    """
    # Calculate distances from the point (x, y) to Tuindorp and Breukelen
    dist_tuindorp = np.sqrt((x - x_tuindorp) ** 2 + (y - y_tuindorp) ** 2)
    dist_breukelen = np.sqrt((x - x_breukelen) ** 2 + (y - y_breukelen) ** 2)
    
    # Calculate the total distance (just for normalization)
    total_dist = dist_tuindorp + dist_breukelen
    
    # If the total distance is zero (this should happen only at Tuindorp or Breukelen),
    # we return the corresponding concentration level
    if total_dist == 0:
        return C_tuindorp if dist_tuindorp < dist_breukelen else C_breukelen
    
    # Linear interpolation: weight pollution levels based on distance
    weight_tuindorp = dist_breukelen / total_dist  # Closer to Tuindorp means higher weight for C_tuindorp
    weight_breukelen = dist_tuindorp / total_dist  # Closer to Breukelen means higher weight for C_breukelen
    
    # Return the interpolated concentration
    return weight_tuindorp * C_tuindorp + weight_breukelen * C_breukelen



def create_pollution_field(grid, C_tuindorp, C_breukelen, x_tuindorp, y_tuindorp, x_breukelen, y_breukelen):
    """
    Creates a 2D pollution field using the provided interpolation function.

    Parameters:
    - grid: CartesianGrid object defining the spatial grid.
    - f0_function: Function to compute pollution concentration at each grid point.
    - C_tuindorp: Pollution level at Tuindorp.
    - C_breukelen: Pollution level at Breukelen.
    - x_tuindorp, y_tuindorp: Coordinates of Tuindorp.
    - x_breukelen, y_breukelen: Coordinates of Breukelen.

    Returns:
    - pollution_values_2d: 2D numpy array representing the pollution field.
    """
    pollution_values_2d = np.zeros(grid.shape)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            x, y = grid.axes_coords[0][i], grid.axes_coords[1][j]
            pollution_values_2d[i, j] = get_linear_interpolate(x, y, C_tuindorp, C_breukelen, x_tuindorp, y_tuindorp, x_breukelen, y_breukelen)
    return pollution_values_2d

def compute_pde_numerical_const_y_phy(u):
    """
    Computes y_phy using the numerical solution of the advection PDE using PyPDE.
    Initial condition interpolates pollution levels from Tuindorp to Breukelen.
    
    Parameters:
    - u: Input tensor of shape (batch_size, N_HOURS_U, features)

    Returns:
    - y_phy: Tensor of shape (batch_size, N_HOURS_Y, 1) representing pollution levels at Breukelen.
    """
    batch_size, total_time_steps, num_features = u.shape  

    # Extract wind speed and direction
    wind_speed = u[:, :, WIND_SPEED_IDX]  
    wind_direction = u[:, :, WIND_DIR_IDX] * 360  

    wind_speed_kmh = wind_speed * 3.6  

    # Compute wind velocity components (vx, vy)
    vx = wind_speed_kmh * torch.cos(torch.deg2rad(wind_direction))  
    vy = wind_speed_kmh * torch.sin(torch.deg2rad(wind_direction))  

    # Define spatial grid for a region of 15x15 km (adjust based on domain)
    grid = CartesianGrid([[-10, 5], [0, 15]], [100, 100])  # 100x100 grid covering -10 to 5 km along x and 0 to 15 km along y

    # Define advection PDE
    advection_pde = PDE({"c": "-vx * c_x - vy * c_y"}, consts={"vx": 0, "vy": 0})

    # Convert to (x, y) in km
    x_breukelen, y_breukelen = latlon_to_xy(LAT_TUINDORP, LON_TUINDORP, LAT_BREUKELEN, LON_BREUKELEN)
    x_tuindorp, y_tuindorp = 0, 0  # Set Tuindorp as the origin


    # Initialize y_phy output tensor for pollution levels at Breukelen
    y_phy = torch.zeros((batch_size, N_HOURS_Y, 1), device=u.device)

    # Solve PDE for each batch and extract pollution at Breukelen
    for b in range(batch_size):
        # Pollution concentration at Tuindorp
        C_tuindorp = u[b, -N_HOURS_Y -1 , NO2_TUINDORP_IDX].cpu().numpy()
        C_breukelen = u[b, -N_HOURS_Y -1 , NO2_BREUKELEN_IDX].cpu().numpy()

        # Create a 2D pollution field using f0_function
        pollution_values_2d = create_pollution_field(grid, C_tuindorp, C_breukelen, x_tuindorp, y_tuindorp, x_breukelen, y_breukelen)
        # Create the ScalarField representing the initial concentration
        c_m = ScalarField(grid, pollution_values_2d)

        # Loop through time steps and solve PDE for each time step
        for t in range(N_HOURS_Y):
            
            # Use the same wind velocity for the entire grid
            advection_pde.consts["vx"] = vx[b, -N_HOURS_Y -1].cpu().numpy()
            advection_pde.consts["vy"] = vy[b, -N_HOURS_Y -1].cpu().numpy()

            # Solve PDE for the current time step
            result = advection_pde.solve(c_m, t_range = t, dt = 0.1)

            # Extract pollution concentration at Breukelen for the current time step
            y_phy[b, t, 0] = compute_pollution_at_breukelen(result, grid, x_breukelen, y_breukelen)

    return y_phy



def compute_pde_numerical_piecewise_y_phy(u):
    """
    Computes y_phy using the numerical solution of the piecewise constant advection PDE.
    
    Parameters:
    - u: Input tensor of shape (batch_size, N_HOURS_U, features)
    
    Returns:
    - y_phy: Tensor of shape (batch_size, N_HOURS_Y, 1) representing pollution levels in Breukelen.
    """
    batch_size, total_time_steps, num_features = u.shape  # Get batch size and dimensions

    # Extract wind speed and direction from input tensor
    wind_speed = u[:, :, WIND_SPEED_IDX]  # Wind speed (FH) in m/s
    wind_direction = u[:, :, WIND_DIR_IDX] * 360  # Convert normalized [0,1] to degrees

    # Convert wind speed from m/s to km/h
    wind_speed_kmh = wind_speed * 3.6  

    # Compute wind velocity components (vx, vy) using wind direction
    vx = wind_speed_kmh * torch.cos(torch.deg2rad(wind_direction))  # Wind component in x
    vy = wind_speed_kmh * torch.sin(torch.deg2rad(wind_direction))  # Wind component in y

    # Convert to (x, y) in km
    x_breukelen, y_breukelen = latlon_to_xy(LAT_TUINDORP, LON_TUINDORP, LAT_BREUKELEN, LON_BREUKELEN)
    x_tuindorp, y_tuindorp = 0, 0  # Set Tuindorp as the origin

    # Time steps from hour 48 to hour 72 (matching physics assumptions)
    t_eval = np.linspace(0, N_HOURS_Y - 1, N_HOURS_Y)

    def piecewise_advection_pde(t, C, vx_t, vy_t):
        """
        Piecewise constant advection equation:
        dC/dt + vx(t) * dC/dx + vy(t) * dC/dy = 0

        The velocity field (vx, vy) changes in time but remains constant in space.
        """
        # Use piecewise constant velocity: Find closest time index and use corresponding wind speed
        time_idx = min(int(t), len(vx_t) - 1)  
        vx_current = vx_t[time_idx]
        vy_current = vy_t[time_idx]

        dC_dt = -vx_current * (C / abs(x_breukelen - x_tuindorp)) - vy_current * (C / abs(y_breukelen - y_tuindorp))
        return dC_dt

    # Initialize y_phy output tensor
    y_phy = torch.zeros((batch_size, N_HOURS_Y, 1), device=u.device)

    for b in range(batch_size):
        # Initial condition: Pollution concentration at Tuindorp at the starting time (hour 48)
        C0 = u[b, -N_HOURS_Y, NO2_TUINDORP_IDX].cpu().numpy()  # Initial NO2 concentration

        # Solve PDE numerically using solve_ivp
        sol = solve_ivp(
            piecewise_advection_pde, 
            (0, N_HOURS_Y - 1),  # Time span
            [C0],  # Initial condition
            t_eval=t_eval,  # Evaluation time points
            args=(vx[b, -N_HOURS_Y:].cpu().numpy(), vy[b, -N_HOURS_Y:].cpu().numpy())  # Piecewise vx, vy
        )

        # Store the result in y_phy
        # Clamp the values to be within [0, 1] range for backpropagation
        y_phy[b, :, 0] = torch.clamp(torch.tensor(sol.y[0], device=u.device), min=0, max=1)

    return y_phy



# Computing loss for tuning, training, testing the model for actual prediction
def compute_loss(y_pred, y_true, u, loss_function, lambda_phy):
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
        # print(basic_mse_loss)
        return basic_mse_loss

    elif loss_function == "LinearShift_MSE":
        y_phy = compute_linear_y_phy(u, time_step = 1)
        phy_loss = mse_loss(y_pred, y_phy) # L_phy (y_pred, y_phy) = MSE(y_pred, y_phy)
        total_weighted_loss = compute_weighted_total_loss(basic_mse_loss, phy_loss, lambda_phy, u) # L = L_mse + lambda_phy * L_phy
        return total_weighted_loss
        
    elif loss_function == "PDE_nmer_const":
        # after training the y_phy with pde, we can use it to compute the loss
        # Assuming y_train is your ground truth training labels


        y_phy = compute_pde_numerical_const_y_phy(u=u)
        phy_loss = mse_loss(y_pred, y_phy)
        total_weighted_loss = compute_weighted_total_loss(basic_mse_loss, phy_loss, lambda_phy, u)


        
        return total_weighted_loss
    elif loss_function == "Physics_PDE_numerical_piecewise":
        # print("Computing loss for Physics_PDE_numerical_piecewise")
        y_phy = compute_pde_numerical_piecewise_y_phy(u=u)
        phy_loss = mse_loss(y_pred, y_phy)
        total_weighted_loss = compute_weighted_total_loss(basic_mse_loss, phy_loss, lambda_phy, u)

        print("Phy loss", phy_loss)
        print("MSE loss", basic_mse_loss)
        print("total_weighted_loss", total_weighted_loss)
        return total_weighted_loss