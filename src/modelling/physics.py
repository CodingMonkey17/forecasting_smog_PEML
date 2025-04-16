import config
from config import *
import torch
from torch.utils.tensorboard import SummaryWriter
from pde import CartesianGrid, ScalarField, VectorField, PDE
import torch
from torch.utils.tensorboard import SummaryWriter
import math
import numpy as np
import pickle


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

# Function to compute pollution concentration at Breukelen
def compute_pollution_at_breukelen(result_data, grid, x_breukelen, y_breukelen):
    """
    Given the PDE result, extract the pollution concentration at Breukelen's location.
    """
    # Convert Breukelen's physical coordinates (x_breukelen, y_breukelen) to grid indices
    x_idx = np.abs(grid.axes_coords[0] - x_breukelen).argmin()  # Find closest index on x-axis
    y_idx = np.abs(grid.axes_coords[1] - y_breukelen).argmin()  # Find closest index on y-axis
    
    # Extract the pollution concentration at Breukelen's grid point
    concentration_breukelen = result_data[x_idx, y_idx]  # Get value from result grid
    
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

def compute_global_min_max_vx_vy(dataset_loader):
    """ Compute global min and max for vx and vy across the entire dataset. """
    vx_min, vx_max = float('inf'), float('-inf')
    vy_min, vy_max = float('inf'), float('-inf')

    for u, _ in dataset_loader:
        wind_speed = u[:, :, WIND_SPEED_IDX] * 3.6  # Convert m/s to km/h
        wind_direction = u[:, :, WIND_DIR_IDX] * 360  # Convert to degrees

        vx = wind_speed * torch.cos(torch.deg2rad(wind_direction))
        vy = wind_speed * torch.sin(torch.deg2rad(wind_direction))

        vx_min = min(vx_min, vx.min().item())
        vx_max = max(vx_max, vx.max().item())
        vy_min = min(vy_min, vy.min().item())
        vy_max = max(vy_max, vy.max().item())

    return vx_min, vx_max, vy_min, vy_max

def get_scaled_vx_vy(u, vx_min, vx_max, vy_min, vy_max):
    # Compute wind velocity components
    wind_speed = u[:, :, WIND_SPEED_IDX] * 3.6
    wind_direction = u[:, :, WIND_DIR_IDX] * 360

    vx = wind_speed * torch.cos(torch.deg2rad(wind_direction))
    vy = wind_speed * torch.sin(torch.deg2rad(wind_direction))

    # Apply global Min-Max Scaling
    vx = (vx - vx_min) / (vx_max - vx_min + 1e-8)  # Add small constant to avoid div-by-zero
    vy = (vy - vy_min) / (vy_max - vy_min + 1e-8)
    return vx, vy





def precompute_y_phy_for_all_batches_eq1(
    all_dataset_loader, chunk_dataset_loader,
    output_file="output.pkl", log_dir="runs/y_phy_tracking"
):
    """
    Precompute y_phy for all batches, save to a file, and log progress with TensorBoard.
    
    Parameters:
    - dataset_loader: DataLoader for the dataset.
    - grid: The grid for the simulation.
    - vx, vy: Wind velocities (assumed to be batch-specific).
    - N_HOURS_Y: Number of hours to compute pollution for.
    - x_tuindorp, y_tuindorp, x_breukelen, y_breukelen: Coordinates for Tuindorp and Breukelen.
    - output_file: The file to save the computed y_phy tensor.
    - log_dir: Directory for TensorBoard logs.
    """
    print("Computing y phy with equation 1...")
    writer = SummaryWriter(log_dir=log_dir)
    all_y_phy = []
    with open("physics_outputs/testing_empty.pkl", "wb") as f:
        pickle.dump(all_y_phy, f)
    # Step 1: Compute global min/max values
    vx_min, vx_max, vy_min, vy_max = compute_global_min_max_vx_vy(all_dataset_loader)
    # Convert to (x, y) in km
    x_breukelen, y_breukelen = latlon_to_xy(LAT_TUINDORP, LON_TUINDORP, LAT_BREUKELEN, LON_BREUKELEN)
    x_tuindorp, y_tuindorp = 0, 0  # Set Tuindorp as the origin

    advection_pde = PDE({"c": "- vx * d_dx(c) - vy * d_dy(c)"}, consts={"vx": 0, "vy": 0})
    # Define spatial grid for a region of 15x15 km (adjust based on domain)
    grid = CartesianGrid([[-15, 5], [-5, 15]], [20, 20])  # 20x20 grid 

    
    for batch_idx, (u, _) in enumerate(chunk_dataset_loader):
        print(f"Processing batch {batch_idx + 1}/{len(chunk_dataset_loader)}")
        batch_size = u.shape[0]
        y_phy_batch = torch.zeros((batch_size, N_HOURS_Y, 1), device=u.device)

        vx, vy = get_scaled_vx_vy(u, vx_min, vx_max, vy_min, vy_max)

        for b in range(batch_size):
            print(f"Processing batch {batch_idx + 1}/{len(chunk_dataset_loader)}, sample {b + 1}/{batch_size}")
            
            # Extract the intitial pollution levels at Tuindorp and Breukelen at 1 hour before prediction
            C_tuindorp = u[b, -N_HOURS_Y - 1, NO2_TUINDORP_IDX].numpy()
            C_breukelen = u[b, -N_HOURS_Y - 1, NO2_BREUKELEN_IDX].numpy()

            # create pollution field with grid and initial pollution values
            pollution_values_2d = create_pollution_field(grid, C_tuindorp, C_breukelen, x_tuindorp, y_tuindorp, x_breukelen, y_breukelen)
            c_m = ScalarField(grid, pollution_values_2d)

            for t in range(N_HOURS_Y):
                # Assign the wind velocity of 1 hour before the prediction
                advection_pde.consts["vx"] = float(vx[b, -N_HOURS_Y - 1].numpy())
                advection_pde.consts["vy"] = float(vy[b, -N_HOURS_Y - 1].numpy())

                # solve the pde and update c_m for the next timestep
                result_data = advection_pde.solve(c_m, t_range=t, dt = 0.001, tracker=None).data
                c_m.data = result_data

                # Compute calculated pollution concentration from the grid at Breukelens coordinates
                y_phy_batch[b, t, 0] = compute_pollution_at_breukelen(result_data, grid, x_breukelen, y_breukelen)

                # Log some values to TensorBoard
                if b % 5 == 0 and t % 2 == 0:  # Reduce logging overhead
                    writer.add_scalar("y_phy/value", y_phy_batch[b, t, 0].item(), global_step=(batch_idx * batch_size + b) * N_HOURS_Y + t)
            del result_data
            del c_m
            del pollution_values_2d
        all_y_phy.append(y_phy_batch)
        del y_phy_batch

    
        # Save as .pkl file
    with open(output_file, "wb") as f:
        pickle.dump(all_y_phy, f)
    print(f"y_phy for all batches saved to {output_file}")

    writer.close()

def precompute_y_phy_for_all_batches_eq2(
    all_dataset_loader, chunk_dataset_loader,
    output_file="output.pkl", log_dir="runs/y_phy_tracking"
):

    print("Computing y phy with equation 2...")
    writer = SummaryWriter(log_dir=log_dir)
    all_y_phy = []
    with open("physics_outputs/testing_empty.pkl", "wb") as f:
        pickle.dump(all_y_phy, f)
    # Step 1: Compute global min/max values
    vx_min, vx_max, vy_min, vy_max = compute_global_min_max_vx_vy(all_dataset_loader)
    # Convert to (x, y) in km
    x_breukelen, y_breukelen = latlon_to_xy(LAT_TUINDORP, LON_TUINDORP, LAT_BREUKELEN, LON_BREUKELEN)
    x_tuindorp, y_tuindorp = 0, 0  # Set Tuindorp as the origin

    advection_pde = PDE({"c": "- vx * d_dx(c) - vy * d_dy(c)"}, consts={"vx": 0, "vy": 0})
    # Define spatial grid for a region of 15x15 km (adjust based on domain)
    grid = CartesianGrid([[-15, 5], [-5, 15]], [20, 20])  # 20x20 grid 

    
    for batch_idx, (u, _) in enumerate(chunk_dataset_loader):
        print(f"Processing batch {batch_idx + 1}/{len(chunk_dataset_loader)}")
        batch_size = u.shape[0]
        y_phy_batch = torch.zeros((batch_size, N_HOURS_Y, 1), device=u.device)

        vx, vy = get_scaled_vx_vy(u, vx_min, vx_max, vy_min, vy_max)

        for b in range(batch_size):
            print(f"Processing batch {batch_idx + 1}/{len(chunk_dataset_loader)}, sample {b + 1}/{batch_size}")
            
            # Extract the intitial pollution levels at Tuindorp and Breukelen at 1 hour before prediction
            C_tuindorp = u[b, -N_HOURS_Y - 1, NO2_TUINDORP_IDX].numpy()
            C_breukelen = u[b, -N_HOURS_Y - 1, NO2_BREUKELEN_IDX].numpy()
            vx_output = vx[:, -N_HOURS_Y:]
            vy_output = vy[:, -N_HOURS_Y:]
            # create pollution field with grid and initial pollution values
            pollution_values_2d = create_pollution_field(grid, C_tuindorp, C_breukelen, x_tuindorp, y_tuindorp, x_breukelen, y_breukelen)
            c_m = ScalarField(grid, pollution_values_2d)

            for t in range(N_HOURS_Y):
                # Set vx, vy from KNMI predictions at time t (relative to start of prediction horizon)
                advection_pde.consts["vx"] = float(vx_output[b, t].numpy())
                advection_pde.consts["vy"] = float(vy_output[b, t].numpy())

                # solve the pde and update c_m for the next timestep
                result_data = advection_pde.solve(c_m, t_range=t, dt = 0.001, tracker=None).data
                c_m.data = result_data

                # Compute calculated pollution concentration from the grid at Breukelens coordinates
                y_phy_batch[b, t, 0] = compute_pollution_at_breukelen(result_data, grid, x_breukelen, y_breukelen)

                # Log some values to TensorBoard
                if b % 5 == 0 and t % 2 == 0:  # Reduce logging overhead
                    writer.add_scalar("y_phy/value", y_phy_batch[b, t, 0].item(), global_step=(batch_idx * batch_size + b) * N_HOURS_Y + t)
            del result_data
            del c_m
            del pollution_values_2d
        all_y_phy.append(y_phy_batch)
        del y_phy_batch

    
        # Save as .pkl file
    with open(output_file, "wb") as f:
        pickle.dump(all_y_phy, f)
    print(f"y_phy for all batches saved to {output_file}")

    writer.close()


def load_all_y_phy(phy_output_path, y_phy_filename):
    # File paths for each chunk
    file_1 = f"{phy_output_path}/{y_phy_filename}_1.pkl"
    file_2 = f"{phy_output_path}/{y_phy_filename}_2.pkl"
    file_3 = f"{phy_output_path}/{y_phy_filename}_3.pkl"
    file_4 = f"{phy_output_path}/{y_phy_filename}_4.pkl"

    # Load all four files
    with open(file_1, "rb") as f:
        y_phy_1 = pickle.load(f)
    with open(file_2, "rb") as f:
        y_phy_2 = pickle.load(f)

    with open(file_3, "rb") as f:
        y_phy_3 = pickle.load(f)
    with open(file_4, "rb") as f:
        y_phy_4 = pickle.load(f)
    # Concatenate the arrays
    all_y_phy = np.concatenate([y_phy_1, y_phy_2, y_phy_3, y_phy_4], axis=0)

    # Verify the shape of the concatenated array (optional)
    print("comcatemated four chunks of y phy")
    print(f"Shape of concatenated y_phy: {all_y_phy.shape}")
    return all_y_phy


def compute_pinn_phy_loss(y_pred, u, all_dataset_loader):

    # --- 1. Extract necessary data from input tensor 'u' ---
    output_idx_start = N_HOURS_U - N_HOURS_Y + 1
    x_breukelen, y_breukelen = latlon_to_xy(LAT_TUINDORP, LON_TUINDORP, LAT_BREUKELEN, LON_BREUKELEN)
    delta_x = abs(x_breukelen - 0)  # Tuindorp is at (0, 0)
    delta_y = abs(y_breukelen - 0)  # Tuindorp is at (0, 0)
    vx_min, vx_max, vy_min, vy_max = compute_global_min_max_vx_vy(all_dataset_loader)
    vx, vy = get_scaled_vx_vy(u, vx_min, vx_max, vy_min, vy_max)
    vx_output = vx[:, -N_HOURS_Y:]
    vy_output = vy[:, -N_HOURS_Y:]
    
    c_tuindorp_t = u[:, -N_HOURS_Y:, NO2_TUINDORP_IDX]
    c_breukelen_hist_t48 = u[:, output_idx_start -1, NO2_BREUKELEN_IDX].unsqueeze(1) # u[:, 48, :].unsqueeze(1) -> (batch, 1)

    if y_pred.dim() == 3 and y_pred.shape[-1] == 1:
        y_pred_squeezed = y_pred.squeeze(-1)
    else:
        # If y_pred is already (batch, 24), use it directly
        # Add a check or warning if shape is unexpected
        if y_pred.dim() != 2 or y_pred.shape[1] != N_HOURS_Y:
             print(f"Warning: Unexpected y_pred shape {y_pred.shape} in compute_pinn_phy_loss. Expected (batch, {N_HOURS_Y}) or (batch, {N_HOURS_Y}, 1).")
        y_pred_squeezed = y_pred
    # equation : dc/dt + vx * dc/dx + vy * dc/dy = 0


    # --- 2. Calculate PDE Terms (Units: km, h) ---
    # ∂c/∂t ≈ (c(t) - c(t-1)) / Δt_hours (where Δt = 1 hour)
    # Sequence needed: [c(48)_hist, c(49)_pred, ..., c(72)_pred]
    c_breukelen_full = torch.cat([c_breukelen_hist_t48, y_pred_squeezed], dim=1) # Shape: (batch, 1+24=25)
    delta_c_t = torch.diff(c_breukelen_full, dim=1) # Shape: (batch, 24) - Represents c(t) - c(t-1) for t=49 to 72
    dcdt = delta_c_t / 1.0 # Units: [C]/h

    

    delta_c_spatial = y_pred_squeezed - c_tuindorp_t # Shape: (batch, 24)
    dcdx = delta_c_spatial / delta_x # Shape: (batch, 24)
    dcdy = delta_c_spatial / delta_y # Shape: (batch, 24)

    # --- 3. Calculate PDE Residual ---
    # Residual = ∂c/∂t + v_x * ∂c/∂x + v_y * ∂c/∂y for t=49 to 72
    residual = dcdt + vx_output * dcdx + vy_output * dcdy # Shape: (batch, 24)

    # --- 4. Calculate Physics Loss ---
    # Mean Squared Error of the residual over the 24 predicted hours
    phy_loss = torch.mean(residual**2)

    return phy_loss
    