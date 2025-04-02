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

def latlon_to_xy(lat1, lon1, lat2, lon2):
    """Convert lat/lon to approximate x, y in km using the equirectangular projection."""
    R = 6371  # Radius of Earth in km
    x = (lon2 - lon1) * (math.pi / 180) * R * math.cos(math.radians((lat1 + lat2) / 2))
    y = (lat2 - lat1) * (math.pi / 180) * R
    return x, y



def precompute_y_phy_for_all_batches_eq1(
    dataset_loader,
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
    
    writer = SummaryWriter(log_dir=log_dir)
    all_y_phy = []
    with open("physics_outputs/testing_empty.pkl", "wb") as f:
        pickle.dump(all_y_phy, f)
    # Step 1: Compute global min/max values
    vx_min, vx_max, vy_min, vy_max = compute_global_min_max_vx_vy(dataset_loader)
    # Convert to (x, y) in km
    x_breukelen, y_breukelen = latlon_to_xy(LAT_TUINDORP, LON_TUINDORP, LAT_BREUKELEN, LON_BREUKELEN)
    x_tuindorp, y_tuindorp = 0, 0  # Set Tuindorp as the origin

    advection_pde = PDE({"c": "- vx * d_dx(c) - vy * d_dy(c)"}, consts={"vx": 0, "vy": 0})
    # Define spatial grid for a region of 15x15 km (adjust based on domain)
    grid = CartesianGrid([[-10, 5], [0, 15]], [20, 20])  # 20x20 grid covering -10 to 5 km along x and 0 to 15 km along y

    
    for batch_idx, (u, _) in enumerate(dataset_loader):
        print(f"Processing batch {batch_idx + 1}/{len(dataset_loader)}")
        batch_size = u.shape[0]
        y_phy_batch = torch.zeros((batch_size, N_HOURS_Y, 1), device=u.device)

        vx, vy = get_scaled_vx_vy(u, vx_min, vx_max, vy_min, vy_max)

        for b in range(batch_size):
            print(f"Processing batch {batch_idx + 1}/{len(dataset_loader)}, sample {b + 1}/{batch_size}")
            
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


def load_all_y_phy(loss_function):
    """
    Load all y_phy values for the entire dataset.
    - loss_function: "MSE" or "Physics_Linear_MSE" or "PDE_nmer_const" or "Physics_PDE_numerical_piecewise"

    Returns: List of all y_phy values for the entire dataset.
    """
    all_y_phy = []

    if loss_function == "PDE_nmer_const":
        with open("physics_outputs/y_phy_batchsize16_eq1_2017.pkl", "rb") as f:
            all_y_phy = pickle.load(f)
    elif loss_function == "Physics_PDE_numerical_piecewise":
        # Load y_phy values from file
        all_y_phy = torch.load("data/y_phy_pde_numerical_piecewise.pt")

    return all_y_phy