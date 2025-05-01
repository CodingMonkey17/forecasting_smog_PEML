import config
from config import *
import torch
from torch.utils.tensorboard import SummaryWriter
from pde import CartesianGrid, ScalarField, VectorField, PDE
import torch
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt 


def latlon_to_xy(lat1, lon1, lat2, lon2):
    """Convert lat/lon to approximate x, y in km using the equirectangular projection."""
    R = 6371  # Radius of Earth in km
    x = (lon2 - lon1) * (math.pi / 180) * R * math.cos(math.radians((lat1 + lat2) / 2))
    y = (lat2 - lat1) * (math.pi / 180) * R
    return x, y

def extract_station_names_from_idx_dict(idx_dict):
    """
    Extracts all station names from an idx_dict with keys like 'NO2_TUINDORP_IDX'.

    Parameters:
    - idx_dict (dict): Dictionary with keys such as 'NO2_TUINDORP_IDX'

    Returns:
    - List of station names in lowercase (e.g., ['tuindorp', 'breukelen'])
    """
    station_names = []
    for key in idx_dict:
        if key.startswith("NO2_") and key.endswith("_IDX"):
            station = key[len("NO2_"):-len("_IDX")].lower()
            station_names.append(station)
    return station_names

def compute_linear_y_phy_multi(u, time_step=1, idx_dict=None):
    """
    Computes y_phy at Breukelen by modeling wind-based transport from multiple nearby stations.

    Parameters:
    - u: Tensor (batch_size, time_steps, features)
    - time_step: Time resolution (in hours)
    - idx_dict: Dict containing NO2_X_IDX and WIND_SPEED_IDX keys

    Returns:
    - y_phy: Tensor of shape (batch_size, N_HOURS_Y, 1)
    """
    assert idx_dict is not None, "idx_dict must be provided"

    
    
    batch_size, total_time_steps, _ = u.shape
    y_phy_total = torch.zeros((batch_size, N_HOURS_Y, 1), device=u.device)
    contributing_stations = extract_station_names_from_idx_dict(idx_dict)

    # Remove 'breukelen' since it's the destination
    contributing_stations = [s for s in contributing_stations if s != "breukelen"]

    # Destination (Breukelen)
    lat_dst, lon_dst = LAT_BREUKELEN, LON_BREUKELEN

    # Wind speed index
    wind_speed = u[:, :, idx_dict["WIND_SPEED_IDX"]] * 3.6  # Convert to km/h

    for station in contributing_stations:
        lat_src = globals()[f"LAT_{station.upper()}"]
        lon_src = globals()[f"LON_{station.upper()}"]

        # Compute relative coordinates
        x_src, y_src = latlon_to_xy(lat_dst, lon_dst, lat_src, lon_src)
        distance = math.sqrt(x_src**2 + y_src**2)

        # Get NO2 index
        no2_idx = idx_dict[f"NO2_{station.upper()}_IDX"]
        pollution = u[:, :, no2_idx]

        # Travel time per sample and time step
        travel_time = distance / (wind_speed + 1e-6)
        time_shifts = torch.ceil(travel_time / time_step).long()
        time_shifts = torch.clamp(time_shifts, min=0, max=N_HOURS_U - N_HOURS_Y)

        # Build the y_phy contribution from this station
        y_phy_station = torch.zeros((batch_size, N_HOURS_Y, 1), device=u.device)
        for b in range(batch_size):
            for t in range(N_HOURS_Y):
                shift_t = time_shifts[b, -N_HOURS_Y + t]
                src_idx = max(0, N_HOURS_U - N_HOURS_Y - shift_t + t)
                y_phy_station[b, t, 0] = pollution[b, src_idx]

        # Sum contribution (simple average here)
        y_phy_total += y_phy_station

    # Average across contributing stations
    y_phy = y_phy_total / len(contributing_stations)
    return y_phy


def compute_linear_y_phy_utrecht(u, time_step=1, idx_dict=None):
    """
    Computes y_phy using the same indexing technique as y.
    
    - u: Input tensor containing weather and pollution data (shape: batch_size, time_steps, features)
    - time_step: Time step interval (default 1 hour)
    
    Returns: y_phy of shape (batch_size, input_length, 1)
    """
    batch_size, total_time_steps, num_features = u.shape  # Get dimensions

    # Extract relevant features
    wind_speed = u[:, :, idx_dict[f"WIND_SPEED_IDX"]]  # Wind speed (FH), assumed in m/s
    pollution = u[:, :, idx_dict[f"NO2_TUINDORP_IDX"]]  # NO2 pollution (pollution at Breukelen)

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

def create_pollution_field_multi(grid, station_coords, station_pollutions, power=2):
    """
    Creates a 2D pollution field using inverse-distance weighted interpolation from multiple stations.

    Parameters:
    - grid: CartesianGrid object defining the spatial grid.
    - station_coords: List of (x, y) tuples for station positions (in km, relative to origin).
    - station_pollutions: List of pollution values (same order as station_coords).
    - power: Power for inverse-distance weighting (default: 2)

    Returns:
    - pollution_values_2d: 2D numpy array representing the pollution field.
    """
    pollution_values_2d = np.zeros(grid.shape)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            x, y = grid.axes_coords[0][i], grid.axes_coords[1][j]
            weights = []
            values = []
            for (x_s, y_s), pollution in zip(station_coords, station_pollutions):
                dist = np.sqrt((x - x_s) ** 2 + (y - y_s) ** 2) + 1e-6  # avoid divide by zero
                weights.append(1 / dist ** power)
                values.append(pollution)
            weights = np.array(weights)
            values = np.array(values)
            pollution_values_2d[i, j] = np.sum(weights * values) / np.sum(weights)
    return pollution_values_2d


def compute_global_min_max_vx_vy(dataset_loader, idx_dict):
    """ Compute global min and max for vx and vy across the entire dataset. """
    vx_min, vx_max = float('inf'), float('-inf')
    vy_min, vy_max = float('inf'), float('-inf')

    for u, _ in dataset_loader:
        wind_speed = u[:, :, idx_dict[f"WIND_SPEED_IDX"]] * 3.6  # Convert m/s to km/h
        wind_direction = u[:, :, idx_dict[f"WIND_DIR_IDX"]] * 360  # Convert to degrees

        vx = wind_speed * torch.cos(torch.deg2rad(wind_direction))
        vy = wind_speed * torch.sin(torch.deg2rad(wind_direction))

        vx_min = min(vx_min, vx.min().item())
        vx_max = max(vx_max, vx.max().item())
        vy_min = min(vy_min, vy.min().item())
        vy_max = max(vy_max, vy.max().item())

    return vx_min, vx_max, vy_min, vy_max

def get_scaled_vx_vy(u, vx_min, vx_max, vy_min, vy_max, idx_dict):
    # Compute wind velocity components
    wind_speed = u[:, :, idx_dict[f"WIND_SPEED_IDX"]] * 3.6
    wind_direction = u[:, :, idx_dict[f"WIND_DIR_IDX"]] * 360

    vx = wind_speed * torch.cos(torch.deg2rad(wind_direction))
    vy = wind_speed * torch.sin(torch.deg2rad(wind_direction))

    # Apply global Min-Max Scaling
    vx = (vx - vx_min) / (vx_max - vx_min + 1e-8)  # Add small constant to avoid div-by-zero
    vy = (vy - vy_min) / (vy_max - vy_min + 1e-8)
    return vx, vy



def precompute_y_phy_for_all_batches_multi(
    all_dataset_loader,
    chunk_dataset_loader,
    station_idx_dict,
    equation_version=1,
    output_file="output.pkl",
    log_dir="runs/y_phy_tracking",
    grid_fig_path="grid_layout.png"
):
    assert equation_version in [1, 2], "Only equation_version 1 or 2 is supported."

    print(f"Computing y_phy MULTI CITIY with equation {equation_version}...")
    writer = SummaryWriter(log_dir=log_dir)
    all_y_phy = []

    vx_min, vx_max, vy_min, vy_max = compute_global_min_max_vx_vy(all_dataset_loader, station_idx_dict)

    # Compute relative station positions from Breukelen
    all_stations = extract_station_names_from_idx_dict(station_idx_dict)
    other_stations = [s for s in all_stations if s != "breukelen"]
    station_coords = {}

    for station in other_stations:
        lat = globals()[f"LAT_{station.upper()}"]
        lon = globals()[f"LON_{station.upper()}"]
        x, y = latlon_to_xy(LAT_BREUKELEN, LON_BREUKELEN, lat, lon)
        station_coords[station] = (x, y)
    
    # Breukelen at origin
    station_coords["breukelen"] = (0, 0)

    # Determine bounds with margin
    xs, ys = zip(*station_coords.values())
    x_min, x_max = min(xs) - 5, max(xs) + 5
    y_min, y_max = min(ys) - 5, max(ys) + 5

    grid = CartesianGrid([[x_min, x_max], [y_min, y_max]], [20, 20])

    # PDE setup
    advection_pde = PDE({"c": "- vx * d_dx(c) - vy * d_dy(c)"}, consts={"vx": 0, "vy": 0})
    x_breukelen, y_breukelen = 0, 0

    for batch_idx, (u, _) in enumerate(chunk_dataset_loader):
        print(f"Processing batch {batch_idx + 1}/{len(chunk_dataset_loader)}")
        batch_size = u.shape[0]
        y_phy_batch = torch.zeros((batch_size, N_HOURS_Y, 1), device=u.device)

        vx, vy = get_scaled_vx_vy(u, vx_min, vx_max, vy_min, vy_max, station_idx_dict)
        vx_output = vx[:, -N_HOURS_Y:]
        vy_output = vy[:, -N_HOURS_Y:]

        for b in range(batch_size):
            print(f"  Sample {b + 1}/{batch_size}")

            # Create field with pollution from all stations except Breukelen
            pollution_values_2d = np.zeros(grid.shape)

            station_names = extract_station_names_from_idx_dict(station_idx_dict)
            # 1. Define coordinates relative to Breukelen
            station_coords = []
            station_pollutions = []

            for station in station_names:  # station_names should include all surrounding stations
                if station == "breukelen":
                    x, y = 0.0, 0.0  # Origin
                else:
                    lat, lon = globals()[f"LAT_{station.upper()}"], globals()[f"LON_{station.upper()}"]  
                    x, y = latlon_to_xy(LAT_BREUKELEN, LON_BREUKELEN, lat, lon)
                
                station_coords.append((x, y))
                
                # 2. Extract pollution value at the current timestep
                station_key = f"NO2_{station.upper()}_IDX"
                pollution_val = u[b, -N_HOURS_Y - 1, station_idx_dict[station_key]].item()  # .item() to get scalar
                
                station_pollutions.append(pollution_val)
                print(f"Station: {station}, Pollution: {pollution_val}, Coordinates: ({x}, {y})")

            # 3. Create the pollution field
            pollution_values_2d = create_pollution_field_multi(grid, station_coords, station_pollutions)
            c_m = ScalarField(grid, pollution_values_2d)

            # Save initial concentration field as image
            if batch_idx == 0 and b == 0:
                fig, ax = plt.subplots(figsize=(6, 6))
                c_m.plot(ax=ax)
                ax.set_title(f"Initial Pollution Concentration Field with boundaries")
                ax.set_xlabel("X (km)")
                ax.set_ylabel("Y (km)")
                plt.tight_layout()
                plt.savefig("initial_pollution_field.png")
                plt.close()
                print("Saved initial pollution field to initial_pollution_field.png")
                print('grid created with boundaries:', [x_min, x_max], [y_min, y_max])

            for t in range(N_HOURS_Y):
                vx_t = vx[b, -N_HOURS_Y - 1].item() if equation_version == 1 else vx_output[b, t].item()
                vy_t = vy[b, -N_HOURS_Y - 1].item() if equation_version == 1 else vy_output[b, t].item()

                advection_pde.consts["vx"] = vx_t
                advection_pde.consts["vy"] = vy_t

                result_data = advection_pde.solve(c_m, t_range=t, dt=0.001, tracker=None).data
                c_m.data = result_data

                y_phy_batch[b, t, 0] = compute_pollution_at_breukelen(result_data, grid, x_breukelen, y_breukelen)

                if b % 5 == 0 and t % 2 == 0:
                    writer.add_scalar("y_phy/value", y_phy_batch[b, t, 0].item(),
                                      global_step=(batch_idx * batch_size + b) * N_HOURS_Y + t)

            del result_data, c_m, pollution_values_2d

        all_y_phy.append(y_phy_batch)
        del y_phy_batch

    with open(output_file, "wb") as f:
        pickle.dump(all_y_phy, f)
    print(f"y_phy for all batches saved to {output_file}")
    writer.close()




def precompute_y_phy_for_all_batches_utrecht(
    all_dataset_loader,
    chunk_dataset_loader,
    station_idx_dict,
    equation_version=1,
    output_file="output.pkl",
    log_dir="runs/y_phy_tracking"
):
    """
    Precompute y_phy using a PDE solver with two equation modes:
    - Equation 1: Constant wind from t-1 across all prediction steps.
    - Equation 2: Wind varies at each timestep.

    Args:
        all_dataset_loader: DataLoader for global wind stats.
        chunk_dataset_loader: DataLoader with batched input u.
        station_idx_dict: Dict of indices for NO2 and wind fields.
        equation_version: 1 or 2, for selecting PDE behavior.
        output_file: Path to save output.
        log_dir: TensorBoard log directory.
    """
    assert equation_version in [1, 2], "Only equation_version 1 or 2 is supported."

    print(f"Computing y_phy with equation {equation_version}...")
    writer = SummaryWriter(log_dir=log_dir)
    all_y_phy = []

    vx_min, vx_max, vy_min, vy_max = compute_global_min_max_vx_vy(all_dataset_loader, station_idx_dict)

    # Coordinates for grid setup
    x_breukelen, y_breukelen = latlon_to_xy(LAT_TUINDORP, LON_TUINDORP, LAT_BREUKELEN, LON_BREUKELEN)
    x_tuindorp, y_tuindorp = 0, 0  # Origin at Tuindorp

    grid = CartesianGrid([[-15, 5], [-5, 15]], [20, 20])
    advection_pde = PDE({"c": "- vx * d_dx(c) - vy * d_dy(c)"}, consts={"vx": 0, "vy": 0})

    for batch_idx, (u, _) in enumerate(chunk_dataset_loader):
        print(f"Processing batch {batch_idx + 1}/{len(chunk_dataset_loader)}")
        batch_size = u.shape[0]
        y_phy_batch = torch.zeros((batch_size, N_HOURS_Y, 1), device=u.device)

        vx, vy = get_scaled_vx_vy(u, vx_min, vx_max, vy_min, vy_max, station_idx_dict)
        vx_output = vx[:, -N_HOURS_Y:]
        vy_output = vy[:, -N_HOURS_Y:]

        for b in range(batch_size):
            print(f"  Sample {b + 1}/{batch_size}")

            C_tuindorp = u[b, -N_HOURS_Y - 1, station_idx_dict['NO2_TUINDORP_IDX']].numpy()
            C_breukelen = u[b, -N_HOURS_Y - 1, station_idx_dict['NO2_BREUKELEN_IDX']].numpy()

            pollution_values_2d = create_pollution_field(grid, C_tuindorp, C_breukelen,
                                                         x_tuindorp, y_tuindorp, x_breukelen, y_breukelen)
            c_m = ScalarField(grid, pollution_values_2d)

            for t in range(N_HOURS_Y):
                if equation_version == 1:
                    vx_t = vx[b, -N_HOURS_Y - 1].item()
                    vy_t = vy[b, -N_HOURS_Y - 1].item()
                elif equation_version == 2:
                    vx_t = vx_output[b, t].item()
                    vy_t = vy_output[b, t].item()

                advection_pde.consts["vx"] = vx_t
                advection_pde.consts["vy"] = vy_t

                result_data = advection_pde.solve(c_m, t_range=t, dt=0.001, tracker=None).data
                c_m.data = result_data

                y_phy_batch[b, t, 0] = compute_pollution_at_breukelen(result_data, grid, x_breukelen, y_breukelen)

                if b % 5 == 0 and t % 2 == 0:
                    writer.add_scalar("y_phy/value", y_phy_batch[b, t, 0].item(),
                                      global_step=(batch_idx * batch_size + b) * N_HOURS_Y + t)

            del result_data, c_m, pollution_values_2d

        all_y_phy.append(y_phy_batch)
        del y_phy_batch

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


def compute_pinn_phy_loss_utrecht(y_pred, u, all_dataset_loader, idx_dict):

    # --- 1. Extract necessary data from input tensor 'u' ---
    output_idx_start = N_HOURS_U - N_HOURS_Y + 1
    x_breukelen, y_breukelen = latlon_to_xy(LAT_TUINDORP, LON_TUINDORP, LAT_BREUKELEN, LON_BREUKELEN)
    delta_x = abs(x_breukelen - 0)  # Tuindorp is at (0, 0)
    delta_y = abs(y_breukelen - 0)  # Tuindorp is at (0, 0)
    vx_min, vx_max, vy_min, vy_max = compute_global_min_max_vx_vy(all_dataset_loader, idx_dict=idx_dict)
    vx, vy = get_scaled_vx_vy(u, vx_min, vx_max, vy_min, vy_max, idx_dict=idx_dict)
    vx_output = vx[:, -N_HOURS_Y:]
    vy_output = vy[:, -N_HOURS_Y:]
    
    c_tuindorp_t = u[:, -N_HOURS_Y:, idx_dict[f"NO2_TUINDORP_IDX"]]
    c_breukelen_hist_t48 = u[:, output_idx_start -1, idx_dict[f"NO2_BREUKELEN_IDX"]].unsqueeze(1) # u[:, 48, :].unsqueeze(1) -> (batch, 1)

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

def compute_pinn_phy_loss_multi(
    y_pred, u, all_dataset_loader, station_names, main_station, idx_dict
):
    """
    Generalized physics loss using least squares gradient from multiple stations.

    Args:
        y_pred: (batch, 24) or (batch, 24, 1)
        u: input tensor (batch, T, features)
        all_dataset_loader: for wind normalization
        station_names: list of station names (e.g., ["tuindorp", "breukelen", "amsterdam"])
        main_station: name of the main station (e.g., "breukelen")
        idx_dict: dictionary with feature indices (e.g., UTRECHT_IDX or AMSTERDAM_IDX)
    """
    if y_pred.dim() == 3 and y_pred.shape[-1] == 1:
        y_pred_squeezed = y_pred.squeeze(-1)
    else:
        y_pred_squeezed = y_pred

    output_idx_start = N_HOURS_U - N_HOURS_Y + 1
    vx_min, vx_max, vy_min, vy_max = compute_global_min_max_vx_vy(all_dataset_loader, idx_dict=idx_dict)
    vx, vy = get_scaled_vx_vy(u, vx_min, vx_max, vy_min, vy_max, idx_dict=idx_dict)
    vx_output = vx[:, -N_HOURS_Y:]
    vy_output = vy[:, -N_HOURS_Y:]

    # Get lat/lon and NO2 index of the main station
    lat_main = globals()[f"LAT_{main_station.upper()}"]
    lon_main = globals()[f"LON_{main_station.upper()}"]
    idx_main = idx_dict[f"NO2_{main_station.upper()}_IDX"]

    # Set origin at main station
    c_main_hist = u[:, output_idx_start - 1, idx_main].unsqueeze(1)
    c_main_full = torch.cat([c_main_hist, y_pred_squeezed], dim=1)
    dcdt = torch.diff(c_main_full, dim=1)

    delta_c_list = []
    coords = []

    for station in station_names:
        if station == main_station:
            continue
        lat = globals()[f"LAT_{station.upper()}"]
        lon = globals()[f"LON_{station.upper()}"]
        idx = idx_dict[f"NO2_{station.upper()}_IDX"]
        x, y = latlon_to_xy(lat_main, lon_main, lat, lon)
        coords.append([x, y])

        c_other = u[:, -N_HOURS_Y:, idx]  # (batch, 24)
        delta_c = c_other - y_pred_squeezed
        delta_c_list.append(delta_c.unsqueeze(2))  # (batch, 24, 1)

    delta_c_spatial = torch.cat(delta_c_list, dim=2)  # (batch, 24, N-1)
    coords = torch.tensor(coords, dtype=torch.float32, device=u.device)  # (N-1, 2)

    coords_T = coords.T  # (2, N-1)
    A = torch.inverse(coords_T @ coords) @ coords_T  # (2, N-1)
    grad_c = torch.einsum("ij,btk->bti", A, delta_c_spatial)  # (batch, 24, 2)
    dcdx = grad_c[:, :, 0]
    dcdy = grad_c[:, :, 1]

    residual = dcdt + vx_output * dcdx + vy_output * dcdy
    phy_loss = torch.mean(residual**2)
    return phy_loss


