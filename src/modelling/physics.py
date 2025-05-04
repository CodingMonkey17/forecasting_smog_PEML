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


def compute_linear_y_phy(u, time_step=1, idx_dict=None, station_names=None, main_station=None):
    """
    Computes y_phy at a main station by modeling wind-based transport from multiple nearby stations.

    Parameters:
    - u: Tensor (batch_size, time_steps, features)
    - time_step: Time resolution (in hours)
    - idx_dict: Dict containing NO2_X_IDX and WIND_SPEED_IDX keys
    - station_names: List of all station names (e.g., ['tuindorp', 'breukelen', 'amsterdam'])
    - main_station: Name of the target station (e.g., 'breukelen')

    Returns:
    - y_phy: Tensor of shape (batch_size, N_HOURS_Y, 1)
    """
    assert idx_dict is not None, "idx_dict must be provided"

    assert station_names is not None, "station_names must be provided"

    assert main_station is not None, "main_station must be provided"

    batch_size, total_time_steps, _ = u.shape
    y_phy_total = torch.zeros((batch_size, N_HOURS_Y, 1), device=u.device)

    # Remove main station from contributing stations
    contributing_stations = [s for s in station_names if s != main_station]

    # Destination coordinates
    lat_dst = globals()[f"LAT_{main_station.upper()}"]
    lon_dst = globals()[f"LON_{main_station.upper()}"]

    # Wind speed
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


def create_pollution_field(grid, station_coords, station_pollutions, power=2):
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



def precompute_y_phy_for_all_batches(
    all_dataset_loader,
    chunk_dataset_loader,
    station_idx_dict,
    station_names,
    main_station,
    equation_version=1,
    output_file="output.pkl",
    log_dir="runs/y_phy_tracking",
    grid_fig_path="grid_layout.png"
):
    assert equation_version in [1, 2], "Only equation_version 1 or 2 is supported."
    assert main_station is not None, "main_station must be provided."
    assert station_names is not None, "station_names must be provided."

    print(f"Computing y_phy MULTI CITIY with equation {equation_version}...")
    writer = SummaryWriter(log_dir=log_dir)
    all_y_phy = []

    vx_min, vx_max, vy_min, vy_max = compute_global_min_max_vx_vy(all_dataset_loader, station_idx_dict)

    # Compute relative station positions from Breukelen
    
    other_stations = [s for s in station_names if s != main_station]
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
            pollution_values_2d = create_pollution_field(grid, station_coords, station_pollutions)
            c_m = ScalarField(grid, pollution_values_2d)

            # Save initial concentration field as image
            if batch_idx == 0 and b == 0:
                fig, ax = plt.subplots(figsize=(6, 6))
                c_m.plot(ax=ax)
                ax.set_title(f"Initial Pollution Concentration Field of Multi Cities")
                ax.set_xlabel("X (km)")
                ax.set_ylabel("Y (km)")
                plt.tight_layout()
                plt.savefig(grid_fig_path)
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



def compute_pinn_phy_loss(
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


def compute_pinn_phy_loss_new(
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
    # Compute gradients of dcdx and dcdy spatially again (second-order central approx)

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

    # Second derivatives (approximate as sum of squares of gradients / mean spacing)
    mean_dx2 = torch.mean(coords[:, 0] ** 2)
    mean_dy2 = torch.mean(coords[:, 1] ** 2)
    d2cdx2 = torch.var(delta_c_spatial, dim=2) / (mean_dx2 + 1e-6)
    d2cdy2 = torch.var(delta_c_spatial, dim=2) / (mean_dy2 + 1e-6)


    batch_size = u.shape[0]
    # Example: fixed temporal source profile
    hour_vector = torch.arange(24, device=u.device).float()
    strength = 0.5
    D = 0.1
    # S_t = strength * (torch.sin(np.pi * hour_vector / 24) ** 2 + 0.5 * torch.sin(2 * np.pi * hour_vector / 24) ** 2)  # shape: (24,)
    # S_t = S_t.unsqueeze(0).repeat(batch_size, 1)  # shape: (batch, 24)
    rush_hour_mask = ((hour_vector >= 7) & (hour_vector <= 9)) | ((hour_vector >= 16) & (hour_vector <= 19))
    S_t = strength * (rush_hour_mask.float() * 1.0 + (~rush_hour_mask).float() * 0.2)

    residual = dcdt + vx_output * dcdx + vy_output * dcdy - D * (d2cdx2 + d2cdy2) - S_t
    phy_loss = torch.mean(residual**2)
    return phy_loss


def compute_source_term(t, k=1.0, sigma=0.75):
    # Fixed peak centers
    morning_center = 8.0     # Between 7–9 AM
    evening_center = 17.25   # Between 4:30–6 PM

    # Fixed shape with two Gaussian peaks
    s_t = torch.exp(-((t - morning_center)**2) / (2 * sigma**2)) + \
          torch.exp(-((t - evening_center)**2) / (2 * sigma**2))

    # Scale with k
    return k * s_t


def compute_pinn_phy_loss_graph(
    y_pred, u, all_dataset_loader, station_names, main_station, idx_dict, knn=2, k=1.0, D=0.1
):
    import torch.nn.functional as F

    if y_pred.dim() == 3 and y_pred.shape[-1] == 1:
        y_pred_squeezed = y_pred.squeeze(-1)
    else:
        y_pred_squeezed = y_pred

    output_idx_start = N_HOURS_U - N_HOURS_Y + 1
    vx_min, vx_max, vy_min, vy_max = compute_global_min_max_vx_vy(all_dataset_loader, idx_dict=idx_dict)
    vx, vy = get_scaled_vx_vy(u, vx_min, vx_max, vy_min, vy_max, idx_dict=idx_dict)
    vx_output = vx[:, -N_HOURS_Y:]
    vy_output = vy[:, -N_HOURS_Y:]

    lat_main = globals()[f"LAT_{main_station.upper()}"]
    lon_main = globals()[f"LON_{main_station.upper()}"]
    idx_main = idx_dict[f"NO2_{main_station.upper()}_IDX"]

    c_main_hist = u[:, output_idx_start - 1, idx_main].unsqueeze(1)
    c_main_full = torch.cat([c_main_hist, y_pred_squeezed], dim=1)
    dcdt = torch.diff(c_main_full, dim=1)

    coords = []
    conc_stack = []
    station_data = {}
    main_station_index = None
    
    for i, station in enumerate(station_names):
        lat = globals()[f"LAT_{station.upper()}"]
        lon = globals()[f"LON_{station.upper()}"]
        x, y = latlon_to_xy(lat_main, lon_main, lat, lon)
        coords.append([x, y])
        
        idx = idx_dict[f"NO2_{station.upper()}_IDX"]
        station_data[station] = {'coords': (x, y), 'idx': idx}
        
        if station == main_station:
            main_station_index = i
            conc_stack.append(y_pred_squeezed.unsqueeze(2))
        else:
            c_other = u[:, -N_HOURS_Y:, idx]
            conc_stack.append(c_other.unsqueeze(2))
            station_data[station]['delta_c'] = (c_other - y_pred_squeezed).unsqueeze(2)

    coords = torch.tensor(coords, dtype=torch.float32, device=u.device)
    conc_tensor = torch.cat(conc_stack, dim=2)

    dist = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
    knn_mask = torch.topk(dist, k=knn + 1, largest=False).indices

    N = len(station_names)
    A = torch.zeros(N, N, device=u.device)
    for i in range(N):
        for j in knn_mask[i]:
            if i != j:
                A[i, j] = torch.exp(-dist[i, j])

    D_mat = torch.diag(A.sum(dim=1))
    L = D_mat - A

    laplacian_term = torch.einsum('ij,bti->btj', L, conc_tensor)
    lap_c_main = laplacian_term[:, :, main_station_index]

    delta_c_spatial = torch.cat(
        [station_data[s]['delta_c'] for s in station_names if s != main_station], dim=2
    )
    coord_deltas = torch.tensor(
        [station_data[s]['coords'] for s in station_names if s != main_station],
        dtype=torch.float32,
        device=u.device
    )

    A_spatial = torch.inverse(coord_deltas.T @ coord_deltas) @ coord_deltas.T
    grad_c = torch.einsum("ij,btk->bti", A_spatial, delta_c_spatial)
    dcdx = grad_c[:, :, 0]
    dcdy = grad_c[:, :, 1]

    
    hour_vector = torch.arange(24, device=u.device).float()
    S_t = compute_source_term(hour_vector, k=k)  # Your global/source strength parameter

    residual = dcdt + vx_output * dcdx + vy_output * dcdy - D * lap_c_main - S_t
    phy_loss = torch.mean(residual**2)
    return phy_loss


def compute_initial_condition_loss(y_pred, u, idx_dict, station_name):
    """
    Enforce initial condition at t = 0 using historical observation.

    Args:
        y_pred: (batch, 24) — model predictions for 24 future hours
        u: input tensor (batch, T, features)
        idx_dict: index map for station features
        station_name: name of the main station (e.g., "breukelen")
    Returns:
        initial condition loss (scalar tensor)
    """
    idx_station = idx_dict[f"NO2_{station_name.upper()}_IDX"]

    # Assume last time step of input (u) corresponds to t = 0
    # That is: the last known observation before prediction starts
    c_initial = u[:, -N_HOURS_Y, idx_station]  # (batch,)
    c_pred_t0 = y_pred[:, 0]                   # (batch,)
    
    loss_ic = torch.mean((c_pred_t0 - c_initial) ** 2)
    return loss_ic

