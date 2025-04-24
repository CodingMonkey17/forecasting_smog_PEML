# Physics Enhanced Machine Learning for Pollution Forecasting

## Project Overview

This repository contains the implementation of a Physics Enhanced Machine Learning approach to forecasting air pollution levels in Breukelen, Netherlands. The model is designed to predict NO2 concentrations 24 hours ahead based on historical weather data and pollution measurements.

## Introduction

This project focuses on accurate forecasting of smog/pollution levels using both machine learning techniques and physical constraints. By incorporating physics-based regularization into traditional neural network architectures, the model achieves more realistic and consistent forecasts compared to purely data-driven approaches.

The core components include:

- Time series dataset management with proper window handling
- MLP implementation with physics-informed loss functions
- Autoregressive prediction capabilities
- Rigorous evaluation framework

## Project Structure
```
forecasting_smog_PEML/
├── data/                  # Data directory (not included in repo)
│   ├── data_raw/          # Raw downloaded data files
│   └── data_combined/     # Preprocessed datasets
├── src/
│   ├── modelling/         # Model implementations
│   │   ├── MLP.py         # Basic MLP and physics-informed variants
│   │   ├── TimeSeriesDataset.py  # Custom dataset for time series handling
│   │   ├── loss.py        # Loss function implementations
│   │   └── physics.py     # Handles all physics calculations
│   ├── physics_outputs/   # All physics outputs computed for Physics models
│   ├── pipeline/          # Data preprocessing pipeline
│   ├── results/           # Results for all models
│   ├── run_models/        # Folder with all notebooks for running models
│   │   ├── run_mlp/       #  Notebooks for all MLPs
│   │   │   ├── run_baseline_mlp_allyears.ipynb         	       # Tuning, training, testing for Baseline model
│   │   │   ├── run_mlp_linear_allyears.ipynb           	       # Tuning, training, testing for Simple Linear Shift Physics model
│   │   │   ├── run_mlp_pde_nmer_const_allyears.ipynb   # Tuning, training, testing for Physics Numerical with Constant Equation model
│   │   │   ├── run_mlp_pde_nmer_piece_allyears.ipynb   # Tuning, training, testing for Physics Numerical with Piecewise Equation model
│   │   │   └── run_mlp_pinn_allyears.ipynb             		       # Tuning, training, testing for PINN model
│   ├── config.py          # All global variables
│   ├── preprocess.ipynb   # Preprocess datasets
│   └── run_pde_compute_y_phy.py    # Computation of Physics Outputs
```

## Data

The dataset includes **hourly time series data from 2017–2023**, excluding 2016 and 2019 (details in preprocessing documentation).

### Weather Data

- **Station**: Utrecht, De Bilt (S260)
- **Source**: KNMI (Royal Netherlands Meteorological Institute)
- **Access**: Automated via `src/pipeline/knmy.ipynb` using the [KNMY GitHub repo](https://github.com/KNMI/knmy)
- **Parameters**: Wind speed/direction, temperature, humidity, etc.

### Pollution Data

- **Stations**:
  - Utrecht, Tuindorp (NL10636)
  - Breukelen (NL10641)
- **Pollutants**: NO₂, O3, PM 2.5, PM 10 (This project is focused on NO₂)
- **Source**: RIVM (Dutch National Institute for Public Health and the Environment)
- **Access**: Manually downloaded from [RIVM Luchtmeetnet Dataset](https://data.rivm.nl/data/luchtmeetnet/Vastgesteld-jaar/) in "breed formaat" to `data/raw`

---

## Preprocessing & Usage

1. Run `preprocess.ipynb` to generate structured datasets from raw files.
2. Explore outdated experimental approaches in the  `old_notebooks/` directory.
3. Use `src/modelling/` for models and datasets ated implementation.

---

## Methodology

The forecasting pipeline is built on:

- **Sliding window** input: 72 hours of historical data
- **Autoregressive prediction**: Each hourly forecast is generated using the input from the previous hour and all prior context, maintaining temporal coherence across predictions
- **Physics-informed regularization**: Loss incorporates advection-like PDE constraints
- **Leakage prevention**: Carefully handles input-output boundaries

### Windowing Strategy

Each sample in the dataset consists of a **sliding window** with:

- **72 hours of input data**, used to provide historical context
- **24 hours of output predictions**, corresponding to the pollution levels in the following day

The windows **overlap** and are extracted in a rolling fashion, where the output portion starts at position `72 - 24 + 1 = 49` within the input sequence. This ensures that:

- For each predicted hour `t`, the model uses information up to `t-1`. For example:
  - In an MLP-based model, to predict the 49th hour, the model takes only the features from the 48th hour as input.
  - In an RNN-based model, the model has access to the full hidden state encoding of all previous time steps up to the 48th hour, effectively incorporating sequential dependencies.
- The prediction proceeds autoregressively, hour-by-hour, using the previous time step's input recursively for the next prediction.

This approach allows the network to maintain temporal continuity and learn patterns over multiple time horizons without data leakage.

### Model Architecture

The core model implemented so far is a **Multi-Layer Perceptron (MLP)** enhanced with physics-informed learning:

- Configurable hidden layers with **ReLU** activation
- **Adam optimizer** with learning rate scheduler
- **Early stopping** to avoid overfitting
- **Custom loss function** combining MSE and physics residual

---

## Dependencies

The dependencies used in this project are listed in requirements.txt, though only "ordinary" libraries such as numpy, pandas, and PyTorch are utilised.
Other libraries from notebooks might be needed, eg: Optuna for tuning
