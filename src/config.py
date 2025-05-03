from pathlib import Path
BASE_DIR = Path.cwd().parents[0] # set it to the root directory of the project, not src

RESULTS_PATH = BASE_DIR / "src" / "results"


DATA_UTRECHT_PATH = BASE_DIR / "data" / "data_combined" / "Utrecht"
DATA_AMSTERDAM_PATH = BASE_DIR / "data" / "data_combined" / "Amsterdam"
DATA_MULTI_PATH = BASE_DIR / "data" / "data_combined" / "Multi"



PHY_OUTPUT_PATH = BASE_DIR / "src" / "physics_outputs"

N_HOURS_U = 24 * 3               # number of hours to use for input (number of days * 24 hours)
N_HOURS_Y = 24                    # number of hours to predict (1 day * 24 hours)
N_HOURS_STEP = 24                 # "sampling rate" in hours of the data; e.g. 24 
                                  # means sample an I/O-pair every 24 hours
                                  # the contaminants and meteorological vars

# for Tuindorp and Breukelens task
# dictionary for station idx
UTRECHT_IDX = {
    'NO2_TUINDORP_IDX': 5,
    'NO2_BREUKELEN_IDX': 4,
    'WIND_DIR_IDX': 0,
    'WIND_SPEED_IDX': 2
}


# for Amsterdam transferability task
# dictionary for station idx
AMSTERDAM_IDX = {
    'NO2_OUDEMEER_IDX' : 5,
    'NO2_HAARLEM_IDX' :  4,
    'WIND_DIR_IDX': 0,
    'WIND_SPEED_IDX': 2
}

MULTI_STATION_IDX = {
    'NO2_TUINDORP_IDX': 7,
    'NO2_BREUKELEN_IDX': 4,
    'NO2_OUDEMEER_IDX': 6,
    'NO2_ZEGVELD_IDX': 8,
    'NO2_KANTERSHOF_IDX': 5,
    'WIND_DIR_IDX': 0,
    'WIND_SPEED_IDX': 2
}




LAT_TUINDORP, LON_TUINDORP = 52.10503, 5.12448 # Coordinates of Tuindorp based on valentijn thesis (52°06’18.1”N, 5°07’28.1”E) and converted
LAT_BREUKELEN, LON_BREUKELEN = 52.20153, 4.98741 # Positioned at a 30° angle from Tuindorp 

# Coordinates of extra stations
LAT_ZEGVELD, LON_ZEGVELD = 52.1379467, 4.8381368
LAT_OUDEMEER, LON_OUDEMEER = 52.2799703, 4.7706757
LAT_KANTERSHOF, LON_KANTERSHOF = 52.320692, 4.9883988