from pathlib import Path
BASE_DIR = Path.cwd().parents[0] # set it to the root directory of the project, not src
MODEL_PATH = BASE_DIR /"src" / "results" / "models"
RESULTS_PATH = BASE_DIR / "src" / "results"



MINMAX_PATH_2017 = BASE_DIR  / "data" / "data_combined" / "only_2017"/ "pollutants_minmax_2017.csv"
MINMAX_PATH_ALLYEARS = BASE_DIR  / "data" / "data_combined" / "all_years"/ "pollutants_minmax_allyears.csv"
MINMAX_PATH_FIRST3YEARS = BASE_DIR  / "data" / "data_combined" / "first_3_years"/ "pollutants_minmax_2017_2018_2020.csv"
DATASET_PATH_2017 = BASE_DIR / "data" / "data_combined" / "only_2017"
DATASET_PATH_ALLYEARS = BASE_DIR / "data" / "data_combined" / "all_years"
DATASET_PATH_FIRST3YEARS = BASE_DIR / "data" / "data_combined" / "first_3_years"


PHY_OUTPUT_PATH = BASE_DIR / "src" / "physics_outputs"

N_HOURS_U = 24 * 3               # number of hours to use for input (number of days * 24 hours)
N_HOURS_Y = 24                    # number of hours to predict (1 day * 24 hours)
N_HOURS_STEP = 24                 # "sampling rate" in hours of the data; e.g. 24 
                                  # means sample an I/O-pair every 24 hours
                                  # the contaminants and meteorological vars

NO2_TUINDORP_IDX =  5
NO2_BREUKELEN_IDX =  4
WIND_DIR_IDX =  0
WIND_SPEED_IDX =  2

LAT_TUINDORP, LON_TUINDORP = 52.10503, 5.12448 # Coordinates of Tuindorp based on valentijn thesis (52°06’18.1”N, 5°07’28.1”E) and converted
LAT_BREUKELEN, LON_BREUKELEN = 52.20153, 4.98741 # Positioned at a 30° angle from Tuindorp 