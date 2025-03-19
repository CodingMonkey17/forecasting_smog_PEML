# %% [markdown]
# # **Preprocessing**
# 
# In preprocessing the data, the following steps are taken:
# 
# > * Prepare packages and setup  
# > * Load in the data  
# > * Tidy the data and store metadata  
# > * Inspect data with various metrics  
# > * Inspect data with visualisations 
# > * Select locations
# > * Select timeframe
# > * Feature engineering
# > * Perform train-validation-test-split  
# > * Normalisation  
# > * Create big combined normalised dataframe

# %% [markdown]
# ### **Prepare packages and setup**

# %%
from pipeline import read_meteo_csv_from_data_raw
from pipeline import read_four_contaminants
from pipeline import get_metadata
from pipeline import tidy_raw_contaminant_data
from pipeline import tidy_raw_meteo_data
from pipeline import print_aggegrated_sensor_metrics
from pipeline import subset_sensors
from pipeline import perform_data_split
from pipeline import perform_data_split_without_train
from pipeline import print_split_ratios
from pipeline import calc_combined_min_max_params
from pipeline import normalise_linear
from pipeline import print_pollutant_extremes
from pipeline import export_minmax
from pipeline import plot_distributions_KDE
from pipeline import concat_frames_horizontally
from pipeline import delete_timezone_from_index
from pipeline import assert_equal_shape
from pipeline import assert_equal_index
from pipeline import assert_no_NaNs
from pipeline import assert_range

# %%
SUBSET_MONTHS = bool(1)                 # If true, only the months specified in the list below will be
                                        # used for the training, validation and testing set
START_MON = '08'                        # starting month for the data
END_MON = '12'                          # ending month for the data

# Days used for the training, validation and testing splits
days_vali = 21
days_test = 21

days_vali_final_yrs = 63
days_test_final_yrs = 63

# ============================================================================+

# Sensor locations in the case of Utrecht area:
DE_BILT = 'S260'                        # starting (and only used) location for meteorological data
TUINDORP = 'NL10636'                    # starting location for contamination data
BREUKELEN = 'NL10641'                   # 'goal' location for contamination data
sensors_1D = [TUINDORP, BREUKELEN]

# =============================================================================

pollutants = ['NO2'] # adjustable for 'NO2', 'PM10', 'PM25', 'O3'
selected_weather = ['T', 'TD', 'DD', 'FH', 'P', 'SQ', 'FF', 'FX'] # features selected from corr matrix, plus 'FF' and 'FX'

# Define the years for the dataset
years = [2017, 2018, 2020, 2021, 2022, 2023] # adjustable for 2017, 2018, 2020, 2021, 2022, 2023

# At multiple locations, a sys.exit() can be used to halt the script

LOG = True

# %% [markdown]
# ### **Load in the data**

# %%
# Explicit variable declaration
df_NO2_2017_raw = None
df_NO2_2017_tidy = None
df_PM25_2017_raw = None
df_PM25_2017_tidy = None
df_PM10_2017_raw = None
df_PM10_2017_tidy = None
df_O3_2017_raw = None
df_O3_2017_tidy = None
df_meteo_2017_raw = None

df_NO2_2018_raw = None
df_NO2_2018_tidy = None
df_PM25_2018_raw = None
df_PM25_2018_tidy = None
df_PM10_2018_raw = None
df_PM10_2018_tidy = None
df_O3_2018_raw = None
df_O3_2018_tidy = None
df_meteo_2018_raw = None

df_NO2_2020_raw = None
df_NO2_2020_tidy = None
df_PM25_2020_raw = None
df_PM25_2020_tidy = None
df_PM10_2020_raw = None
df_PM10_2020_tidy = None
df_O3_2020_raw = None
df_O3_2020_tidy = None
df_meteo_2020_raw = None

df_NO2_2021_raw = None
df_NO2_2021_tidy = None
df_PM25_2021_raw = None
df_PM25_2021_tidy = None
df_PM10_2021_raw = None
df_PM10_2021_tidy = None
df_O3_2021_raw = None
df_O3_2021_tidy = None
df_meteo_2021_raw = None

df_NO2_2022_raw = None
df_NO2_2022_tidy = None
df_PM25_2022_raw = None
df_PM25_2022_tidy = None
df_PM10_2022_raw = None
df_PM10_2022_tidy = None
df_O3_2022_raw = None
df_O3_2022_tidy = None
df_meteo_2022_raw = None

df_NO2_2023_raw = None
df_NO2_2023_tidy = None
df_PM25_2023_raw = None
df_PM25_2023_tidy = None
df_PM10_2023_raw = None
df_PM10_2023_tidy = None
df_O3_2023_raw = None
df_O3_2023_tidy = None
df_meteo_2023_raw = None


# %%
# Loading raw data
for year in years:
    raw_data = read_four_contaminants(year, pollutants)
    for pollutant, data in zip(pollutants, raw_data):
        globals()[f'df_{pollutant}_{year}_raw'] = data

# %%
# Loading meteorological data
for year in years:
    globals()[f'df_meteo_{year}_raw'] = read_meteo_csv_from_data_raw(year)

# %%
df_NO2_2018_raw

# %%
df_NO2_2017_raw

# %%
df_meteo_2017_raw

# %%
df_meteo_2018_raw

# %%
if LOG:
    print('(1/8): Data read successfully')

# %% [markdown]
# ### **Tidy Pollutants data**
# Certain days which might affect pollutants are removed
# 
# Also only made it start from August (START_MON) until December (END_MON)

# %%
# Metadata extraction
for year in years:
    for pollutant in pollutants:
        raw_data_var = f'df_{pollutant}_{year}_raw'
        meta_var = f'{pollutant}_{year}_meta'
        if raw_data_var in globals():
            globals()[meta_var] = get_metadata(globals()[raw_data_var])
        else:
            print(f"Warning: {raw_data_var} is not defined.")

# Tidying data
for year in years:
    for pollutant in pollutants:
        raw_data_var = f'df_{pollutant}_{year}_raw'
        tidy_var = f'df_{pollutant}_{year}_tidy'
        if raw_data_var in globals():
            globals()[tidy_var] = tidy_raw_contaminant_data(
                globals()[raw_data_var], str(year), SUBSET_MONTHS, START_MON, END_MON
            )
        else:
            print(f"Warning: {raw_data_var} is not defined.")


# %%
df_NO2_2017_raw

# %%
df_NO2_2017_tidy

# %%
df_NO2_2018_tidy

# %%
pollutant_data = []
for year in years:
    for pollutant in pollutants:
        pollutant_data.append(eval(f'df_{pollutant}_{year}_tidy'))

assert_equal_shape(pollutant_data, True, False, 'Tidying of pollutant data')
print('(2/8): Pollutant data tidied successfully')

# %%
df_NO2_2017_tidy

# %%
df_meteo_2017_raw.columns

# %%
df_meteo_2018_raw.columns

# %%
# print the column of 2017 meteo data from 'FF'
df_meteo_2017_raw['FF']

# %% [markdown]
# ## **Extract and tidy meteorological data**

# %%
only_DeBilt = True  # True: only De Bilt is used
dataframes = {}

for year in years:
    raw_data = globals()[f'df_meteo_{year}_raw']
    for var in selected_weather:
        var_name = var # if var != 'P' else 'P_'  # Handling 'P' separately to match original naming (removed for now)
        df_name = f"df_{var_name}_{year}_tidy"
        dataframes[df_name] = tidy_raw_meteo_data(
            raw_data, var, only_DeBilt, str(year), SUBSET_MONTHS, START_MON, END_MON)

# Assign variables dynamically
globals().update(dataframes)


# %%
# print all the df names of dataframes
for df_name in dataframes:
    print(df_name)


# %%
df_T_2017_tidy

# %%
# Extract all DataFrames from the dictionary
datasets = list(dataframes.values())

# Ensure there is at least one dataset to check
assert len(datasets) > 0, "Error: No datasets found!"

# 1. Assert all datasets have the same shape
reference_shape = datasets[0].shape  # Take the shape of the first dataset as reference
assert all(df.shape == reference_shape for df in datasets), "Error: Not all datasets have the same shape!"

# 2. Assert no NaNs in any dataset
assert all(not df.isnull().values.any() for df in datasets), "Error: Some datasets contain NaN values!"

print("All datasets have the same shape and contain no NaN values.")
print('(3/8): Meteorological data tidied successfully')


# %% [markdown]
# ### **Inspect data with various metrics**
# 
# pipeline/statistics.py contains more functions

# %%
print("Printing some basic statistics for the pollutants:")
print("(Sensor NL10636 is TUINDORP)\n")

for pollutant in pollutants:
    dataframes = []
    for year in years:
        df_name = f'df_{pollutant}_{year}_tidy'
        try:
            dataframes.append(eval(df_name))
        except NameError:
            print(f"Warning: {df_name} does not exist")
    
    if dataframes:
        meta_name = f'{pollutant}_2017_meta' # meta data is just to know the pollutant name and unit
        print_aggegrated_sensor_metrics(dataframes, TUINDORP, eval(meta_name))


# %%
# Delete pollutant data
for year in years:
    for pollutant in pollutants:
        var_name = f"df_{pollutant}_{year}_raw"
        if var_name in globals():
            del globals()[var_name]

# Delete meteorological data
for year in years:
    var_name = f"df_meteo_{year}_raw"
    if var_name in globals():
        del globals()[var_name]


# %% [markdown]
# ### **Inspect data with visualisations**
# 
# plots.py contains plotting functions

# %% [markdown]
# ### **Select locations**

# %%
# Here, we'll select the locations we want to use. The
# generated dataframes will be 1-dimensional
# datasets are called df_pollutant_year_tidy_subset_1D


# Create subset dataframes dynamically
for year in years:
    for pollutant in pollutants:
        raw_var = f"df_{pollutant}_{year}_tidy"
        subset_var = f"{raw_var}_subset_1D"
        
        if raw_var in globals():
            globals()[subset_var] = subset_sensors(globals()[raw_var], sensors_1D)

# Delete original tidy dataframes after subsetting
for year in years:
    for pollutant in pollutants:
        raw_var = f"df_{pollutant}_{year}_tidy"
        if raw_var in globals():
            del globals()[raw_var]


# %%
df_NO2_2017_tidy_subset_1D

# %%
df_NO2_2018_tidy_subset_1D

# %% [markdown]
# Making sure all years and pollutants datasets have the same shapes

# %%
for year in years:
    for pollutant in pollutants:
        df_name = f'df_{pollutant}_{year}_tidy_subset_1D'
        try:
            df = eval(df_name)
            print(f"Year: {year}, Pollutant: {pollutant}, Shape: {df.shape}")
        except NameError:
            print(f"Warning: {df_name} does not exist")

# %% [markdown]
# ### **Perform assertions for pollutant tidy 1D data**

# %%
if LOG:
    
    # Collect dataframes dynamically
    shape_check_dfs = [
        globals()[f"df_{pollutant}_{year}_tidy_subset_1D"]
        for year in years
            for pollutant in pollutants
                if f"df_{pollutant}_{year}_tidy_subset_1D" in globals()
    ]

    nan_check_dfs = [
        globals()[f"df_{pollutant}_{year}_tidy_subset_1D"]
        for year in years
            for pollutant in pollutants
                if f"df_{pollutant}_{year}_tidy_subset_1D" in globals()
    ]

    # Perform assertions
    assert_equal_shape(shape_check_dfs, True, True, 'Location-wise subsetting of pollutant data')
    assert_no_NaNs(nan_check_dfs, 'Location-wise subsetting of pollutant data')

    print('(4/8): Location-wise subsetting of pollutant data successful')


# %% [markdown]
# ### **Select timeframe**
# 
# Timeframe selection (excluding 2016 and 2019) was done iteratively and manually by inspecting the data and inspecting the distributions. It is discussed in the thesis, Section 3.1, from Valentijn's thesis.
# 
# This has already been done in the beginning of the script.
# 

# %% [markdown]
# ### **Feature Engineering**
# 
# This is discussed in Section 3.2 of the thesis and done mainly with a plain correlation matrix.
# I'll use the same features that Valentijn has selected from correlation matrix, but add other wind features as well for the PEML model.

# %% [markdown]
# ### **Finding number of days for validation and test to achirve 75-15-15 splits**

# %%
import pandas as pd
# Calculate the total number of days from August 1st to December 30th
# if years are not [2017, 2018, 2020, 2021, 2022, 2023], then days vali and test are calculated manually
# if years are [2017, 2018, 2020, 2021, 2022, 2023], then days vali and test are already defined at the top
if years != [2017, 2018, 2020, 2021, 2022, 2023]:
    print("Assuming start date is August 1st and end date is December 30th in 2017.")
    print("Change accordingly if that's not the case")
    start_date = pd.Timestamp('2017-08-01')
    end_date = pd.Timestamp('2017-12-30')
    total_days = (end_date - start_date).days + 1

    # Calculate 15% of the total number of days for validation and test sets
    days_vali = int(total_days * 0.15)
    days_test = int(total_days * 0.15)
    print(days_vali, days_test)

# %% [markdown]
# ### **Perform train-validation-test-split**

# %%
# Splitting the data according to the years
# According to the splitting defined from Valentijn, 2017 2018 2020 are used for training
# 2021 2022 are used for validation and 2023 is used for testing
if len(years) == 1:
    year = years[0]
    years_splitted = years
    years_unsplitted = []
    year_final = 2023 # if only one year is used, the same year is used for training, validation and testing
    for pollutant in pollutants:
        df_train, df_val, df_test = perform_data_split(eval(f'df_{pollutant}_{year}_tidy_subset_1D'), days_vali, days_test)
        exec(f'df_{pollutant}_{year}_train_1D = df_train')
        exec(f'df_{pollutant}_{year}_val_1D = df_val')
        exec(f'df_{pollutant}_{year}_test_1D = df_test')

    for meteo in selected_weather:
        df_train, df_val, df_test = perform_data_split(eval(f'df_{meteo}_{year}_tidy'), days_vali, days_test)
        exec(f'df_{meteo}_{year}_train = df_train')
        exec(f'df_{meteo}_{year}_val = df_val')
        exec(f'df_{meteo}_{year}_test = df_test')
elif years == [2017, 2018, 2020, 2021, 2022, 2023]:
    years_unsplitted = [2017, 2018, 2020]
    years_splitted = [2021, 2022]
    year_final = 2023
    for year in years:
        if year in years_unsplitted:
            for pollutant in pollutants:
                exec(f'df_{pollutant}_{year}_train_1D = df_{pollutant}_{year}_tidy_subset_1D.copy()')
            for meteo in selected_weather:
                exec(f'df_{meteo}_{year}_train = df_{meteo}_{year}_tidy.copy()')
        elif year in years_splitted:
            for pollutant in pollutants:
                exec(f"df_{pollutant}_{year}_train_1D, df_{pollutant}_{year}_val_1D, df_{pollutant}_{year}_test_1D = "
                    f"perform_data_split(df_{pollutant}_{year}_tidy_subset_1D, days_vali, days_test)")
            for meteo in selected_weather:
                exec(f"df_{meteo}_{year}_train, df_{meteo}_{year}_val, df_{meteo}_{year}_test = "
                    f"perform_data_split(df_{meteo}_{year}_tidy, days_vali, days_test)")
        elif year == year_final:
            for pollutant in pollutants:
                exec(f"df_{pollutant}_{year}_val_1D, df_{pollutant}_{year}_test_1D = "
                    f"perform_data_split_without_train(df_{pollutant}_{year}_tidy_subset_1D, days_vali_final_yrs, days_test_final_yrs)")

            for meteo in selected_weather:
                exec(f"df_{meteo}_{year}_val, df_{meteo}_{year}_test = "
                    f"perform_data_split_without_train(df_{meteo}_{year}_tidy, days_vali_final_yrs, days_test_final_yrs)")

else:
    print("Error: train val test split not defined for the years")

# %% [markdown]
# ### Checking if shapes are the same for all datasets

# %%
if LOG:
     # First, check for equal shape of pollutant data of unsplitted years
    pollutant_data_unsplitted = []
    for year in years_unsplitted:
        if year in years:
            for pollutant in pollutants:
                df_name = f'df_{pollutant}_{year}_train_1D'
                pollutant_data_unsplitted.append(eval(df_name))
    assert_equal_shape(pollutant_data_unsplitted, True, True, 'Split of pollutant train set for 2017, 2018 and 2020')

    # Second, check for equal shape of meteorological data of unsplitted years
    meteorological_data_unsplitted = []
    for year in years_unsplitted:
        if year in years:
            for meteo in selected_weather:
                df_name = f'df_{meteo}_{year}_train'
                meteorological_data_unsplitted.append(eval(df_name))
    assert_equal_shape(meteorological_data_unsplitted, True, True, 'Split of meteorological train set for 2017, 2018 and 2020')

    # Third, check for equal row number of training set in 2021 and 2022
    training_data_splitted = []
    for year in years_splitted:
        if year in years:
            for pollutant in pollutants:
                df_name = f'df_{pollutant}_{year}_train_1D'
                training_data_splitted.append(eval(df_name))
            for meteo in selected_weather:
                df_name = f'df_{meteo}_{year}_train'
                training_data_splitted.append(eval(df_name))
    assert_equal_shape(training_data_splitted, True, False, 'Split of training data for 2021 and 2022')

    # Fourth, check for equal row number of validation set in 2021 and 2022
    validation_data_splitted = []
    for year in years_splitted:
        if year in years:
            for pollutant in pollutants:
                df_name = f'df_{pollutant}_{year}_val_1D'
                validation_data_splitted.append(eval(df_name))
            for meteo in selected_weather:
                df_name = f'df_{meteo}_{year}_val'
                validation_data_splitted.append(eval(df_name))
    assert_equal_shape(validation_data_splitted, True, False, 'Split of validation data for 2021 and 2022')

    # Fifth, check for equal row number of test set in 2021 and 2022
    test_data_splitted = []
    for year in years_splitted:
        if year in years:
            for pollutant in pollutants:
                df_name = f'df_{pollutant}_{year}_test_1D'
                test_data_splitted.append(eval(df_name))
            for meteo in selected_weather:
                df_name = f'df_{meteo}_{year}_test'
                test_data_splitted.append(eval(df_name))
    assert_equal_shape(test_data_splitted, True, False, 'Split of test data for 2021 and 2022')

    # Sixth, check for equal row number of validation set in 2023
    validation_data_final = []
    if year_final in years:
        for pollutant in pollutants:
            df_name = f'df_{pollutant}_{year_final}_val_1D'
            validation_data_final.append(eval(df_name))
        for meteo in selected_weather:
            df_name = f'df_{meteo}_{year_final}_val'
            validation_data_final.append(eval(df_name))
    assert_equal_shape(validation_data_final, True, False, 'Split of validation data for 2023')

    # Seventh, check for equal row number of test set in 2023
    test_data_final = []
    if year_final in years:
        for pollutant in pollutants:
            df_name = f'df_{pollutant}_{year_final}_test_1D'
            test_data_final.append(eval(df_name))
        for meteo in selected_weather:
            df_name = f'df_{meteo}_{year_final}_test'
            test_data_final.append(eval(df_name))
    assert_equal_shape(test_data_final, True, False, 'Split of test data for 2023')

    print('(5/8): Train-validation-test split successful')
    del pollutant_data_unsplitted, meteorological_data_unsplitted, training_data_splitted, validation_data_splitted, test_data_splitted, validation_data_final, test_data_final

# %%
df_NO2_2018_train_1D

# %%
df_NO2_2021_test_1D

# %%
df_NO2_2021_val_1D

# %% [markdown]
# ### Double checking the train val test split ratio

# %%
for pollutant in pollutants:
    train_dfs = []
    val_dfs = []
    test_dfs = []

    for year in years_unsplitted:
        if year in years:
            df_name = f'df_{pollutant}_{year}_train_1D'
            train_dfs.append(eval(df_name))

    for year in years_splitted:
        if year in years:
            train_dfs.append(eval(f'df_{pollutant}_{year}_train_1D'))
            val_dfs.append(eval(f'df_{pollutant}_{year}_val_1D'))
            test_dfs.append(eval(f'df_{pollutant}_{year}_test_1D'))

    if year_final in years:
        val_dfs.append(eval(f'df_{pollutant}_{year_final}_val_1D'))
        test_dfs.append(eval(f'df_{pollutant}_{year_final}_test_1D'))

    print_split_ratios(train_dfs, val_dfs, test_dfs, pollutant)
del train_dfs, val_dfs, test_dfs

# %% [markdown]
# ### **Normalisation**
# 
# Warning: make sure the pollutants are parsed in, in order of NO2, O3, PM2.5, PM10

# %% [markdown]
# #### Finding Min Max for pollutants and weather data

# %%
# Normalise each component separately, using the training data extremes
years_training = years_unsplitted + years_splitted
import numpy as np
# Calculate min and max parameters for pollutants
pollutant_min_max = {}
for pollutant in pollutants:
    dfs = [eval(f'df_{pollutant}_{year}_train_1D') for year in years_training]
    min_train, max_train = calc_combined_min_max_params(dfs)
    pollutant_min_max[f'{pollutant}_min_train'] = min_train
    pollutant_min_max[f'{pollutant}_max_train'] = max_train

# Calculate min and max parameters for meteorological data
weather_min_max = {}
for meteo in selected_weather:
    dfs = [eval(f'df_{meteo}_{year}_train') for year in years_training]
    min_train, max_train = calc_combined_min_max_params(dfs)
    weather_min_max[f'{meteo}_min_train'] = min_train
    weather_min_max[f'{meteo}_max_train'] = max_train

# Print pollutant extremes
print()
df_minmax = print_pollutant_extremes(
    [pollutant_min_max[f'{pollutant}_min_train'] for pollutant in pollutants] +
    [pollutant_min_max[f'{pollutant}_max_train'] for pollutant in pollutants],
    pollutants
)
print()

# Export min and max parameters
export_minmax(df_minmax, 'pollutants_minmax_allyears')

# %% [markdown]
# ### Normalising according to min max

# %%
# Normalise each component separately, using the training data extremes

print("------Normalising the pollutants------")
for pollutant in pollutants:
    print(f"Normalising {pollutant}...")
    min_train = pollutant_min_max[f'{pollutant}_min_train']
    max_train = pollutant_min_max[f'{pollutant}_max_train']
    for year in years:
        for split in ['train_1D', 'val_1D', 'test_1D']:
            df_name = f'df_{pollutant}_{year}_{split}'
            norm_df_name = f'df_{pollutant}_{year}_{split}_norm'
            try:
                df = eval(df_name)
                norm_df = normalise_linear(df, min_train, max_train)
                globals()[norm_df_name] = norm_df
                print(f"Normalised {df_name}")
            except NameError:
                print(f"Warning: {df_name} does not exist.")

print("-------Normalising the weather data-------")
for meteo in selected_weather:
    print(f"Normalising {meteo}...")
    min_train = weather_min_max[f'{meteo}_min_train']
    max_train = weather_min_max[f'{meteo}_max_train']
    for year in years:
        for split in ['train', 'val', 'test']:
            df_name = f'df_{meteo}_{year}_{split}'
            norm_df_name = f'df_{meteo}_{year}_{split}_norm'
            try:
                df = eval(df_name)
                norm_df = normalise_linear(df, min_train, max_train)
                globals()[norm_df_name] = norm_df
                print(f"Normalised {df_name}")
            except NameError:
                print(f"Warning: {df_name} does not exist.")

# %%
if LOG:
    # Assert range only for training frames, validation and test
    # frames can, very theoretically, have unlimited values
    for pollutant in pollutants:
        train_dfs = []
        for year in years:
            df_name = f'df_{pollutant}_{year}_train_norm_1D'
            try:
                df = eval(df_name)
                train_dfs.append(df)
            except NameError:
                pass
        assert_range(train_dfs, 0, 1, f'Normalisation of {pollutant} data')

    for meteo in selected_weather:
        train_dfs = []
        for year in years:

            df_name = f'df_{meteo}_{year}_train_norm'
            try:
                df = eval(df_name)
                train_dfs.append(df)
            except NameError:
                pass
        assert_range(train_dfs, 0, 1, f'Normalisation of {meteo} data')

    print('(6/8): Normalisation successful')

# %%
df_FX_2021_val_norm

# %%
df_NO2_2021_train_1D_norm

# %% [markdown]
# ### **Create big combined normalised dataframe**

# %% [markdown]
# #### Creating all the input dataframes with data from Tuindorp

# %%
# Create a dictionary to store the frames for each year and split
frames_dict_u = {}

splits = ['train', 'val', 'test']

# Loop through each year and split to create the combined normalized dataframes
for year in years:
    frames_dict_u[year] = {}
    for split in splits:
        frames_dict_u[year][split] = {}
        for pollutant in pollutants:
            df_name = f'df_{pollutant}_{year}_{split}_1D_norm'
            try:
                df = eval(df_name)
                frames_dict_u[year][split][pollutant] = df.loc[:, [TUINDORP]]
                print(f"frame dict added {pollutant}_{split}_{year}")
            except NameError:
                print(f"Warning: {df_name} does not exist.")
        for meteo in selected_weather:
            df_name = f'df_{meteo}_{year}_{split}_norm'
            try:
                df = eval(df_name)
                frames_dict_u[year][split][meteo] = df
            except NameError:
                print(f"Warning: {df_name} does not exist.")

# Print all the names of frames in frames_dict_u
print("Frames in frames_dict_u:")
for year in frames_dict_u:
    for split in frames_dict_u[year]:
        for key in frames_dict_u[year][split]:
            print(f"{key}_{split}_{year}")

# Example of accessing specific frames
print(frames_dict_u[2017]['train']['NO2'])
print("---------------")
print(frames_dict_u[2021]['val']['T'])

# %%
frames_dict_u[2017]['train']['DD']

# %% [markdown]
# #### Creating all the input dataframes with data from Breukelen

# %%
# Create a dictionary to store the frames for each year and split
frames_dict_y = {}

# Loop through each year and split to create the combined normalized dataframes for Breukelen
for year in years:
    frames_dict_y[year] = {}
    for split in splits:
        frames_dict_y[year][split] = {}
        for pollutant in pollutants:
            df_name = f'df_{pollutant}_{year}_{split}_1D_norm'
            try:
                df = eval(df_name)
                frames_dict_y[year][split][pollutant] = df.loc[:, [BREUKELEN]]
                print(f"frame dict added {pollutant}_{split}_{year} for Breukelen")
            except NameError:
                print(f"Warning: {df_name} does not exist.")
# Print all the names of frames in frames_dict_y
print("Frames in frames_dict_y:")
for year in frames_dict_y:
    for split in frames_dict_y[year]:
        for key in frames_dict_y[year][split]:
            print(f"{key}_{split}_{year}")

# Example of accessing specific frames
print(frames_dict_y[2017]['train']['NO2'])
print(frames_dict_y[2021]['val']['NO2'])

# %%
print(frames_dict_u[2022]['val']['DD'])

#%%
years


# %%
input_keys = pollutants + selected_weather
output_keys = pollutants

# %%
input_keys, output_keys

# %% [markdown]
# ### Concatenating and putting the keys in alpha order

# %%
# Create dictionaries to store the concatenated dataframes for each year and split
concat_frames_dict_u = {}
concat_frames_dict_y = {}

# Loop through each year and split to concatenate the frames horizontally for Tuindorp
for year in years:
    concat_frames_dict_u[year] = {}
    for split in splits:
        frames = []
        for key in input_keys:
            try:
                frames.append(frames_dict_u[year][split][key])
            except KeyError:
                print(f"Warning: {key}_{split}_{year} does not exist.")
        
        print(year, split, "FRAMES", frames)
        concat_frames_dict_u[year][split] = concat_frames_horizontally(frames, input_keys)
        print(f"Concatenated Input frames for {split} {year} for Tuindorp")

# # Loop through each year and split to concatenate the frames horizontally for Breukelen
# for year in years:
#     concat_frames_dict_y[year] = {}
#     for split in splits:
#         frames = []
#         for key in output_keys:
#             try:
#                 frames.append(frames_dict_y[year][split][key])
#             except KeyError:
#                 print(f"Warning: {key}_{split}_{year} does not exist.")
#         concat_frames_dict_y[year][split] = concat_frames_horizontally(frames, output_keys)
#         print(f"Concatenated Output frames for {split} {year} for Breukelen")

# Example of accessing specific concatenated frames
print(concat_frames_dict_u[2017]['train'])
print(concat_frames_dict_y[2017]['train'])

# %%
# At last, a final check before exporting

if LOG:
    if years == [2017, 2018, 2020, 2021, 2022, 2023]:
        # Define the years for unsplitted and splitted data
        years_unsplitted = [2017, 2018, 2020]
        years_splitted = [2021, 2022]

        # First, check if u-dataframes of unsplitted years have the same shape
        u_data_unsplitted = [concat_frames_dict_u[year]['train'] for year in years_unsplitted]
        assert_equal_shape(u_data_unsplitted, True, True, 'Shape of u-dataframes of unsplitted years')

        # Second, check if y-dataframes of unsplitted years have the same shape
        y_data_unsplitted = [concat_frames_dict_y[year]['train'] for year in years_unsplitted]
        assert_equal_shape(y_data_unsplitted, True, True, 'Shape of y-dataframes of unsplitted years')

        # Third, check if validation/test u-dataframes of splitted years have the same shape
        u_data_splitted = [concat_frames_dict_u[year][split] for year in years_splitted for split in ['val', 'test']]
        assert_equal_shape(u_data_splitted, True, True, 'Shape of u-dataframes of splitted years')

        # Fourth, check if validation/test y-dataframes of splitted years have the same shape
        y_data_splitted = [concat_frames_dict_y[year][split] for year in years_splitted for split in ['val', 'test']]
        assert_equal_shape(y_data_splitted, True, True, 'Shape of y-dataframes of splitted years')

        print('(7/8): All data concatenations successful for years 2017, 2018, 2020, 2021, 2022')
    elif len(years) == 1:
        # print the shape of the concat frames for that year, for all train, val and test
        for split in splits:
            print(f"Shape of concatenated frames for {split} {years[0]}:")
            print(concat_frames_dict_u[years[0]][split].shape)
            print(concat_frames_dict_y[years[0]][split].shape)
    else:
        print("Error: Data concatenation not defined for the specific years")


# %%
# Save the dataframes to data_combined/ folder. The windowing will be performed
# by a PyTorch Dataset class in the model scripts.

# Loop through each year and split to save the concatenated dataframes for Tuindorp
for year in years:
    for split in splits:
        try:
            df = concat_frames_dict_u[year][split]
            df.to_csv(f"../data/data_combined/{split}_{year}_combined_u.csv",
                      index=True, sep=';', decimal='.', encoding='utf-8')
            print(f"Saved {split}_{year}_combined_u.csv")
        except KeyError:
            print(f"Warning: {split}_{year}_combined_u does not exist.")

# Loop through each year and split to save the concatenated dataframes for Breukelen
for year in years:
    for split in splits:
        try:
            df = concat_frames_dict_y[year][split]
            df.to_csv(f"../data/data_combined/{split}_{year}_combined_y.csv",
                      index=True, sep=';', decimal='.', encoding='utf-8')
            print(f"Saved {split}_{year}_combined_y.csv")
        except KeyError:
            print(f"Warning: {split}_{year}_combined_y does not exist.")

# %%
if LOG:
    print('(8/8): Data exported successfully')
    print('Data preparation finished')


