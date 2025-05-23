# src/modelling/__init__.py

# __init__py for the modelling package

__version__ = '0.0.0' # MAJOR.MINOR.PATCH versioning
__author__ = 'valentijn7' # GitHub username

print("\nRunning __init__.py for data pipeline...")

from .extract import import_csv
from .extract import get_dataframes
from .train import train
from .train import train_hierarchical
from .TimeSeriesDataset import TimeSeriesDataset
from .PrintManager import PrintManager
# from .test import test_hierarchical
# from .test import test_hierarchical_separately

# from .plots import get_pred_and_gt
# from .plots import get_index
# from .plots import choose_plot_component_values
# from .plots import plot_pred_vs_gt
# from .plots import plot_losses
# from .plots import plot_flexibility
# from .plots import plot_losses_normalised

from modelling.initialise import *
from .GRU import GRU
from .HGRU import HGRU

print("Modelling package initialized\n")