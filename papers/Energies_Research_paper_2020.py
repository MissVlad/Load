from Ploting.fast_plot_Func import *
import numpy as np
import pandas as pd
from numpy import ndarray
from prepare_datasets import ampds2_dataset_full_df
from TimeSeries_Class import TimeSeries
from FFT_Class import STFTProcessor
from Correlation_Modeling.utils import CorrelationAnalyser


def calculate_thermally_correlated_components():
    ampds2_df = ampds2_dataset_full_df(60)
    tt = 1


if __name__ == '__main__':
    calculate_thermally_correlated_components()
