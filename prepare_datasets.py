import nilmtk
from nilmtk import DataSet, MeterGroup
from nilmtk.dataset_converters.refit.convert_refit import convert_refit
from nilmtk.utils import print_dict
import os
from File_Management.path_and_file_management_Func import try_to_find_file
import pandas as pd
import numpy as np
from Ploting.fast_plot_Func import series, hist, scatter
import matplotlib.pyplot as plt
import csv
from dateutil import tz

DATASET_ROOT_DIRECTORY = r'E:\OneDrive_Extra\Database\Load_Disaggregation'


def load_ampds2_weather():
    _path = os.path.join(DATASET_ROOT_DIRECTORY,
                         r'AMPds2/MERRA2/')
    if not os.path.exists(_path + "weather.pkl"):
        read_results = pd.DataFrame()
        for file in ('2012.csv', '2013.csv', '2014.csv'):
            reading = pd.read_csv(_path + file,
                                  sep=',',
                                  skiprows=3)
            read_results = pd.concat((read_results,
                                      pd.DataFrame(index=pd.DatetimeIndex(reading['local_time']),
                                                   data={'temperature': reading['temperature'].values,
                                                         'solar irradiation': reading['radiation_surface'].values,
                                                         'precipitation': reading['precipitation'].values,
                                                         'air density': reading['air_density'].values})
                                      ))
        read_results = read_results.loc[~read_results.index.duplicated()]
        read_results.to_pickle(_path + "weather.pkl")
    else:
        read_results = pd.read_pickle(_path + "weather.pkl")  # type: pd.DataFrame

    return read_results


def convert_refit_to_h5():
    h5_file_ = os.path.join(DATASET_ROOT_DIRECTORY,
                            r'REFIT\Cleaned\CLEAN_REFIT_081116\Refit.h5')
    if try_to_find_file(h5_file_):
        return
    convert_refit(input_path=os.path.join(DATASET_ROOT_DIRECTORY,
                                          r'REFIT\Cleaned\CLEAN_REFIT_081116'),
                  output_filename=h5_file_,
                  format='HDF')


def load_datasets():
    _ampds2_dataset = DataSet(os.path.join(DATASET_ROOT_DIRECTORY,
                                           r'AMPds2\dataverse_files\AMPds2.h5'))
    _refit_dataset = DataSet(os.path.join(DATASET_ROOT_DIRECTORY,
                                          r'REFIT\Cleaned\CLEAN_REFIT_081116\Refit.h5'))

    convert_refit_to_h5()
    _uk_dale_dataset = DataSet(os.path.join(DATASET_ROOT_DIRECTORY,
                                            r'UK-DALE\ukdale.h5\ukdale.h5'))
    return _ampds2_dataset, _refit_dataset, _uk_dale_dataset


if __name__ == '__main__':
    ampds2_dataset, refit_dataset, uk_dale_dataset = load_datasets()
    # load_ampds2_weather()
