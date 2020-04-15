import nilmtk
from nilmtk import DataSet, MeterGroup
from nilmtk.dataset_converters.refit.convert_refit import convert_refit
from nilmtk.utils import print_dict
import os
from File_Management.path_and_file_management_Func import try_to_find_file
from File_Management.load_save_Func import load_exist_pkl_file_otherwise_run_and_save, load_pkl_file, save_pkl_file
import pandas as pd
import numpy as np
from numpy import ndarray
from Ploting.fast_plot_Func import series, hist, scatter, time_series
import matplotlib.pyplot as plt
import csv
from dateutil import tz
from typing import Tuple
from FFT_Class import FFTProcessor
import copy
from torch.utils.data import Dataset as TorchDataSet, DataLoader as TorchDataLoader
import torch
from pathlib import Path
from Time_Processing.datetime_utils import DatetimeOnehotEncoder

DATASET_ROOT_DIRECTORY = r'E:\OneDrive_Extra\Database\Load_Disaggregation'


def load_ampds2_weather():
    """
    载入ampds2的天气数据
    :return:
    """
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
    """
    将refit的csv全部转成一个h5
    :return:
    """
    h5_file_ = os.path.join(DATASET_ROOT_DIRECTORY,
                            r'REFIT\Cleaned\CLEAN_REFIT_081116\Refit.h5')
    if try_to_find_file(h5_file_):
        return
    convert_refit(input_path=os.path.join(DATASET_ROOT_DIRECTORY,
                                          r'REFIT\Cleaned\CLEAN_REFIT_081116'),
                  output_filename=h5_file_,
                  format='HDF')


def load_datasets():
    """
    载入全部三个数据集，h5格式
    :return: 三个数据集的tuple
    """
    _ampds2_dataset = DataSet(os.path.join(DATASET_ROOT_DIRECTORY,
                                           r'AMPds2\dataverse_files\AMPds2.h5'))
    _refit_dataset = DataSet(os.path.join(DATASET_ROOT_DIRECTORY,
                                          r'REFIT\Cleaned\CLEAN_REFIT_081116\Refit.h5'))

    convert_refit_to_h5()
    _uk_dale_dataset = DataSet(os.path.join(DATASET_ROOT_DIRECTORY,
                                            r'UK-DALE\ukdale.h5\ukdale.h5'))
    return _ampds2_dataset, _refit_dataset, _uk_dale_dataset


def get_training_set_test_set_for_ampds2_dataset() -> Tuple[MeterGroup, MeterGroup]:
    """
    从ampds2_dataset中分离出training set和test set，
    只考虑building 1
    2012-4-1 0:00到2013-4-1 0:00 是training set
    2013-4-1 0:00到2014-4-1 0:00 是test set
    :return:
    """
    training_set, _, _ = load_datasets()
    test_set, _, _ = load_datasets()
    training_set.set_window(end='2013-4-1')
    training_set = training_set.buildings[1].elec
    test_set.set_window(start='2013-4-1')
    test_set = test_set.buildings[1].elec
    return training_set, test_set


class NILMTorchDataset(TorchDataSet):
    """
    专门针对PyTorch的模型的Dataset
    """
    __slots__ = ('data', 'transformed_data', 'sequence_length', 'over_lapping', 'country')

    def __init__(self, data: pd.DataFrame,
                 *, country,
                 sequence_length: int,
                 over_lapping: bool = False,
                 transform_args_file_path: Path,
                 ):
        """
        :param data 所有数据，包括x和y，形如下面的一个pd.DataFrame
                           temperature  solar irradiation   precipitation   air density   mains_var   appliance_var
        time_stamp_index
        .
        .
        .
        TODO: solar/moon
        transform_args: 形如下面的一个pd.DataFrame
                  temperature  solar irradiation   precipitation   air density   mains_var   appliance_var

        minimum
        maximum
        :param transform_args_file_path
        """
        self.data = data  # type: pd.DataFrame
        self.sequence_length = sequence_length  # type: int
        self.over_lapping = over_lapping
        self.country = country
        # 最后进行transform
        self.transformed_data = self._transform(transform_args_file_path)  # type: Tuple[torch.tensor, torch.tensor]

    def __len__(self):
        if self.over_lapping:
            return self.data.__len__() - self.sequence_length + 1
        else:
            return int(self.data.__len__() / self.sequence_length)

    def __getitem__(self, index: int):
        # 决定索引的位置
        if self.over_lapping:
            index_slice = slice(index, index + self.sequence_length)
        else:
            index_slice = slice(index * self.sequence_length, (index + 1) * self.sequence_length)
        data_x = self.transformed_data[0][index_slice]  # type: torch.tensor
        data_y = self.transformed_data[1][index_slice]  # type: torch.tensor
        return data_x, data_y

    def _transform(self, transform_args_file_path) -> Tuple[torch.tensor, torch.tensor]:
        # 时间变量
        datetime_onehot_encoder = DatetimeOnehotEncoder()
        time_var_transformed = datetime_onehot_encoder(self.data.index,
                                                       country=self.country)
        # 其它
        transform_args = self._get_transform_args(transform_args_file_path)
        other_var_transformed = pd.DataFrame(columns=transform_args.columns)
        for this_col in transform_args.columns:
            other_var_transformed.loc[:, this_col] = (self.data.loc[:, this_col] -
                                                      transform_args.loc['minimum', this_col]) / (
                                                             transform_args.loc['maximum', this_col] -
                                                             transform_args.loc['minimum', this_col])
        transformed_x_y = pd.concat(
            (time_var_transformed, other_var_transformed.reset_index().drop('TIMESTAMP', axis=1)), axis=1)
        data_x = torch.tensor(transformed_x_y.iloc[:, :-1].values,
                              device='cuda:0',
                              dtype=torch.float)
        data_y = torch.tensor(transformed_x_y.iloc[:, [-1]].values,
                              device='cuda:0',
                              dtype=torch.float)
        return data_x, data_y

    def _get_transform_args(self, file_path: Path) -> pd.DataFrame:
        """
        根据self.weather_var, self.main_var, self.appliance_var，载入 OR 计算并保存min-max变换需要的参数
        """
        if not file_path:
            raise Exception('Should specify file_path')

        @load_exist_pkl_file_otherwise_run_and_save(file_path)
        def func() -> pd.DataFrame:
            transform_args = pd.DataFrame(index=('minimum', 'maximum'),
                                          columns=self.data.columns)
            for i in transform_args.columns:
                transform_args[i] = (np.nanmin(self.data[i].values), np.nanmax(self.data[i].values))
            return transform_args

        return func


if __name__ == '__main__':
    ampds2_dataset, refit_dataset, uk_dale_dataset = load_datasets()
