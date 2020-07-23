import copy
import os
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from nilmtk import DataSet, MeterGroup
from nilmtk.dataset_converters.refit.convert_refit import convert_refit
from scipy.io import loadmat
from torch.utils.data import Dataset as TorchDataSet

from File_Management.load_save_Func import load_exist_pkl_file_otherwise_run_and_save
from File_Management.path_and_file_management_Func import try_to_find_file
from Ploting.fast_plot_Func import *
from TimeSeries_Class import TimeSeries, UnivariateTimeSeries, merge_two_time_series_df
from Time_Processing.datetime_utils import DatetimeOnehotEncoder
from project_path_Var import project_path_
import copy
from Writting import docx_document_template_to_collect_figures
from docx.shared import Cm, Pt
from docx.enum.text import WD_BREAK
from dateutil import tz
from pandas import DataFrame
from workalendar.america import Canada
from Time_Processing.datetime_utils import DatetimeOnehotEncoder
from pyorbital.moon_phase import moon_phase

DATASET_ROOT_DIRECTORY = r'E:\OneDrive_Extra\Database\Load_Disaggregation'
LOW_CARBON_LONDON_ROOT = r'C:\Users\SoapClancy\OneDrive\PhD\01-PhDProject\Database\UK Power Networks'


def load_low_carbon_london_heat(this_no, df_name: str):
    """
    Depreciated codes. Only for very specific task. Do not be fooled.
    """
    # available_customer_no = (1, 2, 3, 4, 5, 7, 9)
    heat_data_root = Path(
        LOW_CARBON_LONDON_ROOT) / r'Low Carbon London Heat Pump Load Profiles\Data supply 1\Data supply 1\Heat Profiles'
    pq_data_root = Path(
        LOW_CARBON_LONDON_ROOT) / r'Low Carbon London Heat Pump Load Profiles\Data supply 1\Data supply 1\Power Quality'
    figure_buffer_list = []
    names_list = []

    # for this_no in available_customer_no:
    this_figure_buffer_list_elec = []
    this_figure_buffer_list_heat = []
    this_figure_buffer_list_elec_and_heat = []

    # read heat pump
    this_heat_df = pd.read_csv(heat_data_root / f'S1_Customer_L_{this_no}.csv',
                               sep=',', index_col='timestamp')
    this_heat_df = TimeSeries(
        this_heat_df[['external_temperature', 'zone_1_temperature', 'heat_pump_energy_consumption']],
        index=pd.DatetimeIndex(this_heat_df.index)
    )
    this_heat_df['heat_pump_energy_consumption'].iloc[1:] = np.diff(
        this_heat_df['heat_pump_energy_consumption'].values
    )
    this_heat_df.rename({'heat_pump_energy_consumption': 'heat_pump_energy_consumption diff'}, axis=1, inplace=True)
    this_heat_df = this_heat_df.iloc[1:]
    # read electric
    this_pq_df_index = pd.read_csv(pq_data_root / f'S1_Customer_PQ_{this_no}.csv', nrows=2, header=None)
    this_pq_df_index = pd.date_range(start=pd.to_datetime(this_pq_df_index.iloc[0, 1], format="%d/%m/%Y %H:%M"),
                                     end=pd.to_datetime(this_pq_df_index.iloc[1, 1], format="%d/%m/%Y %H:%M"),
                                     freq='T')
    this_pq_df = pd.read_csv(pq_data_root / f'S1_Customer_PQ_{this_no}.csv',
                             sep=',', skiprows=5,
                             engine='python', usecols=['kW of Vln * Il - Mean [kW]'])
    # Apply Sasa's method. He said to take the absolute values for negative power, despite they may take 100 %
    this_pq_df[this_pq_df < 0] = np.abs(this_pq_df[this_pq_df < 0])
    try:
        this_pq_df.index = this_pq_df_index
        this_pq_df_aggregate = this_pq_df.resample('30T').mean()
    except ValueError:
        names_list.append(' ')
        figure_buffer_list.append([])
        return

    # Try align heating and elec
    this_pq_df_and_heat_df = merge_two_time_series_df(this_pq_df,
                                                      this_heat_df,
                                                      interpolate_method='nearest')
    this_pq_df_and_heat_df = this_pq_df_and_heat_df.resample('30T').mean()

    # %% Plot electric all
    def plot_for_original_elec(x, y):
        return time_series(x=x, y=y, color='b',
                           x_label='Date time (original, resolution: 1 minute)', y_label='Active power [kW]',
                           save_to_buffer=True)

    def plot_for_aggregate_elec(x, y):
        return time_series(x=x, y=y, color='b',
                           x_label='Date time (aggregated, resolution: 30 minute)', y_label='Active power [kW]',
                           save_to_buffer=True)

    def plot_for_heat(x, y):
        return time_series(x=x, y=y, color='r', linestyle='--',
                           x_label='Date time (original, resolution: 30 minute)',
                           y_label='heat_pump_energy_consumption\n'
                                   'difference [unknown unit]',
                           save_to_buffer=True)

    def plot_for_aggregate_elec_and_heat(x_elec, y_elec, x_heat, y_heat):
        buffer_1 = plot_for_aggregate_elec(x_elec, y_elec)
        buffer_2 = plot_for_heat(x_heat, y_heat)
        return buffer_1, buffer_2

    this_figure_buffer_list_elec.append(plot_for_original_elec(this_pq_df.index, this_pq_df.iloc[:, 0].values))
    this_figure_buffer_list_elec.append(
        plot_for_aggregate_elec(this_pq_df_aggregate.index, this_pq_df_aggregate.iloc[:, 0].values)
    )

    this_figure_buffer_list_heat.append(plot_for_heat(this_heat_df.index, this_heat_df.iloc[:, 0].values))

    this_figure_buffer_list_elec_and_heat.extend(plot_for_aggregate_elec_and_heat(
        x_elec=this_pq_df_and_heat_df.index,
        y_elec=this_pq_df_and_heat_df.iloc[:, 0],
        x_heat=this_pq_df_and_heat_df.index,
        y_heat=this_pq_df_and_heat_df.iloc[:, -1]
    ))
    iter_dict = {'electric': this_pq_df,
                 'heat': this_heat_df,
                 'electric and heat': this_pq_df_and_heat_df}
    considered_df = iter_dict[df_name]
    figure_buffer_list_specific = copy.deepcopy(figure_buffer_list)
    this_figure_buffer_list_elec_specific = copy.deepcopy(this_figure_buffer_list_elec)
    date_range = pd.date_range(considered_df.first_valid_index().date(),
                               considered_df.last_valid_index().date(),
                               freq='D')
    for i in range(date_range.__len__()):
        if i == date_range.__len__() - 1:
            if considered_df.loc[date_range[i]:].size < 40:
                break
            this_df_i = considered_df.loc[date_range[i]:]
            if df_name == 'electric':
                this_pq_df_aggregate_i = this_pq_df_aggregate.loc[date_range[i]:]

        else:
            this_df_i = considered_df.loc[date_range[i]:date_range[i + 1]].iloc[:-1]
            if df_name == 'electric':
                this_pq_df_aggregate_i = this_pq_df_aggregate.loc[date_range[i]:date_range[i + 1]].iloc[:-1]
        if df_name == 'electric':
            this_figure_buffer_list_elec_specific.append(
                plot_for_original_elec(this_df_i.index, this_df_i.iloc[:, 0].values))
            this_figure_buffer_list_elec_specific.append(
                plot_for_aggregate_elec(this_pq_df_aggregate_i.index, this_pq_df_aggregate_i.iloc[:, 0].values)
            )
            figure_buffer_list_specific.append(this_figure_buffer_list_elec_specific)

        elif df_name == 'heat':
            this_figure_buffer_list_heat.append(
                plot_for_heat(this_df_i.index, this_df_i.iloc[:, 0].values))
            figure_buffer_list_specific.append(this_figure_buffer_list_heat)

        else:
            this_figure_buffer_list_elec_and_heat.extend(
                plot_for_aggregate_elec_and_heat(this_df_i.index, this_df_i.iloc[:, 0].values,
                                                 this_df_i.index, this_df_i.iloc[:, -1].values)
            )
            figure_buffer_list_specific.append(this_figure_buffer_list_elec_and_heat)

    names_list.append(f'S1_Customer_L_{this_no}.csv')
    # Write
    document = docx_document_template_to_collect_figures()
    for i in range(names_list.__len__()):
        document.add_heading(names_list[i], level=1)
        p = document.add_paragraph()
        p.add_run().add_break()
        for j in range(figure_buffer_list_specific[i].__len__()):
            p = document.add_paragraph()
            p.add_run().add_picture(figure_buffer_list_specific[i][j], width=Cm(14))
            # if (j % 2) == 1:
            #     p.add_run().add_break()
        document.add_page_break()
    document.save(f'.\\London {df_name} data household_{this_no}.docx')


def load_ampds2_weather():
    """
    载入ampds2的天气数据
    :return:
    """
    _path = os.path.join(DATASET_ROOT_DIRECTORY,
                         r'AMPds2/dataverse_files/Climate_HourlyWeather.csv')
    reading = pd.read_csv(_path,
                          sep=',')
    read_results = reading[['Temp (C)', 'Rel Hum (%)', 'Stn Press (kPa)']]
    read_results.index = pd.DatetimeIndex(pd.to_datetime(reading.iloc[:, 0],
                                                         format='%Y-%m-%d %H:%M'))
    # _path = os.path.join(DATASET_ROOT_DIRECTORY,
    #                      r'AMPds2/MERRA2/')
    # # America / Vancouver
    # if not os.path.exists(_path + "weather.pkl"):
    #     read_results = pd.DataFrame()
    #     for file in ('2012.csv', '2013.csv', '2014.csv'):
    #         reading = pd.read_csv(_path + file,
    #                               sep=',',
    #                               skiprows=3)
    #         read_results = pd.concat((read_results,
    #                                   pd.DataFrame(index=pd.DatetimeIndex(reading['local_time']),
    #                                                data={'temperature': reading['temperature'].values,
    #                                                      'solar irradiation': reading['radiation_surface'].values,
    #                                                      'precipitation': reading['precipitation'].values,
    #                                                      'air density': reading['air_density'].values})
    #                                   ))
    #     read_results = read_results.loc[~read_results.index.duplicated()]
    #     read_results.to_pickle(_path + "weather.pkl")
    # else:
    #     read_results = pd.read_pickle(_path + "weather.pkl")  # type: pd.DataFrame
    #
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


def get_training_set_and_test_set_for_ampds2_dataset() -> Tuple[MeterGroup, MeterGroup, MeterGroup]:
    """
    从ampds2_dataset中分离出training set和test set，
    只考虑building 1
    2012-4-1 0:00到2013-4-1 0:00 是training set
    2013-4-1 0:00到2014-4-1 0:00 是test set
    :return:
    """
    training_set, _, _ = load_datasets()
    test_set, _, _ = load_datasets()
    whole_set, _, _ = load_datasets()

    training_set.set_window(end='2013-4-1')
    training_set = training_set.buildings[1].elec

    test_set.set_window(start='2013-4-1')
    test_set = test_set.buildings[1].elec

    whole_set = whole_set.buildings[1].elec
    return training_set, test_set, whole_set


def ampds2_dataset_full_df(resolution: int) -> pd.DataFrame:
    """
    ampds2_dataset的heat，main，和对应的气象，和对应的时间
    """
    _, _, ampds2 = get_training_set_and_test_set_for_ampds2_dataset()
    heating_df = next(ampds2.select_using_appliances(
        original_name='HPE').meters[0].load(ac_type='active', sample_period=resolution))
    heating_df = heating_df.droplevel('physical_quantity', axis=1)  # type: DataFrame
    heating_df.rename(columns={'active': 'HPE'}, inplace=True)

    mains_df = next(ampds2.mains().load(
        ac_type='active', sample_period=resolution)).droplevel('physical_quantity', axis=1)  # type: DataFrame
    mains_df.rename(columns={mains_df.columns[0]: 'Mains'}, inplace=True)

    ampds2_weather_df = load_ampds2_weather()
    mains_weather_df_merged = merge_two_time_series_df(mains_df, ampds2_weather_df)
    mains_weather_df_merged = mains_weather_df_merged.reindex(columns=mains_weather_df_merged.columns[1:].append(
        mains_weather_df_merged.columns[slice(1)]))

    full_data_df = pd.concat((mains_weather_df_merged, heating_df), axis=1)

    full_data_df['year'] = full_data_df.index.year
    full_data_df['month'] = full_data_df.index.month
    full_data_df['day'] = full_data_df.index.day
    full_data_df['dayofweek'] = full_data_df.index.dayofweek + 1
    full_data_df['hour'] = full_data_df.index.hour
    full_data_df['minute'] = full_data_df.index.minute
    full_data_df['moon_phase'] = moon_phase(full_data_df.index.to_numpy())
    date_time_one_hot_encoder = DatetimeOnehotEncoder(to_encoding_args=('holiday', 'summer_time'))
    time_var_transformed = date_time_one_hot_encoder(full_data_df.index,
                                                     country=Canada())
    full_data_df['holiday'] = np.array(time_var_transformed.iloc[:, 0] == 0, dtype=int)
    full_data_df['summer_time'] = np.array(time_var_transformed.iloc[:, 2] == 1, dtype=int)
    return full_data_df


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
        TODO: solar
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

    def _transform(self, transform_args_file_path, mode='NILM') -> Tuple[torch.tensor, torch.tensor]:
        # 时间变量
        # datetime_onehot_encoder = DatetimeOnehotEncoder(to_encoding_args=('month',
        #                                                                   'weekday',
        #                                                                   'holiday',
        #                                                                   'hour',
        #                                                                   'minute'))
        datetime_onehot_encoder = DatetimeOnehotEncoder(to_encoding_args=('month',
                                                                          'weekday',
                                                                          'holiday',
                                                                          'summer_time'))
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
        if mode == 'NILM':
            data_y = torch.tensor(transformed_x_y.iloc[:, [-1]].values,
                                  device='cuda:0',
                                  dtype=torch.float)
        elif mode == 'forecast':
            data_y = torch.tensor(transformed_x_y.iloc[:, -2:].values,
                                  device='cuda:0',
                                  dtype=torch.float)
        else:
            raise Exception
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

        return func()


class NILMTorchDatasetForecast(NILMTorchDataset):

    def __len__(self):
        if self.over_lapping:
            raise NotImplementedError
        else:
            return int(self.data.__len__() / self.sequence_length) - 1

    def __getitem__(self, index: int):
        # 决定索引的位置
        if self.over_lapping:
            raise NotImplementedError
        else:
            index_slice_x = slice(index * self.sequence_length, (index + 1) * self.sequence_length)
            index_slice_y = slice((index + 1) * self.sequence_length, (index + 2) * self.sequence_length)
        data_x = self.transformed_data[0][index_slice_x]  # type: torch.tensor
        data_y = self.transformed_data[1][index_slice_y]  # type: torch.tensor
        return data_x, data_y

    def _transform(self, transform_args_file_path, mode='forecast') -> Tuple[torch.tensor, torch.tensor]:
        return super(NILMTorchDatasetForecast, self)._transform(transform_args_file_path, mode='forecast')


class ScotlandDataset(metaclass=ABCMeta):
    data_path_root = project_path_ / r'Data\Raw\Scotland selected'
    __slots__ = ('name', 'matlab_mat_file_folder', 'dataset')

    def __init__(self, name: str):
        """
        name可选: ['Drum', 'John', 'MAYB', 'STHA']
        """
        if name not in ['Drum', 'John', 'MAYB', 'STHA']:
            raise Exception('Unknown bus')
        self.name = name
        self.matlab_mat_file_folder = self.data_path_root / self.name
        self.dataset = self.load_raw_data()  # type: pd.DataFrame

    def __str__(self):
        return f'Scotland dataset: {self.name}, from {self.dataset.index[0]} to {self.dataset.index[-1]}'

    def load_raw_data(self) -> pd.DataFrame:
        """
        重复利用以前的结果。载入matlab那些原始数据
        """
        time_index = self._get_time_index()
        holiday_ndarray = self._get_holiday_ndarray()
        bst_ndarray = self._get_british_summer_time()
        raw_data = pd.DataFrame(data={'holiday': holiday_ndarray,
                                      'BST': bst_ndarray,
                                      'active power': self.load_active_power_mat(),
                                      'temperature': self.get_temperature()},
                                index=time_index)
        return raw_data

    def set_weekends_and_holiday_to_zeros(self, inplace=False) -> Tuple[pd.DataFrame, ndarray]:
        """
        把节假日或者周末的数据置为0
        :return 置0后的pd.DataFrame和对应的mask
        """
        mask = np.any((self.dataset.index.weekday == 5,  # 周六，因为Monday=0, Sunday=6.
                       self.dataset.index.weekday == 6,  # 周天，因为Monday=0, Sunday=6.
                       self.dataset['holiday'] == 1), axis=0)
        if inplace:
            self.dataset.loc[mask, 'active power'] = 0
            return self.dataset, mask
        else:
            dataset_copy = copy.deepcopy(self.dataset)
            dataset_copy.loc[mask, 'active power'] = 0
            return dataset_copy, mask

    def _get_time_index(self) -> pd.DatetimeIndex:
        """
        基于matlab那些原始数据得到python的pd.DatetimeIndex
        """
        ts_matrix = self.load_ts_mat()
        datetime_tuple = [datetime.datetime(year=int(x[0]),
                                            month=int(x[1]),
                                            day=int(x[2]),
                                            hour=int(x[3]),
                                            minute=int(60 * (x[3] - int(x[3])))) for x in ts_matrix]
        time_index = pd.DatetimeIndex(datetime_tuple)
        return time_index

    def _get_holiday_ndarray(self) -> ndarray:
        """
        基于matlab那些原始数据，提取节假日标志量：1代表是holiday，0代表不是
        """
        ts_matrix = self.load_ts_mat()
        return ts_matrix[:, -1].astype(int)

    def _get_british_summer_time(self) -> ndarray:
        ts_matrix = self.load_ts_mat()
        return ts_matrix[:, -2].astype(int)

    @abstractmethod
    def load_active_power_mat(self) -> ndarray:
        """
        载入Data_P_modified.mat或者Data_P.mat
        """
        pass

    @abstractmethod
    def load_ts_mat(self) -> ndarray:
        """
        载入Data_ts_modified.mat或者Data_ts.mat
        """
        pass

    @abstractmethod
    def get_temperature(self) -> ndarray:
        """
        载入Data_temperature_modified.mat或者Data_temperature.mat
        """
        pass


class ScotlandLongerDataset(ScotlandDataset):
    """
    指那些有四五年记录的bus，比如'Drum'和'John'
    """

    def __init__(self, name: str):
        """
        需要注意的是闰年2月29号没有记录，active power全部用0去填充的
        """
        if name not in ('Drum', 'John'):
            raise Exception('Wrong name')
        super().__init__(name)

    def load_active_power_mat(self) -> ndarray:
        return loadmat(self.matlab_mat_file_folder / 'Data_P_modified.mat')['P'].flatten()

    def load_ts_mat(self) -> ndarray:
        return loadmat(self.matlab_mat_file_folder / 'Data_ts_modified.mat')['ts']

    def get_temperature(self) -> ndarray:
        return loadmat(self.matlab_mat_file_folder / 'Data_temperature_modified.mat')['temperature'].flatten()


class ScotlandShorterDataset(ScotlandDataset):
    """
    指那些只有一年记录的bus，比如'MAYB'和'STHA'
    """

    def __init__(self, name: str):
        if name not in ('MAYB', 'STHA'):
            raise Exception('Wrong name')
        super().__init__(name)

    def load_active_power_mat(self) -> ndarray:
        return loadmat(self.matlab_mat_file_folder / 'Data_P.mat')['P'].flatten()

    def load_ts_mat(self) -> ndarray:
        return loadmat(self.matlab_mat_file_folder / 'Data_ts.mat')['ts']

    def get_temperature(self) -> ndarray:
        return loadmat(self.matlab_mat_file_folder / 'Data_temperature.mat')['temperature'].flatten()


if __name__ == '__main__':
    # ampds2_dataset, refit_dataset, uk_dale_dataset = load_datasets()
    # get_training_set_and_test_set_for_ampds2_dataset()
    # John_data = ScotlandLongerDataset('John')
    # John_data.set_weekends_and_holiday_to_zeros()
    for _this_no in (1, 2, 3, 4, 5, 7):
        for this_type in ('electric', 'heat', 'electric and heat'):
            load_low_carbon_london_heat(_this_no, this_type)
    # load_ampds2_weather()
