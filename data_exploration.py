import numpy as np
import pandas as pd
from pandas import DataFrame
from nilmtk import DataSet, MeterGroup, Building
from prepare_datasets import load_datasets, load_ampds2_weather
from typing import Tuple
import copy
from matplotlib import pyplot as plt
from Ploting.adjust_Func import reassign_linestyles_recursively_in_ax, adjust_legend_in_ax
from Ploting.fast_plot_Func import series, time_series, scatter, hist
import datetime
from dateutil import tz
from project_path_Var import project_path_
import os
from Time_Processing.SynchronousTimeSeriesData_Class import merge_two_time_series_df
from FFT_Class import FFTProcessor
from scipy.io import loadmat
from pathlib import Path


def plot_active_power_for_meter_group(meter_group: MeterGroup, sample_period=60, **kwargs):
    ax_ = None
    for this_meter in meter_group.meters:
        if isinstance(this_meter, MeterGroup):
            raise Exception("this_meter是MeterGroup")
        else:
            if this_meter.appliances.__len__() > 1:
                raise Exception("this_meter.appliances的个数超过1个")
            else:
                label = 'WHE' if this_meter.is_site_meter() else this_meter.appliances[0].metadata['original_name']
                this_meter_df = next(this_meter.load(ac_type='active', sample_period=sample_period))  # type: DataFrame
                x = this_meter_df.index.to_pydatetime()
                y = this_meter_df.iloc[:, 0].values
                ax_ = time_series(x=x, y=y, ax=ax_,
                                  y_label='Active power (W)', label=label, ncol=3,
                                  y_lim=(-5, None),
                                  x_axis_format=kwargs.get('x_axis_format'), tz=kwargs.get('tz'))
    # 调整线形和legend位置
    ax_ = reassign_linestyles_recursively_in_ax(ax_)
    return ax_


def get_category_and_consumption(meter_group: MeterGroup):
    """
    得到一个MeterGroup中的所有子表的能量消耗量，能量消耗比例，名字和分类
    :param meter_group:
    :return:
    """
    # 比例
    fraction = meter_group.submeters().fraction_per_meter().dropna()
    fraction_df = pd.DataFrame(fraction * 100, columns=['fraction'])
    # 能量
    energy = meter_group.submeters().energy_per_meter().transpose()['active']
    energy_df = pd.DataFrame(energy)
    # 能量和比例
    energy_fraction_df = pd.DataFrame(['{:.2f}, {:.2f} %'.format(energy_df.values.flatten()[i],
                                                                 fraction_df.values.flatten()[i])
                                       for i in range(fraction.size)],
                                      columns=['energy_fraction_str'],
                                      index=fraction.index)
    # 名字和分类
    name = []
    category = []
    for i in range(fraction.index.__len__()):
        name.append(meter_group.submeters()[fraction.index[i]].appliances[0].metadata['original_name'])
        category.append(meter_group.submeters()[fraction.index[i]].appliances[0].type.get('categories'))
    name_df = pd.DataFrame(name, index=fraction.index, columns=['name'])
    category_df = pd.DataFrame(category, index=fraction.index)
    category_df.reindex(columns=pd.MultiIndex.from_product([['category'], category_df.columns]))
    # merge导出csv
    final = pd.concat((name_df, energy_df, fraction_df, category_df, energy_fraction_df), axis=1)
    final.to_csv(project_path_ + '/Report/Energies paper/{}_get_category_and_consumption.csv'.format(
        meter_group.meters[0].identifier.dataset
    ))
    return final


def energies_paper_one_day_visulisation_for_ampds2_dataset():
    def one_day_visulisation(this_dataset, start, end):
        this_dataset.set_window(start=start, end=end)
        elec = this_dataset.buildings[1].elec  # type: MeterGroup
        _ax = plot_active_power_for_meter_group(elec, x_axis_format="%H", tz=tz.gettz('Canada/Pacific'))
        adjust_legend_in_ax(_ax, protocol='Outside center left', ncol=3)
        _ax.set_xlabel('Time (hour of a day, time zone: America/Vancouver)')
        return _ax

    # ampds2_dataset
    ax = one_day_visulisation(ampds2_dataset, "2013-02-06", "2013-02-07")
    plt.savefig(os.path.join(project_path_,
                             'Report/ampds2_dataset_2013-02-06.svg'))
    ax = one_day_visulisation(ampds2_dataset, "2013-08-07", "2013-08-08")
    plt.savefig(os.path.join(project_path_,
                             'Report/ampds2_dataset_2013-08-07.svg'))


def energies_paper_correlation_exploration_for_ampds2_dataset():
    _ampds2_weather_df = load_ampds2_weather()
    # %% main
    mains_df = next(ampds2_dataset.buildings[1].elec.mains().load(ac_type='active', sample_period=60))

    mains_weather_df_merged = merge_two_time_series_df(mains_df, _ampds2_weather_df)
    # 温度
    # scatter(mains_weather_df_merged['temperature'].values,
    #         mains_weather_df_merged[('power', 'active')].values,
    #         c=mains_weather_df_merged.index.month.values,
    #         cmap='hsv', alpha=0.5,
    #         x_label='Temperature ($^\circ$C)',
    #         y_label='WHE active power (W)',
    #         color_bar_name='Month',
    #         save_file_=project_path_ + 'Report/temperature_WOE', save_format='svg')

    # 光照
    # scatter(mains_weather_df_merged['solar irradiation'].values,
    #         mains_weather_df_merged[('power', 'active')].values,
    #         c=mains_weather_df_merged.index.hour.values,
    #         cmap='hsv', alpha=0.5,
    #         x_label='Solar irradiation (W/m$\mathregular{^{2}}$)',
    #         y_label='WHE active power (W)',
    #         color_bar_name='Hour',
    #         save_file_=project_path_ + 'Report/solar irradiation_WOE', save_format='svg')

    # %% heating和lighting
    heating_category = next(ampds2_dataset.buildings[1].elec.select_using_appliances(category='heating').load(
        ac_type='active',
        physical_quantity='power',
        sample_period=60))
    heating_category_weather_df_merged = merge_two_time_series_df(heating_category, _ampds2_weather_df)
    # scatter(heating_category_weather_df_merged['temperature'].values,
    #         heating_category_weather_df_merged[('power', 'active')].values,
    #         c=heating_category_weather_df_merged.index.month.values,
    #         cmap='hsv', alpha=0.5,
    #         x_label='Temperature ($^\circ$C)',
    #         y_label='Heating category active power (W)',
    #         color_bar_name='Month',
    #         save_file_=project_path_ + 'Report/temperature_heating', save_format='svg')

    lighting_category = next(ampds2_dataset.buildings[1].elec.select_using_appliances(category='lighting').load(
        ac_type='active',
        physical_quantity='power',
        sample_period=60))
    lighting_category_weather_df_merged = merge_two_time_series_df(lighting_category, _ampds2_weather_df)
    scatter(lighting_category_weather_df_merged['solar irradiation'].values,
            lighting_category_weather_df_merged[('power', 'active')].values,
            c=lighting_category_weather_df_merged.index.hour.values,
            cmap='hsv', alpha=0.5,
            x_label='Solar irradiation (W/m$\mathregular{^{2}}$)',
            y_label='Lighting category active power (W)',
            color_bar_name='Hour',
            save_file_=project_path_ + 'Report/solar irradiation_lighting', save_format='svg')
    # flag = np.bitwise_and(lighting_category_weather_df_merged[('power', 'active')].values > 20,
    #                       lighting_category_weather_df_merged['solar irradiation'].values < 10)


def energies_paper_get_category_and_consumption_for_ampds2_dataset():
    elec = ampds2_dataset.buildings[1].elec
    get_category_and_consumption(elec)


def energies_paper_fft_for_ampds2_dataset():
    mains = next(ampds2_dataset.buildings[1].elec.mains().load(physical_quantity='power', ac_type='active'))
    fft_results = FFTProcessor(mains.values.flatten(), 60)
    fft_results.single_sided_fft_results_usable(ordered_by_magnitude=True).iloc[:1000].to_csv(
        project_path_ + 'Report/Ampds2 mains fft top 1000 descending by magnitude.csv')
    series(fft_results.single_sided_frequency_axis, np.log(fft_results.single_sided_amplitude_spectrum),
           x_label='Frequency (Hz)', y_label='Log magnitude',
           title='Ampds2 mains FFT')
    series(fft_results.single_sided_frequency_axis, fft_results.single_sided_angle,
           x_label='Frequency (Hz)', y_label='Phase angle (rad)',
           title='Ampds2 mains FFT')

    series(fft_results.single_sided_frequency_axis, np.log(fft_results.single_sided_amplitude_spectrum),
           x_label='Frequency (Hz)', y_label='Log magnitude',
           x_lim=(-0.000001, 0.00011), y_lim=(10, 22),
           title='Ampds2 mains FFT (zoom in)')
    series(fft_results.single_sided_frequency_axis, fft_results.single_sided_angle,
           x_label='Frequency (Hz)', y_label='Phase angle (rad)',
           x_lim=(-0.000001, 0.00011),
           title='Ampds2 mains FFT (zoom in)')


def energies_paper_fft_for_scotland():
    considered_bus = {'JOHN': loadmat(Path(project_path_) / 'Data/Raw/John_and_China/Data_load.mat')['John_load'],
                      'DRUM': loadmat(Path(project_path_) / 'Data/Raw/Scotland selected/Drum/Data_P.mat')['P'],
                      'MAYB': loadmat(Path(project_path_) / 'Data/Raw/Scotland selected/MAYB/Data_P.mat')['P'],
                      'STHA': loadmat(Path(project_path_) / 'Data/Raw/Scotland selected/STHA/Data_P.mat')['P']}
    for key, values in considered_bus.items():
        fft_results = FFTProcessor(values.flatten(), 60 * 30)
        fft_results.single_sided_fft_results_usable(ordered_by_magnitude=True).iloc[:1000].to_csv(
            project_path_ + f'Report/Scotland {key} fft top 1000 descending by magnitude.csv')
    # series(fft_results.single_sided_frequency_axis, np.log(fft_results.single_sided_amplitude_spectrum),
    #        x_label='Frequency (Hz)', y_label='Log magnitude',
    #        title='Ampds2 mains FFT')
    # series(fft_results.single_sided_frequency_axis, fft_results.single_sided_angle,
    #        x_label='Frequency (Hz)', y_label='Phase angle (rad)',
    #        title='Ampds2 mains FFT')
    #
    # series(fft_results.single_sided_frequency_axis, np.log(fft_results.single_sided_amplitude_spectrum),
    #        x_label='Frequency (Hz)', y_label='Log magnitude',
    #        x_lim=(-0.000001, 0.00011), y_lim=(10, 22),
    #        title='Ampds2 mains FFT (zoom in)')
    # series(fft_results.single_sided_frequency_axis, fft_results.single_sided_angle,
    #        x_label='Frequency (Hz)', y_label='Phase angle (rad)',
    #        x_lim=(-0.000001, 0.00011),
    #        title='Ampds2 mains FFT (zoom in)')


def energies_paper():
    # energies_paper_one_day_visulisation_for_ampds2_dataset()
    # energies_paper_correlation_exploration_for_ampds2_dataset()
    # energies_paper_get_category_and_consumption_for_ampds2_dataset()
    # energies_paper_fft_for_ampds2_dataset()
    energies_paper_fft_for_scotland()


if __name__ == '__main__':
    ampds2_dataset, refit_dataset, uk_dale_dataset = load_datasets()
    energies_paper()
