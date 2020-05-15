import numpy as np
import pandas as pd
from pandas import DataFrame
from nilmtk import MeterGroup
from prepare_datasets import load_datasets, load_ampds2_weather, ScotlandLongerDataset, ScotlandShorterDataset
from matplotlib import pyplot as plt
from Ploting.adjust_Func import reassign_linestyles_recursively_in_ax, adjust_legend_in_ax
from Ploting.fast_plot_Func import series, time_series, scatter
from dateutil import tz
from project_path_Var import project_path_
import os
from TimeSeries_Class import merge_two_time_series_df
from FFT_Class import FFTProcessor, STFT
from Time_Processing.datetime_utils import DatetimeOnehotEncoder
from workalendar.america import Canada
from TimeSeries_Class import UnivariateTimeSeries
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
                this_meter_df = next(
                    this_meter.load(ac_type='active', sample_period=sample_period))  # type: DataFrame
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

    lighting_category = next(
        ampds2_dataset.buildings[1].elec.select_using_appliances(category='lighting').load(
            ac_type='active',
            physical_quantity='power',
            sample_period=60))
    lighting_category_weather_df_merged = merge_two_time_series_df(lighting_category, _ampds2_weather_df)
    scatter(lighting_category_weather_df_merged['solar irradiation'].values,
            lighting_category_weather_df_merged[('power', 'active')].values,
            c=lighting_category_weather_df_merged.index.hour.values,
            cmap='hsv', alpha=0.5,
            x_label=r'Solar irradiation (W/m$\mathregular{^{2}}$)',
            y_label='Lighting category active power (W)',
            color_bar_name='Hour',
            save_file_=project_path_ + 'Report/solar irradiation_lighting', save_format='svg')
    # flag = np.bitwise_and(lighting_category_weather_df_merged[('power', 'active')].values > 20,
    #                       lighting_category_weather_df_merged['solar irradiation'].values < 10)


def energies_paper_get_category_and_consumption_for_ampds2_dataset():
    elec = ampds2_dataset.buildings[1].elec
    get_category_and_consumption(elec)


def energies_paper_fft_for_ampds2_dataset(sample_period):
    """
    :param sample_period: 从ampds2中载入数据的重采样的aggregate窗口长度，单位是秒
    """
    mains = next(ampds2_dataset.buildings[1].elec.mains().load(physical_quantity='power',
                                                               ac_type='active',
                                                               sample_period=sample_period))
    hpe = next(ampds2_dataset.buildings[1].elec.select_using_appliances(
        original_name=['HPE']).meters[0].load(ac_type='active', sample_period=sample_period))
    hpe = hpe.droplevel('physical_quantity', axis=1)  # type: DataFrame
    # np.log
    mains = mains.apply(np.log)
    hpe = hpe.apply(np.log)  # type: pd.DataFrame
    # detrend
    mains = UnivariateTimeSeries(mains)
    mains.detrend(resample_args_dict={'rule': '168H'},
                  inplace=True)  # type: pd.DataFrame

    hpe = UnivariateTimeSeries(hpe)
    hpe.detrend(resample_args_dict={'rule': '168H'},
                inplace=True)  # type: pd.DataFrame
    # 画图，一周周的叠加
    save_figure_path = Path(project_path_) / 'Data/Results/Energies_paper/fft/ampds2'
    save_figure_path.mkdir(parents=True, exist_ok=True)
    # mains.plot_group_by_week(
    #     x_label='Day of week',
    #     y_label='Detrended log-load',
    #     save_file_=str(save_figure_path / 'detrend_log'),
    #     save_format='png')
    # hpe.plot_group_by_week(
    #     x_label='Day of week',
    #     y_label='Detrended log-load',
    #     save_file_=str(save_figure_path / 'detrend_log_hpe'),
    #     save_format='png')

    # 将holiday和weekends置为zeros
    datetime_onehot_encoder = DatetimeOnehotEncoder(to_encoding_args=('holiday', 'weekday'))
    time_var_transformed = datetime_onehot_encoder(mains.data.index,
                                                   country=Canada())
    mask = np.any((time_var_transformed[('holiday', 1)] == 1,
                   time_var_transformed[('weekday', 6)] == 1,
                   time_var_transformed[('weekday', 7)] == 1), axis=0)
    mains.data[mask] = 0  # type:pd.DataFrame
    hpe.data[mask] = 0  # type:pd.DataFrame

    # 将处理好（p.log，detrend, 部分被置零）的数据进行FFT
    # fft_results = FFTProcessor(mains.data.values.flatten(), sampling_period=sample_period, name='mains')
    # fft_results.plot(save_as_docx_path=save_figure_path / 'mains.docx')
    # fft_results = FFTProcessor(hpe.data.values.flatten(), sampling_period=sample_period, name='hpe')
    # fft_results.plot(save_as_docx_path=save_figure_path / 'hpe.docx')
    fft_results = LASSOFFT(mains.data.values.flatten(), sampling_period=sample_period, name='mains')


def energies_paper_fft_for_scotland():
    considered_bus = {'JOHN': ScotlandLongerDataset('John'),
                      'DRUM': ScotlandLongerDataset('Drum'),
                      'MAYB': ScotlandShorterDataset('MAYB'),
                      'STHA': ScotlandShorterDataset('STHA')}
    for key, this_bus in considered_bus.items():
        save_figure_folder_path = Path(project_path_) / f'Data/Results/Energies_paper/fft/{key}'
        save_figure_folder_path.mkdir(parents=True, exist_ok=True)
        # np.log, detrend
        this_bus_active_power = UnivariateTimeSeries(
            this_bus.dataset[['active power']].apply(np.log)
        )  # type: UnivariateTimeSeries
        this_bus_active_power.detrend({'rule': '168H'},  # 以周为单位detrend
                                      inplace=True)
        # 画图，一周周的叠加
        # this_bus_active_power.plot_group_by_week(x_label='Day of week',
        #                                          y_label='Detrended log-load',
        #                                          save_file_=str(save_figure_folder_path / 'detrend_log'),
        #                                          save_format='png')
        # 置零
        this_bus_active_power = this_bus_active_power.data  # type:pd.DataFrame
        _, zeros_mask = this_bus.set_weekends_and_holiday_to_zeros(inplace=False)
        this_bus_active_power[zeros_mask] = 0  # type: pd.DataFrame
        # 删inf，nan
        this_bus_active_power[np.isinf(this_bus_active_power)] = np.nan
        this_bus_active_power = this_bus_active_power.fillna(0)  # type: pd.DataFrame
        # 开始FFT
        fft_results = FFTProcessor(this_bus_active_power.values.flatten(),
                                   sampling_period=60 * 30,
                                   name=key)
        fft_results.plot(save_as_docx_path=save_figure_folder_path / f'{key}.docx')


def energies_paper():
    # energies_paper_one_day_visulisation_for_ampds2_dataset()
    # energies_paper_correlation_exploration_for_ampds2_dataset()
    # energies_paper_get_category_and_consumption_for_ampds2_dataset()
    energies_paper_fft_for_ampds2_dataset(60 * 30)
    energies_paper_fft_for_scotland()


if __name__ == '__main__':
    ampds2_dataset, refit_dataset, uk_dale_dataset = load_datasets()
    energies_paper()
