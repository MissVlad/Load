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
from FFT_Class import FFTProcessor, STFTProcessor
from Time_Processing.datetime_utils import DatetimeOnehotEncoder
from workalendar.america import Canada
from TimeSeries_Class import UnivariateTimeSeries, TimeSeries, WindowedTimeSeries
from pathlib import Path
import datetime
from Correlation_Modeling.FFTCorrelation_Class import BivariateFFTCorrelation, FFTCorrelation
import re
from Writting.utils import put_all_png_in_a_path_into_a_docx


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
    fft_results = FFTProcessor(mains.data.values.flatten(), sampling_period=sample_period, name='mains')
    fft_results.plot(save_as_docx_path=save_figure_path / 'mains.docx')
    fft_results = FFTProcessor(hpe.data.values.flatten(), sampling_period=sample_period, name='hpe')
    fft_results.plot(save_as_docx_path=save_figure_path / 'hpe.docx')


def energies_paper_fft_for_scotland():
    considered_bus = {'JOHN': ScotlandLongerDataset('John'),
                      'DRUM': ScotlandLongerDataset('Drum'),
                      'MAYB': ScotlandShorterDataset('MAYB'),
                      'STHA': ScotlandShorterDataset('STHA')}
    for key, this_bus in considered_bus.items():
        """
        目前只考虑John
        """
        if key != 'JOHN':
            continue
        save_figure_folder_path = Path(project_path_) / f'Data/Results/Energies_paper/fft/{key}'
        save_figure_folder_path.mkdir(parents=True, exist_ok=True)
        # np.log, detrend
        this_bus_active_power = UnivariateTimeSeries(
            this_bus.dataset[['active power']].apply(np.log)
        )  # type: UnivariateTimeSeries
        # this_bus_active_power.detrend({'rule': '168H'},  # 以周为单位detrend
        #                               inplace=True)
        # 画图，一周周的叠加
        # this_bus_active_power.plot_group_by_week(x_label='Day of week',
        #                                          y_label='Detrended log-load',
        #                                          save_file_=str(save_figure_folder_path / 'detrend_log'),
        #                                          save_format='png')
        # 置零
        # _, zeros_mask = this_bus.set_weekends_and_holiday_to_zeros(inplace=False)
        # this_bus_active_power[zeros_mask] = 0  # type: pd.DataFrame
        # 删inf，nan
        this_bus_active_power[np.isinf(this_bus_active_power)] = np.nan
        this_bus_active_power = this_bus_active_power.fillna(0)
        # 开始FFT
        # fft_results = FFTProcessor(this_bus_active_power.values.flatten(),
        #                            sampling_period=60 * 30,
        #                            name=key)
        # fft_results.plot(save_as_docx_path=save_figure_folder_path / f'{key}.docx')

        stft_class = STFTProcessor(this_bus_active_power,
                                   sampling_period=60 * 30,
                                   name=key,
                                   window_length=datetime.timedelta(days=1),
                                   window=None)
        stft_class.do_stft(2 ** 16)
        """
        stft_class.find_peaks_of_fft_frequency('1/half day',
                                       considered_window_index=(7, 182),
                                       plot_args={'only_plot_peaks': True,
                                                  'annotation_for_peak_f_axis_indices': (1, 2),
                                                  'annotation_y_offset_for_f': (70, 50),
                                                  'annotation_y_offset_for_p': (0.5, -0.2)})
        """

        stft_class.find_peaks_of_fft_frequency('1/half day',
                                               considered_window_index=range(365),
                                               plot_args=None)


def energies_paper_fft_correlation():
    considered_bus = {'JOHN': ScotlandLongerDataset('John'),
                      'DRUM': ScotlandLongerDataset('Drum'),
                      'MAYB': ScotlandShorterDataset('MAYB'),
                      'STHA': ScotlandShorterDataset('STHA')}
    for key, this_bus in considered_bus.items():
        """
        目前只考虑John
        """
        if key != 'JOHN':
            continue
        bivariate_time_series = merge_two_time_series_df(this_bus.dataset[['active power']],
                                                         this_bus.dataset[['temperature']])
        bivariate_time_series = WindowedTimeSeries({'main': bivariate_time_series['temperature'],
                                                    'vice': bivariate_time_series['active power']},
                                                   window_length=datetime.timedelta(days=1))
        for i, this_day in enumerate(bivariate_time_series):
            """
            目前只考虑day7, day189
            """
            if i not in (7, 77, 189, 280):
                continue
            # load_time_series的一些图
            load_time_series = UnivariateTimeSeries(this_day.iloc[:, 0])
            # load_title = f"Load, original, starting from {load_time_series.first_valid_index()}"
            # series(np.arange(0, 24, 0.5), load_time_series.values.flatten(),
            #        x_label='Hour', y_label='Active load (MW)',
            #        title=load_title,
            #        save_file_=f'{project_path_}Code/temp/load_{i}',
            #        save_format='png')
            # load_time_series_fft = FFTProcessor(load_time_series.values.flatten(),
            #                                     sampling_period=load_time_series.adjacent_recordings_timedelta.seconds,
            #                                     n_fft=65536,
            #                                     name='')
            # plot_args = {'only_plot_peaks': True,
            #              'overridden_plot_x_lim': (None, None)}
            # for i_unit, unit in enumerate(('1/day', '1/half day')):
            #     _, _, f_plot, p_plot = load_time_series_fft.find_peaks_of_fft_frequency(considered_frequency_unit=unit,
            #                                                                             plot_args=plot_args,
            #                                                                             base_freq_is_a_peak=False)
            #     f_plot.set_title(load_title)
            #     p_plot.set_title(load_title)
            #
            #     plt.savefig(f'{project_path_}Code/temp/load_{i}_{i_unit}_p.png', format='png', dpi=300)
            #     plt.close()
            #     plt.savefig(f'{project_path_}Code/temp/load_{i}_{i_unit}_f.png', format='png', dpi=300)

            # load_time_series_reconstruct的一些图
            # load_time_series_reconstruct = load_time_series.reconstruct_using_ift(
            #     n_fft=65_536,
            #     find_peaks_of_fft_frequency_args={
            #         'considered_frequency_unit': '1/half day'
            #     },
            #     considered_peaks_index=list(range(0, 24)),
            #     do_plotting=True,
            #     scale_to_minus_plus_one_flag=False
            # )
            # plt.savefig(f'{project_path_}Code/temp/load_{i}.svg', format='svg', dpi=300)

            # series(np.arange(0, 24, 0.5), load_time_series_reconstruct[0],
            #        x_label='Hour', y_label='Active load (MW)',
            #        title=re.sub(r"original", 'ifft (rescaled)', load_title),
            #        save_file_=f'{project_path_}Code/temp/load_{i}_',
            #        save_format='png')

            b_fft_correlation = BivariateFFTCorrelation(n_fft=65_536,
                                                        considered_frequency_unit='1/half day',
                                                        _time_series=this_day,
                                                        correlation_func=('Spearman',),
                                                        main_considered_peaks_index=list(range(1, 7)),
                                                        vice_considered_peaks_index=list(range(1, 7)),
                                                        )
            # vice_find_peaks_args={
            #     'plot_args': {
            #         'only_plot_peaks': True,
            #         'overridden_plot_x_lim': (None, None),
            #         'annotation_for_peak_f_axis_indices': (1, 2),
            #         'annotation_y_offset_for_f': (70, 50),
            #         'annotation_y_offset_for_p': (0.1, -0.1)
            #     }
            # })
            # b_fft_correlation.corr_between_pairwise_peak_f_ifft()
            # b_fft_correlation.corr_between_main_and_one_selected_vice_peak_f_ifft()
            b_fft_correlation.corr_between_main_and_combined_selected_vice_peaks_f_ifft(
                vice_extra_hz_f=[1 / (365 * 24 * 3600),
                                 1 / (7 * 24 * 3600)]
            )


def energies_paper():
    # energies_paper_one_day_visulisation_for_ampds2_dataset()
    # energies_paper_correlation_exploration_for_ampds2_dataset()
    # energies_paper_get_category_and_consumption_for_ampds2_dataset()
    # energies_paper_fft_for_ampds2_dataset(60 * 30)
    # energies_paper_fft_for_scotland()
    energies_paper_fft_correlation()


if __name__ == '__main__':
    ampds2_dataset, refit_dataset, uk_dale_dataset = load_datasets()
    energies_paper()
