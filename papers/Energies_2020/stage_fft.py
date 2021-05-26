import datetime

import numpy as np
import pandas as pd

from TimeSeries_Class import TimeSeries, WindowedTimeSeries
from Correlation_Modeling.FFTCorrelation_Class import BivariateFFTCorrelation
from project_utils import *
from PhysicalInstance_Class import *
from Writting import *
from tqdm import tqdm
from papers.Energies_2020.paper_utils import *
from Time_Processing.format_convert_Func import np_datetime64_to_datetime
from FFT_Class import *
from Correlation_Modeling.utils import *
from sklearn.preprocessing import MinMaxScaler
from functools import reduce
from File_Management.load_save_Func import *
from File_Management.path_and_file_management_Func import *
from tqdm import tqdm

STFT_HST = {
    "AMPDS2": {
        'appliance_name': 'heating',
        'appliance_plot_y_label': 'Heating Active Power [W]',
        'environment_name': 'temperature',
        'environment_plot_y_label': 'Environmental Temperature [\u00B0C]',
        'y_unit': 'W'
    },
    "UKDALE": {
        'appliance_name': 'lighting',
        'appliance_plot_y_label': 'Lighting Active Power [W]',
        'environment_name': 'radiation_surface',
        'environment_plot_y_label': 'Solar Irradiance [W/$\mathregular{m}^2$]'
    },
    "JOHN": {
        'appliance_name': 'heating',
        'appliance_plot_y_label': 'Heating Active Power [W]',
        'environment_name': 'temperature',
        'environment_plot_y_label': 'Environmental Temperature [\u00B0C]',
        'y_unit': 'MW'
    }

}

PERIOD = (
    'half day',
    'day',
    '2days',
    '5days',
    'week',
    'month',
    'season',
    'year'
)

PERIOD2FREQ = {
    'half day': 1 / (0.5 * 24 * 3600),
    'day': 1 / (1 * 24 * 3600),
    '2days': 1 / (2 * 24 * 3600),
    '5days': 1 / (5 * 24 * 3600),
    'week': 1 / (7 * 24 * 3600),
    'month': 1 / (28 * 24 * 3600),
    'season': 1 / (91 * 24 * 3600),
    'year': 1 / (364 * 24 * 3600),
}


def get_data_func(new_mode: bool = False):
    if not new_mode:
        # JOHN MV
        john_data_set_date_start = datetime.datetime(year=2009, month=1, day=1)
        # john_data_set_date_end = datetime.datetime(year=2009, month=1, day=3)
        john_data_set_date_end = datetime.datetime(year=2010, month=1, day=1)
        john_data = ScotlandLongerDataset("John").dataset
        john_data = PhysicalInstanceDataFrame(
            john_data[np.bitwise_and(john_data.index >= john_data_set_date_start,
                                     john_data.index < john_data_set_date_end)],
            obj_name='John MV', predictor_names=(), dependant_names=()
        )
        john_data = john_data.rename(
            {
                'active power': 'mains',
            },
            axis=1
        )
        john_data['heating'] = np.full(john_data.shape[0], 0)
        return john_data
    else:
        john_data = ScotlandLongerDataset("John").dataset
        # sasa removes days
        """
        to_remove_days = []
        for year in range(2007, 2013):
            to_remove_days.append(
                np.bitwise_and(john_data.index >= datetime.datetime(year=year, month=12, day=31),
                               john_data.index < datetime.datetime(year=year + 1, month=1, day=1))
            )
            if year in {2008, 2012}:
                to_remove_days.append(
                    np.bitwise_and(john_data.index >= datetime.datetime(year=year, month=2, day=29),
                                   john_data.index < datetime.datetime(year=year, month=3, day=1))
                )
        to_remove_days = john_data.index[np.any(np.array(to_remove_days), axis=0)]
        john_data.drop(to_remove_days, inplace=True)
        """

        # Get func
        class JohnDataGen:
            per_day = 48

            def __init__(self):
                self.i = 0

            def get(self, day_rank: int, period: str = None):
                datetime_start = datetime.timedelta(days=day_rank) + datetime.datetime(year=2008, month=1, day=1)

                # idx_start = self.per_day * (364 + day_rank)
                idx_start = np.where(john_data.index == datetime_start)[0][0]
                idx_end = idx_start + self.per_day
                ans = john_data.iloc[idx_start:idx_end]
                if period == "half day":
                    pass
                elif period == "day":
                    pass
                elif period == "2days":
                    temp = john_data.iloc[idx_end - self.per_day * 2:idx_start]
                    ans = pd.concat([ans, temp])
                elif period == "5days":
                    temp = john_data.iloc[idx_end - self.per_day * 5:idx_start]
                    ans = pd.concat([ans, temp])
                elif period == "week":
                    temp = john_data.iloc[idx_end - self.per_day * 7:idx_start]
                    ans = pd.concat([ans, temp])
                elif period == "month":
                    temp = john_data.iloc[idx_end - self.per_day * 28:idx_start]
                    ans = pd.concat([ans, temp])
                elif period == "season":
                    temp = john_data.iloc[idx_end - self.per_day * 91:idx_start]
                    ans = pd.concat([ans, temp])
                elif period == "year":
                    temp = john_data.iloc[idx_end - self.per_day * 364:idx_start]
                    ans = pd.concat([ans, temp])
                elif period is None:
                    return {key: self.get(day_rank, key) for key in PERIOD}
                else:
                    raise

                return ans

            def __len__(self):
                return john_data.loc[datetime.datetime(year=2008, month=1, day=1):
                                     datetime.datetime(year=2013, month=1, day=1)].__len__() // self.per_day

            def __iter__(self):
                self.i = 0  # reset，以便以后继续能iter
                return self

            def __next__(self) -> ndarray:
                try:
                    if self.i == self.__len__():
                        raise IndexError
                    fetch_data = self.get(self.i)
                    self.i += 1

                except IndexError:
                    raise StopIteration
                return fetch_data

        return JohnDataGen


def data_to_pseudo_reindex_time_series(_data):
    pseudo_reindex_time_series = copy.deepcopy(_data)
    a = pseudo_reindex_time_series.index[:48]
    b = pseudo_reindex_time_series.index[48:]
    pseudo_reindex_time_series.index = np.concatenate([b, a])
    return TimeSeries(pseudo_reindex_time_series)


def lasso_fft_for_one_feature(data_i, feature, return_fourier_obj: bool = False):
    do_lasso_fitting_args = {'alpha': 1e-8,
                             'max_iter': 1_000_000,
                             'tol': 1e-8,
                             'random_state': 0}
    lasso_fft_reconstruct = {'all': 0}
    freq_arr = []
    coefficient_a_arr = []
    coefficient_b_arr = []
    for period, freq in PERIOD2FREQ.items():
        temp = LASSOFFTProcessor(
            frequency=np.array([freq]),
            target=data_to_pseudo_reindex_time_series(data_i[period][[feature]])
        ).do_lasso_fitting(**do_lasso_fitting_args)
        # single freq to get rid of DC component
        lasso_fft_reconstruct[period] = temp[1]
        lasso_fft_reconstruct[period] += np.mean(
            data_to_pseudo_reindex_time_series(data_i[period][[feature]]).values - lasso_fft_reconstruct[period]
        )
        # Get rid of DC components
        lasso_fft_reconstruct['all'] += temp[1][:48] - temp[2].coefficient_a[0]
        #
        if return_fourier_obj:
            freq_arr.append(temp[2].frequency[1])
            coefficient_a_arr.append(temp[2].coefficient_a[1])
            coefficient_b_arr.append(temp[2].coefficient_b[1])
    # Find the new DC components given the combination of all freq. components
    final_dc = np.mean(
        data_to_pseudo_reindex_time_series(data_i['day'][[feature]]).values - lasso_fft_reconstruct['all']
    )
    lasso_fft_reconstruct['all'] += final_dc
    #
    if not return_fourier_obj:
        return lasso_fft_reconstruct

    freq_arr.insert(0, 0)
    coefficient_a_arr.insert(0, 0)
    coefficient_b_arr.insert(0, final_dc)
    fourier_obj = SCFormFourierSeriesProcessor(frequency=np.array(freq_arr),
                                               coefficient_a=np.array(coefficient_a_arr),
                                               coefficient_b=np.array(coefficient_b_arr))
    return lasso_fft_reconstruct, fourier_obj


def get_combinations(_data, _i):
    if _i == 0:
        return {(("half day", 0),): np.full(_data['half day'].shape, 0.),
                (("half day", 1),): _data['half day'][:48]}

    _forward = get_combinations(_data, _i - 1)
    _ans = dict()
    for _key, _value in _forward.items():
        for _j in range(2):
            _new_key = tuple(list(_key) + [(PERIOD[_i], _j)])
            _new_value = _value + _j * _data[PERIOD[_i]][:48]
            _ans[_new_key] = _new_value

    return _ans


def get_spearman_corr(x, y):
    return BivariateCorrelationAnalyser(x, y)('Spearman').correlation


def find_the_best_combination(_power_com, _other_com):
    _ans = [float('inf'), 0, None]
    for _power_com_key, _power_com_value in _power_com.items():
        for _other_com_key, _other_com_value in _other_com.items():
            _a = MinMaxScaler().fit_transform(_power_com_value.reshape(-1, 1))
            _b = MinMaxScaler().fit_transform(_other_com_value.reshape(-1, 1))
            corr = BivariateCorrelationAnalyser(_a, _b)
            _now_ans = corr('Spearman').correlation
            if _now_ans <= _ans[0]:
                _used_com_num = sum([1 for _a_i in _a if _a_i[1] == 1]) + sum([1 for _b_i in _b if _b_i[1] == 1])
                if _used_com_num > _ans[1]:
                    _ans[0] = _now_ans
                    _ans[1] = _used_com_num
                    _ans[2] = [_power_com_key, _other_com_key]

    return _ans


def cal_freq_correlations(data, power_ifft: dict, other_ifft: dict, feature: str):
    hst = {key: {'full_length_raw': np.nan,
                 'full_length_freq': np.nan,
                 'day_length_raw': np.nan,
                 'day_length_freq': np.nan} for key in PERIOD}
    for period in PERIOD:
        hst[period]['full_length_raw'] = get_spearman_corr(data[period]['active power'].values,
                                                           data[period][feature].values)
        hst[period]['full_length_freq'] = get_spearman_corr(power_ifft[period], other_ifft[period])
        hst[period]['day_length_raw'] = get_spearman_corr(data[period]['active power'].values[:48],
                                                          data[period][feature].values[:48])
        hst[period]['day_length_freq'] = get_spearman_corr(power_ifft[period][:48], other_ifft[period][:48])
    return hst


def sum_up_selected_freq_correlations(power_ifft: dict, other_ifft: dict, freq_corr_hst, normalisation: bool = True):
    power_sum = np.array([0.] * 48)
    other_sum = np.array([0.] * 48)
    for period in PERIOD:
        if freq_corr_hst[period]['day_length_raw'] > freq_corr_hst[period]['day_length_freq']:
            power_sum += power_ifft[period][:48]
            other_sum += other_ifft[period][:48]
    if normalisation:
        power_sum = MinMaxScaler().fit_transform(power_sum.reshape(-1, 1)).flatten()
        other_sum = MinMaxScaler().fit_transform(other_sum.reshape(-1, 1)).flatten()
    return power_sum, other_sum


def stft_and_correlation(name: Union[str, None] = None, save_to_buffer=False, use_lasso: bool = False):
    assert name in {"AMPDS2", "UKDALE", None}

    if name is None:
        data_set = get_data_func()
    else:
        data_set = globals()[name.replace("_", "_DATA_")]["training"].data

    now_data_set = copy.deepcopy(data_set)
    # this_data_set = this_data_set.iloc[24 * 185:24 * 190 - 1]
    # Reindex to make sure it's index is has equal diff
    tz = now_data_set.index.__getattribute__('tz')
    first_index, last_index = now_data_set.index[0], now_data_set.index[-1]
    now_data_set_freq = int(np.min(np.diff(now_data_set.index)) / np.timedelta64(1, 's'))
    now_data_set = now_data_set.reindex(pd.DatetimeIndex(
        pd.date_range(
            start=datetime.datetime(first_index.year, first_index.month, first_index.day),
            end=datetime.datetime(last_index.year, last_index.month, last_index.day) + datetime.timedelta(1),
            freq=f"{now_data_set_freq}S",
            tz=tz
        )
    ))
    # %% window length = 1D
    stft_hst_value = STFT_HST[name]
    document = docx_document_template_to_collect_figures()
    for i, now_window_data in enumerate(tqdm(WindowedTimeSeries(now_data_set,
                                                                window_size=datetime.timedelta(days=1),
                                                                window_shift=datetime.timedelta(days=1)))):

        if now_window_data.size == 0:
            continue

        appliance_name = stft_hst_value['appliance_name']
        environment_name = stft_hst_value['environment_name']
        if np.any(np.isnan(now_window_data[['mains', appliance_name, environment_name]].values)):
            continue

        x_axis_value = [np_datetime64_to_datetime(x, tz=tz) for x in now_window_data.index.values]

        # %% Obtain the required dimensions of data
        # Main
        mains = now_window_data['mains']
        y_unit = stft_hst_value['y_unit']

        mains_plot = time_series(
            x=x_axis_value, y=mains.values, tz=tz,
            y_label=f"Main Active Power [{y_unit}]",
            **TIME_SERIES_PLOT_KWARGS, save_to_buffer=save_to_buffer
        )

        appliance = now_window_data[appliance_name]
        environment = now_window_data[environment_name]

        if name != 'JOHN':
            appliance_plot = time_series(x=x_axis_value, y=appliance.values, tz=tz,
                                         y_label=stft_hst_value['appliance_plot_y_label'], **TIME_SERIES_PLOT_KWARGS,
                                         save_to_buffer=save_to_buffer)
        else:
            appliance_plot = None

        environment_plot = time_series(x=x_axis_value, y=environment.values, tz=tz,
                                       y_label=stft_hst_value['environment_plot_y_label'], **TIME_SERIES_PLOT_KWARGS,
                                       save_to_buffer=save_to_buffer)

        # %% LASSO-FFT and Correlation
        b_fft_correlation = BivariateFFTCorrelation(
            n_fft=2 ** 20,
            considered_frequency_unit='1/half day',
            _time_series=now_window_data[[environment_name, 'mains']],
            correlation_func=('Spearman',),
            main_considered_peaks_index=list(range(1, 1)),
            vice_considered_peaks_index=list(range(1, 1)),
            vice_find_peaks_args={
                'scipy_signal_find_peaks_args': {

                }
            }
        )
        lasso_fft_obj, final_correlation_results = \
            b_fft_correlation.corr_between_main_and_combined_selected_vice_peaks_f_ifft(
                vice_extra_hz_f=[
                    1 / (364 * 24 * 3600),
                    1 / (91 * 24 * 3600),
                    1 / (28 * 24 * 3600),
                    1 / (7 * 24 * 3600),
                    1 / (1 * 24 * 3600),
                    1 / (0.5 * 24 * 3600),
                ],
                do_lasso_fitting_args={}
            )

        lasso_fft_plot = lasso_fft_obj.plot('1/half day', use_log=True, save_to_buffer=save_to_buffer)
        reconstructed_main_plot = time_series(
            x=x_axis_value, y=lasso_fft_obj(now_window_data.index)[0], tz=tz,
            y_label=f"Inverse FFT [{y_unit}]",
            **TIME_SERIES_PLOT_KWARGS,
            save_to_buffer=save_to_buffer
        )
        """
        DEBUG
        """
        # series(appliance, title='appliance')
        # series(main, title='main')
        # series(environment, title='environment')
        # series(final_correlation_results['Spearman'][-1].partly_combination_reconstructed,
        #        title='weired')
        # FFTProcessor(n_fft=2 ** 18, original_signal=main.values, sampling_period=this_data_set_freq,
        #              name='test').plot(considered_frequency_units='1/half day',
        #                                overridden_plot_x_lim=ax[0].get_xlim(), plot_log=True)

        final_correlation_reconstructed = final_correlation_results['Spearman'][-1].partly_combination_reconstructed
        final_correlation_plot = time_series(
            x=x_axis_value, y=final_correlation_reconstructed, tz=tz,
            y_label=f"Components of Main (Most\nNegatively Correlated with\n{environment_name.title()}) [W]",
            **TIME_SERIES_PLOT_KWARGS,
            save_to_buffer=save_to_buffer
        )
        print(f"original_corr = {b_fft_correlation.original_corr.correlation}, "
              f"op_corr = {final_correlation_results['Spearman'][-1].corr_value}")
        # Writing
        if save_to_buffer:
            document.add_heading(x_axis_value[0].strftime('%y-%b-%d %a'), level=1)
            to_plot_list_buffer_list = [main_plot, appliance_plot, environment_plot, reconstructed_main_plot]
            to_plot_list_buffer_list.remove(None)
            for this_plot in to_plot_list_buffer_list:
                p = document.add_paragraph()
                p.add_run().add_picture(this_plot, width=Cm(14))
            p = document.add_paragraph()
            p.add_run().add_picture(lasso_fft_plot[0], width=Cm(7))
            p.add_run().add_picture(lasso_fft_plot[1], width=Cm(7))
            p = document.add_paragraph()
            p.add_run().add_picture(final_correlation_plot, width=Cm(14))
            document.add_page_break()
    document.save(f'.\\{now_data_set.obj_name}_fft_and_corr.docx')


def get_freq_corr_hst_and_sum_of_sel_freq(name: str, feature: str):
    assert name in {"AMPDS2", "UKDALE", "JOHN"}
    assert feature in {'temperature', 'solar'}

    if name == "JOHN":
        get_data_obj = get_data_func(True)()
    else:
        raise NotImplementedError

    @load_exist_pkl_file_otherwise_run_and_save(
        Path(project_path_ / r'Data\Results\Energies_paper\freq_corr_hst_and_sum_of_sel_freq' / f'{name}_{feature}.pkl')
    )
    def func():
        _freq_corr_hst = []
        _sum_of_sel_freq = []
        for i in tqdm(range(len(get_data_obj))):
            # for i in tqdm(range(3)):
            data = get_data_obj.get(i)
            power_ifft = lasso_fft_for_one_feature(data, 'active power')
            other_ifft = lasso_fft_for_one_feature(data, feature)
            # Calculate correlations
            now_freq_corr_hst = cal_freq_correlations(data, power_ifft, other_ifft, feature)
            now_sum_of_sel_freq = sum_up_selected_freq_correlations(power_ifft, other_ifft, now_freq_corr_hst)
            _freq_corr_hst.append(now_freq_corr_hst)
            _sum_of_sel_freq.append(now_sum_of_sel_freq)
        return _freq_corr_hst, _sum_of_sel_freq

    freq_corr_hst, sum_of_sel_freq = func()
    return get_data_obj, freq_corr_hst, sum_of_sel_freq


def make_report(name: str, feature: str):
    get_data_obj, freq_corr_hst, sum_of_sel_freq = get_freq_corr_hst_and_sum_of_sel_freq(name, feature)
    if feature == 'temperature':
        y_label_2 = 'Temperature [\u00B0C]'
        plot_task_2 = 'temperature'
    else:
        y_label_2 = 'Irradiance [W/$\mathregular{m}^2$]'
        plot_task_2 = 'solar'

    def make_legend(_ax1, _ax2):
        lines, labels = now_ax1.get_legend_handles_labels()
        lines2, labels2 = now_ax2.get_legend_handles_labels()
        now_ax1.get_legend().remove()
        now_ax2.legend(lines + lines2, labels + labels2, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                       ncol=4, mode="expand", borderaxespad=0., prop={'size': 10})

    document = docx_document_template_to_collect_figures()
    for i in tqdm(range(len(get_data_obj))):
        if name == 'JOHN' and i < 365 * 4 + 1:
            continue
        # for i in tqdm(range(3)):

        data = get_data_obj.get(i)
        power_ifft, power_fourier_obj = lasso_fft_for_one_feature(data, 'active power', True)
        other_ifft, other_fourier_obj = lasso_fft_for_one_feature(data, feature, True)
        # produce figures
        y_label_1 = f"Active Power [{'MW' if name == 'JOHN' else 'W'}]"

        # %%
        for plot_i, period in enumerate(PERIOD):
            if period in {'half day', 'day'}:
                x_ticks = (np.arange(0, 25, 3) * 2, np.arange(0, 25, 3))
                x_label = 'Rank of Recordings [per hour]'
            elif period in {'2days'}:
                x_ticks = (np.arange(0, 49, 6) * 2, np.arange(0, 49, 6))
                x_label = 'Rank of Recordings [per hour]'
            elif period in {'5days'}:
                x_ticks = (np.arange(0, 121, 24) * 2, np.arange(0, 121, 24))
                x_label = 'Rank of Recordings [per hour]'
            elif period in {'week'}:
                x_ticks = (np.arange(0, 169, 24) * 2, np.arange(0, 7.1, 1).astype(int))
                x_label = 'Rank of Recordings [per day]'
            elif period in {'month'}:
                x_ticks = (np.arange(0, 24 * 28 + 1, 24 * 7) * 2, np.arange(0, 29, 7).astype(int))
                x_label = 'Rank of Recordings [per day]'
            elif period in {'season'}:
                x_ticks = (np.arange(0, 24 * 91 + 1, 24 * 7) * 2, np.arange(0, 14, 1).astype(int))
                x_label = 'Rank of Recordings [per 7 days]'
            else:
                x_ticks = (np.arange(0, 24 * 364 + 1, 24 * 28) * 2, np.arange(0, 14, 1).astype(int))
                x_label = 'Rank of Recordings [per 28 days]'
            now_ax1 = series(data[period]['active power'].values, color='royalblue', label='Actl-P', zorder=2)
            now_ax1 = series(power_ifft[period], ax=now_ax1, color='red', linestyle='--', label='IFT-P', zorder=4,
                             x_lim=(-0.025 * len(power_ifft[period]), 1.025 * len(power_ifft[period])),
                             x_label=x_label, y_label=y_label_1,
                             x_ticks=x_ticks)
            now_ax2 = now_ax1.twinx()  # instantiate a second axes that shares the same x-axis
            now_ax2.set_ylabel(y_label_2, fontdict={'size': 10})  # we already handled the x-label with ax1
            series(data[period][plot_task_2].values, color='black', label=f'Actl-{y_label_2[0]}', ax=now_ax2, zorder=1,
                   alpha=0.9, linestyle=':')
            series(other_ifft[period], color='lime', label=f'IFT-{y_label_2[0]}', ax=now_ax2, zorder=3, linestyle='-.')
            plt.grid(False)
            # ask matplotlib for the plotted objects and their labels
            make_legend(now_ax1, now_ax2)
            plt.savefig(f'./temp/{feature}/{period}' + '.png', format='png', dpi=300)
            plt.close()

        # %% freq and mag
        power_fourier_obj.plot('1/day', y_label_unit=re.findall(r'\[(.*)]', y_label_1)[0],
                               extra_zoom_in_x_idx=[5, 6, 7, 8],
                               extra_zoom_in_x_ticks=['Weekly', 'Monthly', 'Seasonally', 'Yearly'],
                               save_freq_full_path=f"./temp/{feature}/power_freq.png",
                               save_mag_full_path=f"./temp/{feature}/power_mag.png")
        other_fourier_obj.plot('1/day', y_label_unit=re.findall(r'\[(.*)]', y_label_2)[0],
                               extra_zoom_in_x_idx=[5, 6, 7, 8],
                               extra_zoom_in_x_ticks=['Weekly', 'Monthly', 'Seasonally', 'Yearly'],
                               save_freq_full_path=f"./temp/{feature}/{plot_task_2}_freq.png",
                               save_mag_full_path=f"./temp/{feature}/{plot_task_2}_mag.png")
        # %% comb_naive
        now_ax1 = series(data['day']['active power'].values, color='royalblue', label='Actl-P', zorder=2)
        now_ax1 = series(power_ifft['all'], ax=now_ax1, color='red', linestyle='--', label='IFT-P', zorder=4,
                         x_lim=(-0.025 * len(power_ifft['all']), 1.025 * len(power_ifft['all'])),
                         x_ticks=(np.arange(0, 25, 3) * 2, np.arange(0, 25, 3)),
                         x_label='Rank of Recordings [per hour]',
                         y_label=y_label_1)
        now_ax2 = now_ax1.twinx()  # instantiate a second axes that shares the same x-axis
        now_ax2.set_ylabel(y_label_2, fontdict={'size': 10})  # we already handled the x-label with ax1
        series(data['day'][plot_task_2].values, color='black', label=f'Actl-{y_label_2[0]}', ax=now_ax2, zorder=1,
               alpha=0.9, linestyle=':')
        series(other_ifft['all'], color='lime', label=f'IFT-{y_label_2[0]}', ax=now_ax2, zorder=3, linestyle='-.')
        plt.grid(False)
        # ask matplotlib for the plotted objects and their labels
        make_legend(now_ax1, now_ax2)
        plt.savefig(f'./temp/{feature}/comb_naive' + '.png', format='png', dpi=300)
        plt.close()

        # %% comb_corr
        now_ax1 = series(sum_of_sel_freq[i][0], color='red', linestyle='--', label='IFT-P', zorder=4,
                         x_lim=(-0.025 * len(power_ifft['all']), 1.025 * len(power_ifft['all'])),
                         x_ticks=(np.arange(0, 25, 3) * 2, np.arange(0, 25, 3)),
                         x_label='Rank of Recordings [per hour]',
                         y_label='Normalised ' + y_label_1[:y_label_1.find('[') - 1])
        now_ax2 = now_ax1.twinx()  # instantiate a second axes that shares the same x-axis
        now_ax2.set_ylabel('Normalised ' + y_label_2[:y_label_2.find('[') - 1], fontdict={'size': 10})
        series(sum_of_sel_freq[i][1], color='lime', label=f'IFT-{y_label_2[0]}', ax=now_ax2, zorder=3, linestyle='-.')
        plt.grid(False)
        # ask matplotlib for the plotted objects and their labels
        make_legend(now_ax1, now_ax2)
        plt.savefig(f'./temp/{feature}/comb_corr' + '.png', format='png', dpi=300)
        plt.close()

        # Make report
        document.add_heading(data['day'].index[0].strftime('%y-%b-%d %a') + ', ' +
                             f"Holiday = {'True' if data['day']['holiday'][0] == 1 else 'False'}", level=1)
        for plot_i, period in enumerate(PERIOD):
            p = document.add_paragraph()
            p.add_run(f"Frequency components = {period}")
            p = document.add_paragraph()
            p.add_run().add_picture(f"./temp/{feature}/{period}.png", width=Cm(7))
            p = document.add_paragraph()
            p.add_run(f"Correlation between actual recordings (day length) = "
                      f"{freq_corr_hst[i][period]['day_length_raw']:.4f}\n")
            p.add_run(f"Correlation between IFT results (day length) = "
                      f"{freq_corr_hst[i][period]['day_length_freq']:.4f}\n")
            p.add_run(f"Correlation between actual recordings (full length) = "
                      f"{freq_corr_hst[i][period]['full_length_raw']:.4f}\n")
            p.add_run(f"Correlation between IFT results (full length) = "
                      f"{freq_corr_hst[i][period]['full_length_freq']:.4f}\n")
        #
        p = document.add_paragraph()
        p.add_run(f"Frequency magnitude and phase angle plots for power")
        p = document.add_paragraph()
        p.add_run().add_picture(f"./temp/{feature}/power_freq.png", width=Cm(7))
        p.add_run().add_picture(f"./temp/{feature}/power_mag.png", width=Cm(7))
        #
        p = document.add_paragraph()
        p.add_run(f"Frequency magnitude and phase angle plots for {plot_task_2}")
        p = document.add_paragraph()
        p.add_run().add_picture(f"./temp/{feature}/{plot_task_2}_freq.png", width=Cm(7))
        p.add_run().add_picture(f"./temp/{feature}/{plot_task_2}_mag.png", width=Cm(7))
        #
        p = document.add_paragraph()
        p.add_run(f"Combination using all components and only selected components")
        p = document.add_paragraph()
        p.add_run().add_picture(f"./temp/{feature}/comb_naive.png", width=Cm(7))
        p.add_run().add_picture(f"./temp/{feature}/comb_corr.png", width=Cm(7))
        #
        p = document.add_paragraph()
        p.add_run(f"In the left figure, correlation between actual recordings is "
                  f"{freq_corr_hst[i]['day']['day_length_raw']:.4f}\n")
        p.add_run(f"In the left figure, correlation between IFT results is"
                  f" {get_spearman_corr(other_ifft['all'], power_ifft['all']):.4f}\n")
        p.add_run(
            f"In the right figure, the selected periods are"
            f" {[key for key, value in freq_corr_hst[i].items() if value['day_length_raw'] > value['day_length_freq']]}"
        )
        p.add_run(f"In the right figure, correlation between IFT results is"
                  f" {get_spearman_corr(*sum_of_sel_freq[i]):.4f}\n")

        #
        document.add_page_break()
    document.save(project_path_ / r'Data\Results\Energies_paper\freq_corr_hst_and_sum_of_sel_freq' /
                  f'{name}_{feature}_fft_and_corr.docx')


if __name__ == "__main__":
    pass
    get_freq_corr_hst_and_sum_of_sel_freq("AMPDS2", "temperature")
    # get_freq_corr_hst_and_sum_of_sel_freq("JOHN", 'temperature')
    # make_report("JOHN", 'solar')
