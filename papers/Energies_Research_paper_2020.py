from Ploting.fast_plot_Func import *
import numpy as np
import pandas as pd
from numpy import ndarray
from TimeSeries_Class import TimeSeries, WindowedTimeSeries
from Correlation_Modeling.utils import CorrelationAnalyser
from Correlation_Modeling.FFTCorrelation_Class import BivariateFFTCorrelation
from project_utils import *
from File_Management.path_and_file_management_Func import *
from File_Management.load_save_Func import *
from PhysicalInstance_Class import *
from FFT_Class import FFTProcessor, LASSOFFTProcessor
from Time_Processing.format_convert_Func import np_datetime64_to_datetime
from locale import setlocale, LC_ALL
from Writting import docx_document_template_to_collect_figures

setlocale(LC_ALL, "en_US")

remove_win10_max_path_limit()

# %% Load all data sets used in the paper
# Ampds2
raw_data_path = project_path_ / r"Data\Raw\for_Energies_Research_paper_2020"
AMPDS2_DATA = PhysicalInstanceDataFrame(load_pkl_file(raw_data_path / 'Ampds2.pkl'),
                                        obj_name='Ampds2', predictor_names=(), dependant_names=())
AMPDS2_DATA = AMPDS2_DATA.rename(
    {
        'Mains': 'main',
        'HPE': 'heating',
        'Temp (C)': 'environmental temperature'
    },
    axis=1
)
# Turkey apartment
TURKEY_APARTMENT_DATA = PhysicalInstanceDataFrame(load_pkl_file(raw_data_path / 'Turkey.pkl')['Apartment'],
                                                  obj_name='Turkey apartment', predictor_names=(), dependant_names=())
TURKEY_APARTMENT_DATA = TURKEY_APARTMENT_DATA.rename(
    {
        'temperature': 'environmental temperature',
        'radiation_surface': 'solar irradiance'
    },
    axis=1
)
# Turkey detached house
TURKEY_HOUSE_DATA = PhysicalInstanceDataFrame(load_pkl_file(raw_data_path / 'Turkey.pkl')['Detached House'],
                                              obj_name='Turkey detached house', predictor_names=(), dependant_names=())
TURKEY_HOUSE_DATA = TURKEY_HOUSE_DATA.rename(
    {
        'temperature': 'environmental temperature',
        'radiation_surface': 'solar irradiance'
    },
    axis=1
)
# UK DALE
UKDALE_DATA = PhysicalInstanceDataFrame(load_pkl_file(raw_data_path / 'UKDALE.pkl'),
                                        obj_name='UK DALE', predictor_names=(), dependant_names=())


def lasso_fft_and_correlation(task='explore'):
    assert task in ('explore', 'load results')


if __name__ == '__main__':
    this_data_set = copy.deepcopy(AMPDS2_DATA)
    this_data_set = this_data_set.iloc[24 * 185:24 * 187]
    save_to_buffer = True
    # Reindex to make sure it's index is has equal diff
    tz = this_data_set.index.__getattribute__('tz')
    first_index, last_index = this_data_set.index[0], this_data_set.index[-1]
    this_data_set_freq = np.min(np.diff(this_data_set.index))
    this_data_set = this_data_set.reindex(pd.DatetimeIndex(
        pd.date_range(
            start=datetime.datetime(first_index.year, first_index.month, first_index.day),
            end=datetime.datetime(last_index.year, last_index.month, last_index.day) + datetime.timedelta(1),
            freq=this_data_set_freq,
            tz=tz
        )
    ))
    # %% window length = 1D
    main_plot_list = []
    appliance_plot_list = []
    environment_plot_list = []
    lasso_fft_plot_list = []
    final_correlation_plot_list = []
    this_data_set_iter = iter(WindowedTimeSeries(this_data_set, window_length=datetime.timedelta(days=1)))
    document = docx_document_template_to_collect_figures()
    for i, this_window_data in enumerate(this_data_set_iter):
        x_axis_value = [np_datetime64_to_datetime(x, tz=tz) for x in this_window_data.index.values]

        # %% Obtain the required dimensions of data
        # Main
        main = this_window_data['main']
        main_plot = time_series(x=x_axis_value, y=main.values, tz=tz,
                                y_label='Main Active Power [W]', **TIME_SERIES_PLOT_KWARGS,
                                save_to_buffer=save_to_buffer)
        main_plot_list.append(main_plot)

        # depending on the dataset, heating or lighting will be analysed
        # Turkey data set only analyses lighting
        if 'Turkey' in this_data_set.obj_name:
            appliance_name = 'lighting'
            appliance_plot_y_label = 'Lighting Active Power [W]'

            environment_name = 'solar irradiance'
            environment_plot_y_label = r'Solar Irradiance [W/$\mathregular{m}^2$]'
        # Ampds2 data set only analyses heating
        else:
            appliance_name = 'heating'
            appliance_plot_y_label = 'Heating Active Power [W]'

            environment_name = 'environmental temperature'
            environment_plot_y_label = 'Environmental Temperature [\u00B0C]'

        appliance = this_window_data[appliance_name]
        environment = this_window_data[environment_name]

        appliance_plot = time_series(x=x_axis_value, y=appliance.values, tz=tz,
                                     y_label=appliance_plot_y_label, **TIME_SERIES_PLOT_KWARGS,
                                     save_to_buffer=save_to_buffer)
        appliance_plot_list.append(appliance_plot)
        environment_plot = time_series(x=x_axis_value, y=environment.values, tz=tz,
                                       y_label=environment_plot_y_label, **TIME_SERIES_PLOT_KWARGS,
                                       save_to_buffer=save_to_buffer)
        environment_plot_list.append(environment_plot)

        # %% LASSO-FFT and Correlation
        b_fft_correlation = BivariateFFTCorrelation(
            n_fft=2 ** 20,
            considered_frequency_unit='1/half day',
            _time_series=this_window_data[[environment_name, 'main']],
            correlation_func=('Spearman',),
            main_considered_peaks_index=list(range(1, 7)),
            vice_considered_peaks_index=list(range(1, 7)),
            vice_find_peaks_args={
                'scipy_signal_find_peaks_args': {

                }
            }
        )
        lasso_fft_obj, final_correlation_results = \
            b_fft_correlation.corr_between_main_and_combined_selected_vice_peaks_f_ifft(
                vice_extra_hz_f=[1 / (365.25 * 24 * 3600),
                                 1 / (7 * 24 * 3600)],
                do_lasso_fitting_args={}
            )

        lasso_fft_plot = lasso_fft_obj.plot('1/half day', save_to_buffer=save_to_buffer)
        lasso_fft_plot_list.append(lasso_fft_plot)
        """
        DEBUG
        """
        # series(appliance, title='appliance')
        # series(main, title='main')
        # series(environment, title='environment')
        # series(final_correlation_results['Spearman'][-1].partly_combination_reconstructed,
        #        title='weired')
        # FFTProcessor(n_fft=2 ** 18, original_signal=main.values, sampling_period=this_data_set_freq.seconds,
        #              name='test').plot(considered_frequency_units='1/half day',
        #                                overridden_plot_x_lim=ax[0].get_xlim(), plot_log=True)

        final_correlation_reconstructed = final_correlation_results['Spearman'][-1].partly_combination_reconstructed
        final_correlation_plot = time_series(
            x=x_axis_value, y=final_correlation_reconstructed, tz=tz,
            y_label=f"Components of Main \n(Most Negatively Correlated with\n{environment_name.title()}) [W]",
            **TIME_SERIES_PLOT_KWARGS,
            save_to_buffer=save_to_buffer
        )
        final_correlation_plot_list.append(final_correlation_plot)
    # Writing
    for i, (main_plot, appliance_plot, environment_plot, lasso_fft_plot_list, final_correlation_plot) in enumerate(
            zip((main_plot_list,
                 appliance_plot_list,
                 environment_plot_list,
                 lasso_fft_plot_list,
                 final_correlation_plot_list,))):
        pass
