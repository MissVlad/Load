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
from Writting import *
from tqdm import tqdm
from prepare_datasets import NILMDataSet, ampds2_dataset_full_df
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM
from prepare_datasets import ScotlandLongerDataset

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
# JOHN MV
john_data_set_date_start = datetime.datetime(year=2009, month=1, day=1)
# john_data_set_date_end = datetime.datetime(year=2009, month=1, day=3)
john_data_set_date_end = datetime.datetime(year=2010, month=1, day=1)
JOHN_DATA = ScotlandLongerDataset("John").dataset
JOHN_DATA = PhysicalInstanceDataFrame(
    JOHN_DATA[np.bitwise_and(JOHN_DATA.index >= john_data_set_date_start,
                             JOHN_DATA.index < john_data_set_date_end)],
    obj_name='John MV', predictor_names=(), dependant_names=()
)
JOHN_DATA = JOHN_DATA.rename(
    {
        'temperature': 'environmental temperature',
        'active power': 'main',
    },
    axis=1
)
JOHN_DATA['heating'] = np.full(JOHN_DATA.shape[0], 0)


def lasso_fft_and_correlation(task='explore', *, data_set, save_to_buffer=True):
    assert task in ('explore', 'load results')
    this_data_set = copy.deepcopy(data_set)
    # this_data_set = this_data_set.iloc[24 * 185:24 * 190 - 1]
    # Reindex to make sure it's index is has equal diff
    tz = this_data_set.index.__getattribute__('tz')
    first_index, last_index = this_data_set.index[0], this_data_set.index[-1]
    this_data_set_freq = int(np.min(np.diff(this_data_set.index)) / np.timedelta64(1, 's'))
    this_data_set = this_data_set.reindex(pd.DatetimeIndex(
        pd.date_range(
            start=datetime.datetime(first_index.year, first_index.month, first_index.day),
            end=datetime.datetime(last_index.year, last_index.month, last_index.day) + datetime.timedelta(1),
            freq=f"{this_data_set_freq}S",
            tz=tz
        )
    ))
    # %% window length = 1D
    document = docx_document_template_to_collect_figures()
    for i, this_window_data in enumerate(tqdm(WindowedTimeSeries(this_data_set,
                                                                 window_length=datetime.timedelta(days=1)))):

        if this_window_data.size == 0:
            continue
        # depending on the dataset, heating or lighting will be analysed
        # Turkey data set only analyses lighting
        if 'Turkey' in this_data_set.obj_name:
            appliance_name = 'lighting'
            appliance_plot_y_label = 'Lighting Active Power [W]'

            environment_name = 'solar irradiance'
            environment_plot_y_label = r'Solar Irradiance [W/$\mathregular{m}^2$]'
        # Ampds2 data set only analyses heating, other data sets are funny
        else:
            appliance_name = 'heating'
            appliance_plot_y_label = 'Heating Active Power [W]'

            environment_name = 'environmental temperature'
            environment_plot_y_label = 'Environmental Temperature [\u00B0C]'
        if np.any(np.isnan(this_window_data[['main', appliance_name, environment_name]])):
            continue

        x_axis_value = [np_datetime64_to_datetime(x, tz=tz) for x in this_window_data.index.values]

        # %% Obtain the required dimensions of data
        # Main
        main = this_window_data['main']
        if ('Ampds2' in this_data_set.obj_name) or ('Turkey' in this_data_set.obj_name):
            y_unit = "W"
        else:
            y_unit = "MW"
        main_plot = time_series(
            x=x_axis_value, y=main.values, tz=tz,
            y_label=f"Main Active Power [{y_unit}]",
            **TIME_SERIES_PLOT_KWARGS, save_to_buffer=save_to_buffer)

        appliance = this_window_data[appliance_name]
        environment = this_window_data[environment_name]

        if ("Ampds2" in this_data_set.obj_name) or ("Turkey" in this_data_set.obj_name):
            appliance_plot = time_series(x=x_axis_value, y=appliance.values, tz=tz,
                                         y_label=appliance_plot_y_label, **TIME_SERIES_PLOT_KWARGS,
                                         save_to_buffer=save_to_buffer)
        else:
            appliance_plot = None

        environment_plot = time_series(x=x_axis_value, y=environment.values, tz=tz,
                                       y_label=environment_plot_y_label, **TIME_SERIES_PLOT_KWARGS,
                                       save_to_buffer=save_to_buffer)

        # %% LASSO-FFT and Correlation
        b_fft_correlation = BivariateFFTCorrelation(
            n_fft=2 ** 22,
            considered_frequency_unit='1/half day',
            _time_series=this_window_data[[environment_name, 'main']],
            correlation_func=('Spearman',),
            main_considered_peaks_index=list(range(1, 11)),
            vice_considered_peaks_index=list(range(1, 11)),
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
        reconstructed_main_plot = time_series(
            x=x_axis_value, y=lasso_fft_obj(this_window_data.index)[0], tz=tz,
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
    document.save(f'.\\{this_data_set.obj_name}_lasso_fft_and_corr.docx')


def load_ampds2_data_set(resolution: int, mode: str = 'training'):
    assert mode in ('training', 'test')
    ampds2_dataset = ampds2_dataset_full_df(resolution=resolution)
    if mode == 'training':
        _slice = slice(0, int(len(ampds2_dataset) / 2))
    else:
        _slice = slice(int(len(ampds2_dataset) / 2), len(ampds2_dataset) + 1)

    ampds2_training = NILMDataSet(ampds2_dataset.iloc[_slice],
                                  name=f'ampds2_{mode}',
                                  cos_sin_transformed_col=('month',),
                                  one_hot_transformed_col=('dayofweek',),
                                  non_transformed_col=('summer_time', 'holiday'),
                                  min_max_transformed_col=('moon_phase', 'Temp (C)', 'Rel Hum (%)', 'Stn Press (kPa)',
                                                           'Mains', 'HPE'),
                                  dependant_cols=('HPE',))

    # ampds2_training.windowed_dataset(datetime.timedelta(days=1), batch_size=10)
    return ampds2_training


def model_layout(input_shape, output_dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=3,
                               strides=1, padding="causal",
                               activation="relu",
                               input_shape=input_shape),
        tf.keras.layers.Bidirectional(LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(LSTM(128, return_sequences=True)),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(output_dim),
    ])

    return model


if __name__ == '__main__':
    pass
    lasso_fft_and_correlation(task='explore', data_set=JOHN_DATA)
