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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM
from prepare_datasets import ScotlandLongerDataset, NILMDataSet, ampds2_dataset_full_df
from Regression_Analysis.DeepLearning_Class import TensorFlowCovBiLSTMEncoder, TensorFlowLSTMDecoder, GradientsAnalyser
import time
import re
import os
import inspect
import scipy
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
import kerastuner as kt

setlocale(LC_ALL, "en_US")
tf.keras.backend.set_floatx("float32")
remove_win10_max_path_limit()

BATCH_SIZE = 1000
# %% Load all data sets used in the paper
results_root_path = project_path_ / r"Data\Results\Energies_paper"
DATA_PREPARE_PATH = results_root_path / "Data_preparation"
MODEL_PATH = results_root_path / "NN_model"
# Ampds2
AMPDS2_DATA_600_HEATING = {
    'training': NILMDataSet(name='Ampds2_training', resolution=600, appliance='heating',
                            transformation_args_folder_path=DATA_PREPARE_PATH / "Ampds2_600_heating"),
    'test': NILMDataSet(name='Ampds2_test', resolution=600, appliance='heating',
                        transformation_args_folder_path=DATA_PREPARE_PATH / "Ampds2_600_heating"),
}
# Turkey apartment
TURKEY_APARTMENT_DATA_3600_LIGHTING = {
    'training': NILMDataSet(name='Turkey_apartment_training', appliance='lighting',
                            transformation_args_folder_path=DATA_PREPARE_PATH / "Turkey_apartment_3600_lighting"),
    'test': NILMDataSet(name='Turkey_apartment_test', appliance='lighting',
                        transformation_args_folder_path=DATA_PREPARE_PATH / "Turkey_apartment_3600_lighting"),
}

# Turkey detached house
TURKEY_HOUSE_DATA_3600_LIGHTING = {
    'training': NILMDataSet(name='Turkey_Detached House_training', appliance='lighting',
                            transformation_args_folder_path=DATA_PREPARE_PATH / "Turkey_Detached House_3600_lighting"),
    'test': NILMDataSet(name='Turkey_Detached House_test', appliance='lighting',
                        transformation_args_folder_path=DATA_PREPARE_PATH / "Turkey_Detached House_3600_lighting"),
}

# UK DALE
UK_DALE_DATA_600_HEATING = {
    'training': NILMDataSet(name='UKDALE_training', resolution=600, appliance='heating',
                            transformation_args_folder_path=DATA_PREPARE_PATH / "UKDALE_600_heating"),
    'test': NILMDataSet(name='UKDALE_test', resolution=600, appliance='heating',
                        transformation_args_folder_path=DATA_PREPARE_PATH / "UKDALE_600_heating"),
}
UK_DALE_DATA_3600_LIGHTING = {
    'training': NILMDataSet(name='UKDALE_training', resolution=3600, appliance='lighting',
                            transformation_args_folder_path=DATA_PREPARE_PATH / "UKDALE_3600_lighting"),
    'test': NILMDataSet(name='UKDALE_test', resolution=3600, appliance='lighting',
                        transformation_args_folder_path=DATA_PREPARE_PATH / "UKDALE_3600_lighting"),
}


def lasso_fft_and_correlation(task='explore', *, data_set=None, save_to_buffer=True):
    if data_set is None:
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
                'temperature': 'environmental temperature',
                'active power': 'main',
            },
            axis=1
        )
        john_data['heating'] = np.full(john_data.shape[0], 0)
        data_set = john_data

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


def stft(arr: ndarray, frame_length=71, frame_step=1, fft_length=1024):
    # tt = 1
    # stft_f = []
    # stft_dp = []
    # i = 0
    # while i + frame_length <= arr.__len__():
    #     now_signal = arr[i:i + frame_length]
    #     now_fft = np.abs(np.fft.fft(now_signal, fft_length)[:fft_length // 2 + 1])
    #     stft_f.append(now_fft)
    #
    #     analytic_signal = hilbert(now_signal, fft_length)
    #     instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    #     instantaneous_frequency = (np.diff(instantaneous_phase) /
    #                                (2.0 * np.pi))
    #     stft_dp.append(instantaneous_frequency)
    #     i += frame_step
    target = tf.signal.stft(arr, frame_length=frame_length, frame_step=frame_step,
                            fft_length=fft_length, window_fn=None).numpy()
    amp = np.log(np.abs(target))
    phase = np.angle(target)

    amp_norm = (amp - np.min(amp)) / (np.max(amp) - np.min(amp))
    phase_norm = (phase - np.min(phase)) / (np.max(phase) - np.min(phase))
    # padding =_=
    padding_size = arr.shape[0] - amp_norm.shape[0]
    amp_norm = np.pad(amp_norm, [(0, padding_size), (0, 0)])
    phase_norm = np.pad(phase_norm, [(0, padding_size), (0, 0)])
    return amp_norm, phase_norm, padding_size


def plot_stft(stft_f: ndarray = None, stft_p: ndarray = None):
    # TODO
    if stft_f is not None:
        pass
    log_spec = np.log(stft_f).T
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    x = np.linspace(0, np.size(stft_f), num=width, dtype=int)
    y = range(height)
    fig, axes = plt.subplots(2, figsize=(12, 8))

    axes[1].pcolormesh(x, y, log_spec)
    plt.show()


def stft_for_windowed_dataset_ans(windowed_dataset_ans):
    stft_col_idx = windowed_dataset_ans[2].columns.__getattribute__("get_locs")(["mains"])[0]
    ans = {
        "stft_f": [],
        "stft_p": [],
        "stft_padding": 0,
        "x": [],
        "y": []
    }
    for old_x, old_y in windowed_dataset_ans[0]:
        mains = old_x.numpy()[0, :, stft_col_idx]
        # stft
        stft_f, stft_p, stft_padding = stft(mains)
        ans['stft_f'].append(stft_f)
        ans['stft_p'].append(stft_p)
        ans['stft_padding'] = stft_padding
        # other_x
        # ans['x'].append(np.concatenate((old_x.numpy()[0, :, :stft_col_idx],
        #                                 old_x.numpy()[0, :, stft_col_idx + 1:]), axis=-1))
        ans['x'].append(old_x.numpy()[0, ...])
        # y
        ans['y'].append(old_y.numpy()[0, ...])

    return ans


def load_nn_data(name: str):
    assert name in {"AMPDS2_600"}
    x_window_length = datetime.timedelta(days=1)
    y_window_length = datetime.timedelta(days=1)
    x_y_start_index_diff = datetime.timedelta(0)
    window_shift = datetime.timedelta(days=1)
    kwargs = {
        "x_window_length": x_window_length,
        "y_window_length": y_window_length,
        "x_y_start_index_diff": x_y_start_index_diff,
        "window_shift": window_shift,
        "batch_size": 1
    }

    if name == "AMPDS2_600":
        dataset = AMPDS2_DATA_600_HEATING
    else:
        raise

    training_set = stft_for_windowed_dataset_ans(dataset['training'].windowed_dataset(**kwargs))
    test_set = stft_for_windowed_dataset_ans(dataset['test'].windowed_dataset(**kwargs))

    return training_set, test_set


def make_model(training_or_data_set):
    x_shape = training_or_data_set['x'][0].shape
    y_shape = training_or_data_set['y'][0].shape
    stft_f_shape = training_or_data_set['stft_f'][0].shape
    stft_p_shape = training_or_data_set['stft_p'][0].shape
    assert stft_f_shape == stft_p_shape

    def make_individual_layers(input_shape, input_layer_name: str):
        individual_model_layer_1 = tf.keras.Input(shape=input_shape, name=input_layer_name)
        cov1d_input = individual_model_layer_1
        if input_layer_name != 'x':
            individual_model_layer_crop = tf.keras.layers.Cropping1D(
                (0, training_or_data_set['stft_padding']))(individual_model_layer_1)
            cov1d_input = individual_model_layer_crop
        individual_model_layer_2 = tf.keras.layers.Conv1D(filters=10,
                                                          kernel_size=3,
                                                          padding="same",
                                                          activation="relu")(cov1d_input)
        if input_layer_name != 'x':
            individual_model_layer_3 = tf.keras.layers.MaxPooling1D(pool_size=2,
                                                                    strides=2,
                                                                    padding='valid')(individual_model_layer_2)
            individual_model_layer_4 = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=False))(individual_model_layer_3)
            individual_model_layer_5 = tf.keras.layers.Dropout(0.25)(individual_model_layer_4)
            individual_model_layer_6 = tf.keras.layers.RepeatVector(y_shape[0])(individual_model_layer_5)
            individual_model_layer_7 = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=True))(individual_model_layer_6)
        else:
            individual_model_layer_5 = tf.keras.layers.Dropout(0.25)(individual_model_layer_2)
            individual_model_layer_7 = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=True))(individual_model_layer_5)
        individual_model_layer_8 = tf.keras.layers.Dropout(0.25)(individual_model_layer_7)
        return individual_model_layer_1, individual_model_layer_8

    common_model_layer_input, common_model_layer_output = make_individual_layers(x_shape, 'x')
    stft_f_model_layer_input, stft_f_model_layer_output = make_individual_layers(stft_f_shape, 'stft_f')
    stft_p_model_layer_input, stft_p_model_layer_output = make_individual_layers(stft_p_shape, 'stft_p')

    # %% Concatenate
    main_model_layer_1 = tf.keras.layers.concatenate([common_model_layer_output,
                                                      stft_f_model_layer_output,
                                                      stft_p_model_layer_output])
    main_model_layer_2 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True))(main_model_layer_1)
    main_model_layer_3 = tf.keras.layers.Dropout(0.25)(main_model_layer_2)
    main_model_layer_4 = tf.keras.layers.Dense(y_shape[1], name="main_model_output")(main_model_layer_3)
    main_model = tf.keras.Model(
        inputs=[common_model_layer_input, stft_f_model_layer_input, stft_p_model_layer_input],
        outputs=[main_model_layer_4]
    )
    return main_model


def train_model(name: str, continue_training: bool = False):
    training_set, test_set = load_nn_data(name)
    save_folder_path = MODEL_PATH / name
    try_to_find_folder_path_otherwise_make_one(save_folder_path)

    model = make_model(training_set)
    model.summary()

    class SaveCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 1000 == 0:
                model.save_weights(save_folder_path / fr'model_epoch_{epoch}.h5')

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss={'main_model_output': keras.losses.mse},
                  metrics=['mae'])
    if continue_training:
        model.load_weights(save_folder_path / fr'continue.h5')

    model_x = {'x': tf.convert_to_tensor(np.array(training_set['x']), dtype=tf.float32),
               'stft_f': tf.convert_to_tensor(np.array(training_set['stft_f']), dtype=tf.float32),
               'stft_p': tf.convert_to_tensor(np.array(training_set['stft_p']), dtype=tf.float32)}
    model_y = {'main_model_output': tf.convert_to_tensor(np.array(training_set['y']), dtype=tf.float32)}
    validation_data = (
        {'x': tf.convert_to_tensor(np.array(test_set['x']), dtype=tf.float32),
         'stft_f': tf.convert_to_tensor(np.array(test_set['stft_f']), dtype=tf.float32),
         'stft_p': tf.convert_to_tensor(np.array(test_set['stft_p']), dtype=tf.float32)},
        {'main_model_output': tf.convert_to_tensor(np.array(test_set['y']), dtype=tf.float32)}
    )
    history = model.fit(x=model_x,
                        y=model_y,
                        verbose=2, epochs=100000,
                        validation_data=validation_data,
                        callbacks=[SaveCallback()],
                        validation_freq=50,
                        batch_size=BATCH_SIZE)
    model.save_weights(save_folder_path / 'final.h5')
    return history


def test_model(name: str):
    training_set, test_set = load_nn_data(name)
    save_folder_path = MODEL_PATH / name

    # %% Load model
    model = make_model(test_set)
    model.load_weights(save_folder_path / 'final.h5')

    tt = 1
    validation_data = (
        {'x': np.array(test_set['x']),
         'stft_f': np.array(test_set['stft_f']),
         'stft_p': np.array(test_set['stft_p'])},
        {'main_model_output': np.array(test_set['y'])}
    )
    y_pred = model.predict(validation_data[0])
    ii = 0
    for i in range(ii, ii + 10):
        ax = series(y_pred[i, :, 0])
        ax = series(validation_data[1]['main_model_output'][i, :, 0], ax=ax)

    """
    # %% Prepare test set
    test_dataset = test_dataset['test'][test_date_time_range]  # type: NILMDataSet
    test_set_origin = test_dataset.data
    test_set_nn, time_stamp_ndarray, predictor_names, dependant_names = test_dataset.windowed_dataset(
        datetime.timedelta(days=1), batch_size=batch_size)

    # %% Test
    gradients = np.full((*time_stamp_ndarray.shape, predictor_names.__len__()), np.nan)
    predictions = np.full((*time_stamp_ndarray.shape, dependant_names.__len__()), np.nan)
    for i, (x, y) in enumerate(test_set_nn):
        # Predict and calculate gradient
        x_var = tf.Variable(x, dtype="float32")
        with tf.GradientTape() as tape:
            this_prediction_nn = model(x_var)
        this_gradients = tape.gradient(this_prediction_nn, x_var).cpu().numpy()
        gradients[i * batch_size:(i + 1) * batch_size] = this_gradients
        # Inverse transform
        this_prediction = test_dataset.inverse_transform(this_prediction_nn, dependant_names)
        predictions[i * batch_size:(i + 1) * batch_size] = this_prediction
    # %% Analyse gradient
    gradient_analyser_obj = GradientsAnalyser(gradients, predictor_names)
    new_gradients, new_predictor_names = gradient_analyser_obj.aggregate_over_all_samples(apply_abs_op=True)
    gradient_analyser_obj.plot(new_gradients, new_predictor_names, plot_individual_steps=False, log_scale=True)

    # ax = series(model.predict(list(test_set.as_numpy_iterator())[0][0]).flatten())
    # ax = series(this_output.flatten(), ax=ax)
    """


if __name__ == '__main__':
    pass
    # tt, _, _ = AMPDS2_DATA_600_HEATING['training'].windowed_dataset(datetime.timedelta(days=1), batch_size=99)
    # bb = iter(tt)
    # data_list = [AMPDS2_DATA_600_HEATING,
    #              UK_DALE_DATA_600_HEATING]
    # for this_data, this_path in zip(data_list, path_list):
    #     if not try_to_find_file(this_path / r'fitted_model.h5'):
    #         fitted_model = train_model(this_data,
    #                                                     this_path,
    #                                                     batch_size=200, epochs=5000)
    # fitted_model = train_model(TURKEY_APARTMENT_DATA_3600_LIGHTING,
    #                            batch_size=200, epochs=5000, task='')

    # fitted_model = train_model(TURKEY_HOUSE_DATA_3600_LIGHTING,
    #                                             TURKEY_HOUSE_PATH_3600_LIGHTING,
    #                                             batch_size=200, epochs=5000)

    # fitted_model = train_model("AMPDS2_600")
    # test_model("AMPDS2_600")
    # test_model(UK_DALE_DATA_600_HEATING, UK_DALE_DATA_600_HEATING,
    #            test_date_time_range=(datetime.datetime(2016, 4, 25), datetime.datetime(2017, 4, 25)))
