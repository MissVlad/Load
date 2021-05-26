import datetime

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
from Writting import *
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM
from Regression_Analysis.DeepLearning_Class import TensorFlowCovBiLSTMEncoder, TensorFlowLSTMDecoder, GradientsAnalyser
import time
import re
import os
import inspect
import scipy
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
import kerastuner as kt
from paper_utils import *


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


def load_training_and_test_set(name: str):
    assert name in {"AMPDS2_600", "UKDALE_600"}
    x_window_length = datetime.timedelta(days=1)
    y_window_length = datetime.timedelta(days=1)
    x_y_start_index_diff = datetime.timedelta(0)
    kwargs = {
        "x_window_length": x_window_length,
        "y_window_length": y_window_length,
        "x_y_start_index_diff": x_y_start_index_diff,
        "batch_size": 1
    }

    if name == "AMPDS2_600":
        dataset = AMPDS2_DATA_600
    elif name == "UKDALE_600":
        dataset = UK_DALE_DATA_600
    else:
        raise

    training_set = stft_for_windowed_dataset_ans(dataset['training'].windowed_dataset(
        window_shift=datetime.timedelta(hours=12), **kwargs)
    )
    test_set = stft_for_windowed_dataset_ans(dataset['test'].windowed_dataset(
        window_shift=datetime.timedelta(days=1), **kwargs)
    )

    return training_set, test_set


def get_data_for_nn(name: str):
    training_set, test_set = load_training_and_test_set(name)

    model_x = {'x': tf.convert_to_tensor(np.array(training_set['x']), dtype=tf.float32),
               'stft_f': tf.convert_to_tensor(np.array(training_set['stft_f']), dtype=tf.float32),
               'stft_p': tf.convert_to_tensor(np.array(training_set['stft_p']), dtype=tf.float32)}
    model_y = {'main_model_output': tf.convert_to_tensor(np.array(training_set['y']), dtype=tf.float32)}
    test_data = (
        {'x': tf.convert_to_tensor(np.array(test_set['x']), dtype=tf.float32),
         'stft_f': tf.convert_to_tensor(np.array(test_set['stft_f']), dtype=tf.float32),
         'stft_p': tf.convert_to_tensor(np.array(test_set['stft_p']), dtype=tf.float32)},
        {'main_model_output': tf.convert_to_tensor(np.array(test_set['y']), dtype=tf.float32)}
    )
    return model_x, model_y, test_data


def make_hp_model(training_or_data_set, hp: Union[None, kt.HyperParameters], pressure_test):
    x_shape = training_or_data_set['x'][0].shape
    y_shape = training_or_data_set['y'][0].shape
    stft_f_shape = training_or_data_set['stft_f'][0].shape
    stft_p_shape = training_or_data_set['stft_p'][0].shape
    assert stft_f_shape == stft_p_shape

    # %% common_model
    # layer 1
    common_model_layer_1 = tf.keras.Input(shape=x_shape, name='x')
    # layer 2
    hp_common_model_layer_2_filter = hp.Int('hp_common_model_layer_2_filter',
                                            min_value=8,
                                            max_value=16,
                                            step=1) if not pressure_test else 16
    hp_common_model_layer_2_kernel_size = hp.Int('hp_common_model_layer_2_kernel_size',
                                                 min_value=3,
                                                 max_value=7,
                                                 step=2) if not pressure_test else 3
    common_model_layer_2 = tf.keras.layers.Conv1D(
        filters=hp_common_model_layer_2_filter,
        kernel_size=hp_common_model_layer_2_kernel_size,
        padding="same",
        activation="relu"
    )(common_model_layer_1)
    # layer 3
    hp_common_model_layer_3_rate = hp.Float('hp_common_model_layer_3_rate',
                                            min_value=0.05,
                                            max_value=0.5,
                                            step=0.05) if not pressure_test else 0.1
    common_model_layer_3 = tf.keras.layers.Dropout(
        hp_common_model_layer_3_rate
    )(common_model_layer_2)
    # layer 4
    hp_common_model_layer_4_units = hp.Int('hp_common_model_layer_4_units',
                                           min_value=48,
                                           max_value=128,
                                           step=8) if not pressure_test else 128
    common_model_layer_4 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hp_common_model_layer_4_units, return_sequences=True)
    )(common_model_layer_3)
    # layer 5
    hp_common_model_layer_5_rate = hp.Float('hp_common_model_layer_5_rate',
                                            min_value=0.05,
                                            max_value=0.5,
                                            step=0.05) if not pressure_test else 0.1
    common_model_layer_5 = tf.keras.layers.Dropout(
        hp_common_model_layer_5_rate
    )(common_model_layer_4)

    # %% stft_f_model and stft_p_model
    def make_stft_layers(input_shape, input_layer_name: str):
        # layer 1
        individual_model_layer_1 = tf.keras.Input(shape=input_shape, name=input_layer_name)
        individual_model_layer_crop = tf.keras.layers.Cropping1D(
            (0, training_or_data_set['stft_padding']))(individual_model_layer_1)
        # layer 2
        hp_individual_model_layer_2_filter = hp.Int(f'hp_{input_layer_name}_model_layer_2_filter',
                                                    min_value=8,
                                                    max_value=16,
                                                    step=1) if not pressure_test else 16
        hp_individual_model_layer_2_kernel_size = hp.Int(f'hp_{input_layer_name}_model_layer_2_kernel_size',
                                                         min_value=3,
                                                         max_value=7,
                                                         step=2) if not pressure_test else 3
        individual_model_layer_2 = tf.keras.layers.Conv1D(
            filters=hp_individual_model_layer_2_filter,
            kernel_size=hp_individual_model_layer_2_kernel_size,
            padding="same",
            activation="relu"
        )(individual_model_layer_crop)
        # layer 3
        hp_individual_model_layer_3_pool_size = hp.Int(f'hp_{input_layer_name}_model_layer_3_pool_size',
                                                       min_value=2,
                                                       max_value=5,
                                                       step=1) if not pressure_test else 2
        hp_individual_model_layer_3_strides = hp.Int(f'hp_{input_layer_name}_model_layer_3_strides',
                                                     min_value=1,
                                                     max_value=5,
                                                     step=1) if not pressure_test else 1
        individual_model_layer_3 = tf.keras.layers.MaxPooling1D(
            pool_size=hp_individual_model_layer_3_pool_size,
            strides=hp_individual_model_layer_3_strides,
            padding='valid'
        )(individual_model_layer_2)
        # layer 4
        hp_individual_model_layer_4_units = hp.Int(f'hp_{input_layer_name}_model_layer_4_units',
                                                   min_value=32,
                                                   max_value=64,
                                                   step=8) if not pressure_test else 64
        individual_model_layer_4 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hp_individual_model_layer_4_units, return_sequences=False)
        )(individual_model_layer_3)
        # layer 5
        hp_individual_model_layer_5_rate = hp.Float(f'hp_{input_layer_name}_model_layer_5_rate',
                                                    min_value=0.05,
                                                    max_value=0.5,
                                                    step=0.05) if not pressure_test else 0.1
        individual_model_layer_5 = tf.keras.layers.Dropout(
            hp_individual_model_layer_5_rate
        )(individual_model_layer_4)
        # layer 6
        individual_model_layer_6 = tf.keras.layers.RepeatVector(y_shape[0])(individual_model_layer_5)
        # layer 7
        hp_individual_model_layer_7_units = hp.Int(f'hp_{input_layer_name}_model_layer_7_units',
                                                   min_value=32,
                                                   max_value=64,
                                                   step=8) if not pressure_test else 64
        individual_model_layer_7 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hp_individual_model_layer_7_units, return_sequences=True)
        )(individual_model_layer_6)
        # layer 8
        hp_individual_model_layer_8_rate = hp.Float(f'hp_{input_layer_name}_model_layer_8_rate',
                                                    min_value=0.05,
                                                    max_value=0.5,
                                                    step=0.05) if not pressure_test else 0.1
        individual_model_layer_8 = tf.keras.layers.Dropout(
            hp_individual_model_layer_8_rate
        )(individual_model_layer_7)
        return individual_model_layer_1, individual_model_layer_8

    common_model_layer_input, common_model_layer_output = common_model_layer_1, common_model_layer_5
    stft_f_model_layer_input, stft_f_model_layer_output = make_stft_layers(stft_f_shape, 'stft_f')
    stft_p_model_layer_input, stft_p_model_layer_output = make_stft_layers(stft_p_shape, 'stft_p')

    # %% Concatenate
    #
    main_model_layer_1 = tf.keras.layers.concatenate([common_model_layer_output,
                                                      stft_f_model_layer_output,
                                                      stft_p_model_layer_output])
    #
    hp_main_model_layer_2_units = hp.Int('hp_main_model_layer_2_units',
                                         min_value=48,
                                         max_value=128,
                                         step=8) if not pressure_test else 128
    main_model_layer_2 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hp_main_model_layer_2_units, return_sequences=True)
    )(main_model_layer_1)
    #
    hp_main_model_layer_3_rate = hp.Float('hp_main_model_layer_3_rate',
                                          min_value=0.05,
                                          max_value=0.5,
                                          step=0.05) if not pressure_test else 0.1
    main_model_layer_3 = tf.keras.layers.Dropout(
        hp_main_model_layer_3_rate
    )(main_model_layer_2)
    #
    main_model_layer_4 = tf.keras.layers.Dense(y_shape[1], name="main_model_output")(main_model_layer_3)
    main_model = tf.keras.Model(
        inputs=[common_model_layer_input, stft_f_model_layer_input, stft_p_model_layer_input],
        outputs=[main_model_layer_4]
    )
    #
    hp_optimizer_learning_rate = hp.Choice('hp_optimizer_learning_rate',
                                           values=[0.01, 0.005, 0.001, 0.0005, 0.0001]) if not pressure_test else 0.001
    hp_loss_delta = hp.Float('hp_loss_delta',
                             min_value=0.1,
                             max_value=0.5,
                             step=0.01) if not pressure_test else 0.1
    main_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_optimizer_learning_rate),
                       loss={'main_model_output': tf.keras.losses.Huber(hp_loss_delta)},
                       metrics=['mae'])
    return main_model


def tune_model(name: str):
    training_set, test_set = load_training_and_test_set(name)
    save_folder_path = MODEL_PATH / "BO"
    try_to_find_folder_path_otherwise_make_one(save_folder_path / f'{name}')
    model_x, model_y, _ = get_data_for_nn(name)

    # Run pressure test
    @load_exist_pkl_file_otherwise_run_and_save(save_folder_path / f"{name}/pressure_test.pkl")
    def func():
        print("Start pressure test")
        pressure_test_model = make_hp_model(training_set, None, True)
        pressure_test_model.fit(
            model_x, model_y, epochs=10, validation_split=0.1, batch_size=BATCH_SIZE,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EPOCHS)]
        )
        print("Pressure test passed")
        return "pass"

    func()

    tuner = kt.BayesianOptimization(hypermodel=lambda x: make_hp_model(training_set, x, False),
                                    objective='val_mae',
                                    max_trials=32,
                                    directory=save_folder_path.__str__(),
                                    project_name=f'{name}')
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=int(EPOCHS * 0.05))

    tuner.search(model_x,
                 model_y,
                 verbose=2, epochs=EPOCHS,
                 validation_split=0.1,
                 callbacks=[stop_early],
                 batch_size=BATCH_SIZE)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    return tuner, best_hps


def train_model(name: str, continue_training: bool = False):
    training_set, test_set = load_training_and_test_set(name)
    save_folder_path = MODEL_PATH / name
    try_to_find_folder_path_otherwise_make_one(save_folder_path)

    tuner, best_hps = tune_model(name)
    model = tuner.hypermodel.build(best_hps)
    # model = make_hp_model(training_set, None, True)
    model.summary()

    class SaveCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % int(EPOCHS * 0.025) == 0:
                model.save_weights(save_folder_path / fr'model_epoch_{epoch}.h5')

    # model.compile(optimizer=tf.keras.optimizers.Adam(),
    #               loss={'main_model_output': keras.losses.mse},
    #               metrics=['mae'])
    if continue_training:
        model.load_weights(save_folder_path / fr'continue.h5')

    model_x, model_y, _ = get_data_for_nn(name)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=int(EPOCHS * 0.2))

    history = model.fit(x=model_x,
                        y=model_y,
                        verbose=2, epochs=EPOCHS,
                        validation_split=0.1,
                        callbacks=[SaveCallback(), stop_early],
                        batch_size=BATCH_SIZE)
    model.save_weights(save_folder_path / 'final.h5')
    return history


def test_model(name: str):
    training_set, test_set = load_training_and_test_set(name)
    save_folder_path = MODEL_PATH / name

    # %% Load model
    tuner, best_hps = tune_model(name)
    model = tuner.hypermodel.build(best_hps)
    # model = make_hp_model(training_set, None, True)
    model.load_weights(save_folder_path / 'final.h5')

    tt = 1
    test_data = (
        {'x': np.array(test_set['x']),
         'stft_f': np.array(test_set['stft_f']),
         'stft_p': np.array(test_set['stft_p'])},
        {'main_model_output': np.array(test_set['y'])}
    )
    y_pred = model.predict(test_data[0])
    ii = 0
    for i in range(ii, ii + 10):
        ax = series(y_pred[i, :, 0], label="Prediction")
        ax = series(test_data[1]['main_model_output'][i, :, 0], ax=ax, label="Actual")
        ax = series(test_data[0]['x'][i, :, 8], ax=ax, label="mains")

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
    # tt, _, _ = AMPDS2_DATA_600['training'].windowed_dataset(datetime.timedelta(days=1), batch_size=99)
    # bb = iter(tt)
    # data_list = [AMPDS2_DATA_600,
    #              UK_DALE_DATA_600]
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

    # tune_model("UKDALE_600")
    #
    fitted_model = train_model("UKDALE_600")
    # test_model("UKDALE_600")
    # test_model(UK_DALE_DATA_600, UK_DALE_DATA_600,
    #            test_date_time_range=(datetime.datetime(2016, 4, 25), datetime.datetime(2017, 4, 25)))
