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

setlocale(LC_ALL, "en_US")
tf.keras.backend.set_floatx("float64")

remove_win10_max_path_limit()
# %% Load all data sets used in the paper
transformation_args_folder_path_common = project_path_ / r"Data\Results\Energies_paper"
# Ampds2
AMPDS2_PATH_600_HEATING = transformation_args_folder_path_common / "Ampds2_600_heating"
AMPDS2_DATA_600_HEATING = {
    'training': NILMDataSet(name='Ampds2_training', resolution=600, appliance='heating',
                            transformation_args_folder_path=AMPDS2_PATH_600_HEATING),
    'test': NILMDataSet(name='Ampds2_test', resolution=600, appliance='heating',
                        transformation_args_folder_path=AMPDS2_PATH_600_HEATING),
}
# Turkey apartment
TURKEY_APARTMENT_PATH_3600_LIGHTING = transformation_args_folder_path_common / "Turkey_apartment_3600_lighting"
TURKEY_APARTMENT_DATA_3600_LIGHTING = {
    'training': NILMDataSet(name='Turkey_apartment_training', appliance='lighting',
                            transformation_args_folder_path=TURKEY_APARTMENT_PATH_3600_LIGHTING),
    'test': NILMDataSet(name='Turkey_apartment_test', appliance='lighting',
                        transformation_args_folder_path=TURKEY_APARTMENT_PATH_3600_LIGHTING),
}

# Turkey detached house
TURKEY_HOUSE_PATH_3600_LIGHTING = transformation_args_folder_path_common / "Turkey_Detached House_3600_lighting"
TURKEY_HOUSE_DATA_3600_LIGHTING = {
    'training': NILMDataSet(name='Turkey_Detached House_training', appliance='lighting',
                            transformation_args_folder_path=TURKEY_HOUSE_PATH_3600_LIGHTING),
    'test': NILMDataSet(name='Turkey_Detached House_test', appliance='lighting',
                        transformation_args_folder_path=TURKEY_HOUSE_PATH_3600_LIGHTING),
}

# UK DALE
UK_DALE_PATH_600_HEATING = transformation_args_folder_path_common / "UKDALE_600_heating"
UK_DALE_DATA_600_HEATING = {
    'training': NILMDataSet(name='UKDALE_training', resolution=600, appliance='heating',
                            transformation_args_folder_path=UK_DALE_PATH_600_HEATING),
    'test': NILMDataSet(name='UKDALE_test', resolution=600, appliance='heating',
                        transformation_args_folder_path=UK_DALE_PATH_600_HEATING),
}
UK_DALE_PATH_3600_LIGHTING = transformation_args_folder_path_common / "UKDALE_3600_lighting"
UK_DALE_DATA_3600_LIGHTING = {
    'training': NILMDataSet(name='UKDALE_training', resolution=3600, appliance='lighting',
                            transformation_args_folder_path=UK_DALE_PATH_3600_LIGHTING),
    'test': NILMDataSet(name='UKDALE_test', resolution=3600, appliance='lighting',
                        transformation_args_folder_path=UK_DALE_PATH_3600_LIGHTING),
}


# Infer path variable from data variable
def get_path_from_data_variable(data_variable: dict) -> Path:
    data_variable_name = [x[0] for x in globals().items() if x[1] is data_variable and x[0] != "data_variable"][0]
    return globals()[data_variable_name.replace("DATA", "PATH")]


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


def train_model(dataset: dict, *, batch_size: int, epochs=5000, task: str, save_per_epoch: int = 250):
    assert task in ("continue training", "new training", "load results")
    save_folder_path = get_path_from_data_variable(dataset)

    training_set_nn, _, _ = dataset['training'].windowed_dataset(datetime.timedelta(days=1), batch_size=batch_size)
    test_set_nn, _, _ = dataset['test'].windowed_dataset(datetime.timedelta(days=1), batch_size=batch_size)
    try_to_find_folder_path_otherwise_make_one(save_folder_path / fr'temp')

    class SaveCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch % save_per_epoch == 0) and (epoch != 0):
                model.save(save_folder_path / fr'temp\fitting_model_epoch_{epoch}.h5')

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=128, kernel_size=3,
                               strides=1, padding="causal",
                               activation="relu",
                               input_shape=training_set_nn.element_spec[0].shape[1:]),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.MaxPooling1D(pool_size=1, strides=1, padding='same'),
        tf.keras.layers.Bidirectional(tf.keras.layers.RNN(tf.keras.layers.LSTMCell(512),
                                                          return_sequences=True), ),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Bidirectional(tf.keras.layers.RNN(tf.keras.layers.LSTMCell(512),
                                                          return_sequences=True)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(training_set_nn.element_spec[1].shape[-1]),
    ])
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=keras.losses.mse, metrics=['mae'])
    if task == "load results":
        model.load_weights(save_folder_path / fr'fitted_model.h5')
        return model
    if task == "continue training":
        existings = os.listdir(save_folder_path / "temp")
        max_existing_epoch = max([re.findall(r"(?<=_)\d+(?=.h5)", x)[0] for x in existings])
        model.load_weights(save_folder_path / "temp" / [x for x in existings if max_existing_epoch in x][0])
    history = model.fit(training_set_nn, verbose=2, epochs=epochs, validation_data=test_set_nn,
                        callbacks=[SaveCallback()])
    model.save(save_folder_path / fr'fitted_model.h5')
    return model


def train_model_with_attention(dataset: NILMDataSet,
                               batch_size: int,
                               teacher_forcing: float = 0.001):
    dataset_nn, _, _, _ = dataset.windowed_dataset(datetime.timedelta(days=1), batch_size=batch_size)
    output_seq_start_val = 0.

    training_mode = True

    encoder = TensorFlowCovBiLSTMEncoder(lstm_layer_hidden_size=64,
                                         training_mode=training_mode)
    decoder = TensorFlowLSTMDecoder(lstm_layer_hidden_size=64,
                                    training_mode=training_mode,
                                    output_feature_len=1)

    # %% Optimiser and loss function
    optimizer = tf.keras.optimizers.Adam()
    # loss_function = tf.keras.losses.mean_absolute_error
    loss_object = tf.keras.losses.MeanSquaredError()

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        # mask = tf.cast(mask, dtype=loss_.dtype)
        # loss_ *= mask  # TODO  ############?############

        return tf.reduce_mean(loss_)

    # %% Train step
    @tf.function
    def train_step(_x, _y):
        _loss = 0
        _encoder_hidden = encoder.initialize_h_0_c_0(_x.shape[0])

        with tf.GradientTape() as tape:
            _encoder_output, _encoder_hidden = encoder(x=_x, h_0_c_0_list=_encoder_hidden)
            _decoder_hidden = _encoder_hidden
            _decoder_input = tf.zeros((_y.shape[0], 1), dtype="float64") + output_seq_start_val  # -1 represent start
            _decoder_input = tf.expand_dims(_decoder_input, 1)

            for i in range(0, _y.shape[1]):
                _predictions, _decoder_hidden, _ = decoder(_decoder_input,
                                                           _decoder_hidden,
                                                           _encoder_output)
                _loss += loss_function(tf.expand_dims(_y[:, i], 1), _predictions)
                if (_y is not None) and (tf.random.uniform((1,)) < teacher_forcing):
                    _decoder_input = tf.expand_dims(_y[:, i], 1)
                else:
                    _decoder_input = _predictions
            _batch_loss = _loss / int(_y.shape[1])
            print(f"_batch_loss = {_batch_loss}")

            variables = encoder.trainable_variables + decoder.trainable_variables

            gradients = tape.gradient(_loss, variables)

            optimizer.apply_gradients(zip(gradients, variables))

            return _batch_loss

    # %% Train
    epochs = 1000
    # steps_per_epoch = len(dataset_nn) // BATCH_SIZE

    for epoch in range(epochs):
        start = time.time()

        total_loss = 0

        for batch_number, (x, y) in enumerate(dataset_nn):
            batch_loss = train_step(x, y)
            total_loss += batch_loss

            if batch_number % 1 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch_number + 1,
                                                             batch_loss))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            pass

        print('Epoch {} total_loss Loss {:.4f}'.format(epoch + 1, total_loss))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def test_model(model_built_dataset: dict, test_dataset: dict, *,
               test_date_time_range: Tuple[datetime.datetime, datetime.datetime],
               plot: bool = True, batch_size: int = 100):
    # %% Load model
    model_built_dataset_path = get_path_from_data_variable(model_built_dataset)  # type:Path
    model = tf.keras.models.load_model(model_built_dataset_path / "fitted_model.h5")

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
        x_var = tf.Variable(x, dtype="float64")
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
    gradient_analyser_obj.plot(new_gradients, new_predictor_names, plot_individual_steps=False)

    # ax = series(model.predict(list(test_set.as_numpy_iterator())[0][0]).flatten())
    # ax = series(this_output.flatten(), ax=ax)


if __name__ == '__main__':
    # tt, _, _ = AMPDS2_DATA_600_HEATING['training'].windowed_dataset(datetime.timedelta(days=1), batch_size=99)
    # bb = iter(tt)
    data_list = [AMPDS2_DATA_600_HEATING,
                 UK_DALE_DATA_600_HEATING]
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

    # fitted_model = train_model(AMPDS2_DATA_600_HEATING,
    #                            AMPDS2_PATH_600_HEATING,
    #                            batch_size=200, epochs=5000, task="load results")
    test_model(UK_DALE_DATA_600_HEATING, AMPDS2_DATA_600_HEATING,
               test_date_time_range=(datetime.datetime(2013, 11, 1), datetime.datetime(2013, 11, 2)))
