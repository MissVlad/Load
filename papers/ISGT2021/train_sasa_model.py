import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM
from Ploting.fast_plot_Func import *
from pathlib import Path
from scipy.io import loadmat, savemat
from project_utils import project_path_
from File_Management.path_and_file_management_Func import *
from File_Management.load_save_Func import *

DATA_FOLDER = project_path_ / r'Data\Raw\ISGT2021\all'
XTV_FOLDER = project_path_ / r'Data\Raw\ISGT2021\XTV'
RESULTS_FOLDER = project_path_ / r'Data\Results\ISGT2021'
EPOCHS = 25_000


def get_data(point: int, use_xtv: bool):
    if not use_xtv:
        file_path = DATA_FOLDER / f'point{point}.mat'
    else:
        file_path = XTV_FOLDER / f'XTV_{point}.mat'
    data = loadmat(file_path)
    if not use_xtv:
        ans = dict()
        for pair in [('XTrain', 'x_train'), ('YTrain', 'y_train'),
                     ('XVali', 'x_val'), ('YVali', 'y_val'),
                     ('XTest', 'x_test'), ('YTest', 'y_test')]:
            ans[pair[1]] = []
            for ele in data[pair[0]]:
                ans[pair[1]].append(ele[0].T)
            ans[pair[1]] = np.array(ans[pair[1]])
    else:
        ans = []
        for ele in data['XTV']:
            ans.append(ele[0].T)
        ans = np.array(ans)
    return ans


def get_model(x_shape, y_shape):
    #
    model = tf.keras.Sequential([
        tf.keras.Input(shape=x_shape[1:]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(y_shape[-1])
    ])
    #
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.MSE,
                  metrics=['mae'])
    model.summary()
    return model


def train_model(point: int, use_xtv: bool):
    data = get_data(point, use_xtv)
    now_results_folder = RESULTS_FOLDER / f'point{point}'
    try_to_find_folder_path_otherwise_make_one(now_results_folder)

    model = get_model(data['x_train'].shape, data['y_train'].shape)

    #
    class SaveCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % int(EPOCHS * 0.025) == 0:
                model.save_weights(now_results_folder / fr'model_epoch_{epoch}.h5')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=250 * 30)
    #
    history = model.fit(x=data['x_train'],
                        y=data['y_train'],
                        verbose=2, epochs=EPOCHS,
                        validation_data=(data['x_val'], data['y_val']),
                        callbacks=[SaveCallback(), stop_early],
                        batch_size=4096)
    model.save_weights(now_results_folder / 'final.h5')
    return history


def test_model(point: int, use_xtv: bool):
    data = get_data(point, use_xtv)
    now_results_folder = RESULTS_FOLDER / f'point{point}'

    if not use_xtv:
        try_to_find_folder_path_otherwise_make_one(RESULTS_FOLDER / 'all')
        get_model_args = (data['x_train'].shape, data['y_train'].shape)
    else:
        try_to_find_folder_path_otherwise_make_one(RESULTS_FOLDER / 'xtv')
        get_model_args = ([None, 1, 41], [None, 1, 1])

    model = get_model(*get_model_args)
    model.load_weights(now_results_folder / 'final.h5')

    if not use_xtv:
        #
        y_train_pred = model.predict(data['x_train']).flatten()
        y_val_pred = model.predict(data['x_val']).flatten()
        y_test_pred = model.predict(data['x_test']).flatten()
        #
        savemat(now_results_folder / f'y_train_pred.mat', {'y_train_pred': y_train_pred})
        savemat(now_results_folder / f'y_val_pred.mat', {'y_val_pred': y_val_pred})
        savemat(now_results_folder / f'y_test_pred.mat', {'y_test_pred': y_test_pred})
        #
        savemat(RESULTS_FOLDER / f'all/y_train_pred_point{point}.mat', {'y_train_pred': y_train_pred})
        savemat(RESULTS_FOLDER / f'all/y_val_pred_point{point}.mat', {'y_val_pred': y_val_pred})
        savemat(RESULTS_FOLDER / f'all/y_test_pred_point{point}.mat', {'y_test_pred': y_test_pred})
    else:
        y_xtv_pred = model.predict(data).flatten()
        savemat(RESULTS_FOLDER / f'xtv/y_xtv_pred_point{point}.mat', {'y_xtv_pred': y_xtv_pred})


def combine_files(use_xtv: bool):
    if use_xtv:
        folder = RESULTS_FOLDER / 'xtv'
    else:
        folder = RESULTS_FOLDER / 'all'


if __name__ == '__main__':
    for i in range(1, 49):
        # train_model(i)
        test_model(i, True)
    # test_model(1, False)
