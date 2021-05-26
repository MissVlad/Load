from project_utils import *
from Ploting.fast_plot_Func import *
import tensorflow as tf
from tensorflow import keras
from scipy.io.matlab import loadmat
from pathlib import Path
import kerastuner as kt

this_data_set = loadmat(project_path_ / r'Data\Raw\MEEPS2020\MV_weekahead_data.mat')


def convert_to_float_ndarray_and_transpose(cell_data):
    this_ndarray = []
    for i in range(cell_data.shape[0]):
        this_ndarray.append(cell_data[i].tolist()[0].tolist())
    this_ndarray = np.array(this_ndarray)
    this_ndarray = np.transpose(this_ndarray, [0, -1, -2])
    return this_ndarray


def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(10))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


for this_cell in ('Xtrain', 'Ytrain', 'Xtest', 'Ytest'):
    exec(f"{this_cell} = this_data_set['{this_cell}']")
    exec(f"{this_cell} = convert_to_float_ndarray_and_transpose({this_cell})")
del this_data_set
