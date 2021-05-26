import tensorflow as tf
from Ploting.fast_plot_Func import *
from locale import setlocale, LC_ALL
from papers.Energies_2020.prepare_datasets import ScotlandLongerDataset, NILMDataSet, ampds2_dataset_full_df
from File_Management.path_and_file_management_Func import *

setlocale(LC_ALL, "en_US")
tf.keras.backend.set_floatx("float32")
remove_win10_max_path_limit()

BATCH_SIZE = 1250  # 2500 for AMPDS2_600, 1250 for UKDALE_600
EPOCHS = 25_000
# %% Load all data sets used in the paper
RESULTS_ROOT_PATH = Path(r"C:\\")
# results_root_path = project_path_ / r"Data\Results\Energies_paper"
DATA_PREPARE_PATH = RESULTS_ROOT_PATH / "Data_preparation"
MODEL_PATH = RESULTS_ROOT_PATH / "NN"
# Ampds2
AMPDS2_DATA_600 = {
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
UK_DALE_DATA_600 = {
    'training': NILMDataSet(name='UKDALE_training', resolution=600, appliance='lighting',
                            transformation_args_folder_path=DATA_PREPARE_PATH / "UKDALE_600_lighting"),
    'test': NILMDataSet(name='UKDALE_test', resolution=600, appliance='lighting',
                        transformation_args_folder_path=DATA_PREPARE_PATH / "UKDALE_600_lighting"),
}
