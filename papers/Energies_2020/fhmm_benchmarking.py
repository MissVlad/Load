from prepare_datasets import ScotlandLongerDataset, NILMDataSet, ampds2_dataset_full_df
from pathlib import Path
from Ploting.fast_plot_Func import *
import numpy as np
import pandas as pd
from numpy import ndarray
# from nilmtk.disaggregate.fhmm_exact import FHMMExact
# from nilmtk.disaggregate.combinatorial_optimisation import CO

# %% Load all data sets used in the paper
results_root_path = Path(r"C:\\")
# results_root_path = project_path_ / r"Data\Results\Energies_paper"
DATA_PREPARE_PATH = results_root_path / "Data_preparation"

AMPDS2_DATA_600_HEATING = {
    'training': NILMDataSet(name='Ampds2_training', resolution=600, appliance='heating',
                            transformation_args_folder_path=DATA_PREPARE_PATH / "Ampds2_600_heating"),
    'test': NILMDataSet(name='Ampds2_test', resolution=600, appliance='heating',
                        transformation_args_folder_path=DATA_PREPARE_PATH / "Ampds2_600_heating"),
}


def load_data(name: str):
    assert name in {"AMPDS2_600", "UKDALE_600"}

    if name == "AMPDS2_600":
        return AMPDS2_DATA_600_HEATING
    else:
        raise


def run_model(name):
    data = load_data(name)["training"]
    tt = 1
    model = FHMMExact({"num_of_states": 10})
    model.partial_fit([data.data[['mains']]],
                      [('heating', [data.data[['heating']]])])
    model.disaggregate_chunk([data.data[['mains']]])[0]

    model_2 = CO({})
    model_2.partial_fit([data.data[['mains', 'temperature']]],
                        [('heating', [data.data[['heating']]])])
    """
    ax = series(model_2.disaggregate_chunk([data.data[['mains']]])[0].values.flatten())
    ax = series(data.data[['heating']].values.flatten(), ax=ax)
    """


if __name__ == "__main__":
    pass
    run_model("AMPDS2_600")
