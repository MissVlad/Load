import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from Ploting.fast_plot_Func import *
from project_path_Var import project_path_
from prepare_datasets import get_training_set_test_set_for_ampds2_dataset, NILMTorchDataset, \
    load_ampds2_weather
from nilmtk.legacy.disaggregate import CombinatorialOptimisation, FHMM
import time
from pathlib import Path
from File_Management.path_and_file_management_Func import try_to_find_file
from Regression_Analysis.DeepLearning_Class import SimpleLSTM, BayesCOVN1DLSTM
from nilmtk import MeterGroup
from Time_Processing.SynchronousTimeSeriesData_Class import merge_two_time_series_df
from torch.utils.data import DataLoader
import torch
from torch.nn.functional import mse_loss


def energies_paper_prepare_dataset_for_torch_model_for_ampds2_dataset(appliance_original_name: str):
    """
    返回专门针对pytorch的nn模型的dataset
    :param appliance_original_name: 可选['HPE', 'FRE', 'CDE']
    """
    training_set, test_set = get_training_set_test_set_for_ampds2_dataset()
    torch_sets = []
    _path = Path(project_path_) / 'Data/Results/Energies_paper/Ampds2/'
    transform_args_file_path = _path / 'transform_args_{}.pkl'.format(appliance_original_name)
    transformed_data_file_path = _path / 'transformed_data_file_path_{}.pkl'.format(appliance_original_name)
    # 生成training set和test set对应的TorchDataset对象
    for this_set in (training_set, test_set):
        # appliance_var
        appliance_df = next(this_set.select_using_appliances(original_name=[appliance_original_name]).meters[0].load(
            ac_type='active', sample_period=60)).droplevel('physical_quantity', axis=1)  # type: DataFrame
        appliance_df.rename(columns={appliance_df.columns[0]: 'appliance_var'}, inplace=True)
        # mains_var
        mains_df = next(this_set.mains().load(
            ac_type='active', sample_period=60)).droplevel('physical_quantity', axis=1)  # type: DataFrame
        mains_df.rename(columns={mains_df.columns[0]: 'mains_var'}, inplace=True)
        # time_var
        time_df = pd.DataFrame(mains_df.index.to_list(),
                               index=mains_df.index,
                               columns=['time_var'])
        # weather_var
        ampds2_weather_df = load_ampds2_weather()
        mains_weather_df_merged = merge_two_time_series_df(mains_df, ampds2_weather_df)
        mains_weather_df_merged = mains_weather_df_merged.reindex(columns=mains_weather_df_merged.columns[1:].append(
            mains_weather_df_merged.columns[slice(1)]))
        # 组装成TorchDataset对象需要的data
        data = pd.concat((time_df, mains_weather_df_merged, appliance_df), axis=1)
        # 生成TorchDataset对象
        # 注意transform_args.pkl的命名方式，这样让training set和test set共用一组参数，由training set决定，因为loop中它在先
        torch_sets.append(NILMTorchDataset(data,
                                           sequence_length=60 * 24,
                                           transform_args_file_path=transform_args_file_path,
                                           transformed_data_file_path=transformed_data_file_path))
    return tuple(torch_sets)


def energies_paper_train_torch_model_for_ampds2_dataset(appliance_original_name: str):
    ############################################################
    epoch_num = 1
    training_torch_set_dl_bs = 4
    hidden_size = 128
    learning_rate = 1e-4
    #############################################################
    training_torch_set = energies_paper_prepare_dataset_for_torch_model_for_ampds2_dataset(appliance_original_name)[0]
    training_torch_set_dl = DataLoader(training_torch_set,
                                       batch_size=training_torch_set_dl_bs,
                                       shuffle=False)
    # validation_torch_set_dl = DataLoader(training_torch_set, batch_size=training_torch_set_dl_bs * 2, shuffle=False)
    # %% 定义模型
    simple_lstm_model = SimpleLSTM(input_size=training_torch_set[0][0].size()[-1],
                                   hidden_size=hidden_size,
                                   output_size=training_torch_set[0][1].size()[-1])
    # %% 定义优化器
    opt = torch.optim.Adam(simple_lstm_model.parameters(), lr=learning_rate)
    # %% 定义loss函数
    loss_func = mse_loss
    # %% 开始train和validate
    for _ in range(epoch_num):
        simple_lstm_model.train()
        for index, (xb, yb) in enumerate(training_torch_set_dl):
            pred = simple_lstm_model(xb)
            loss = loss_func(pred, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
            print(f"第{index + 1: d}个batch, loss={loss}")

        # simple_lstm_model.eval()
        # with torch.no_grad():
        #     valid_loss = sum(loss_func(simple_lstm_model(xb, yb) for xb, yb in validation_torch_set_dl))


def energies_paper_train_nilm_models_for_ampds2_dataset(top_n: int = 3):
    """
    只从ampds2_dataset中的training_set中选取top_n个（默认3个）appliance。分别是HPE, FRE and CDE。
    这里是符合nilmtk标准的算法。
    torch的算法单独处理
    :return:
    """
    # 准备训练数据
    training_set, test_set = get_training_set_test_set_for_ampds2_dataset()
    top_n_train_elec = training_set.select_using_appliances(original_name=['HPE', 'FRE', 'CDE'])
    # 模型save的路径
    models_path = Path('../Data/Results/Energies_paper/Ampds2')
    models_path.mkdir(parents=True, exist_ok=True)
    # 训练所有模型
    models = {'CO': CombinatorialOptimisation(), 'FHMM': FHMM()}
    sample_period = 60  # 都是down sample到60s
    for this_model_name, this_model in models.items():
        this_model_file = models_path / (this_model_name + '.pkl')
        if try_to_find_file(this_model_file):
            this_model.import_model(this_model_file)
        else:
            print("*" * 20)
            print(this_model_name)
            print("*" * 20)
            start = time.time()
            this_model.train(top_n_train_elec, sample_period=sample_period)
            this_model.export_model(this_model_file)
            end = time.time()
            print("Runtime =", end - start, "seconds.")


if __name__ == '__main__':
    # energies_paper_train_nilm_models_for_ampds2_dataset()
    energies_paper_train_torch_model_for_ampds2_dataset('HPE')
