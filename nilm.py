from Regression_Analysis.DeepLearning_Class import StackedBiLSTM
import pandas as pd
from pandas import DataFrame
from Ploting.fast_plot_Func import *
from project_path_Var import project_path_
from prepare_datasets import get_training_set_and_test_set_for_ampds2_dataset, NILMTorchDatasetForecast, \
    load_ampds2_weather
from nilmtk.legacy.disaggregate import CombinatorialOptimisation, FHMM
from pathlib import Path
from File_Management.path_and_file_management_Func import try_to_find_file
from File_Management.load_save_Func import load_pkl_file, save_pkl_file
from TimeSeries_Class import merge_two_time_series_df
from torch.utils.data import DataLoader
import torch
from torch.nn.functional import mse_loss
from workalendar.america import Canada
import datetime
import time
import re
from Writting.utils import put_cached_png_into_a_docx
from typing import List


def energies_paper_prepare_dataset_for_torch_model_for_ampds2_dataset(*, appliance_original_name: List[str] = None,
                                                                      appliance_type_name: str = None,
                                                                      sample_period: int,
                                                                      transform_args_file_path: Path):
    """
    返回专门针对pytorch的nn模型的dataset
    :param appliance_original_name: 可选['HPE', 'FRE', 'CDE']
    :param appliance_type_name: 可选['Lighting', 'Heating']:
                          'Lighting'是'B1E', 'B2E', 'OFE'的aggregate;
                          'Heating'只包含'HPE'
    :param sample_period: 必选。重采样。单位是秒
    :param transform_args_file_path
    """
    if ((appliance_original_name or appliance_type_name) is None) or (
            (appliance_original_name is not None) and (appliance_type_name is not None)):
        raise Exception('Ambiguous args')
    # convert one appliance to list
    if not isinstance(appliance_original_name, list):
        appliance_original_name = [appliance_original_name]

    training_set, test_set, full_set = get_training_set_and_test_set_for_ampds2_dataset()
    torch_sets = []
    _path = Path(project_path_) / f'Data/Results/Energies_paper/Ampds2/lstm/' \
                                  f'{appliance_original_name or appliance_type_name}/'
    _path.mkdir(parents=True, exist_ok=True)
    transform_args_file_path = transform_args_file_path or (
            _path / f'transform_args_{appliance_original_name or appliance_type_name}_{sample_period}.pkl'
    )
    # 生成training set和test set对应的TorchDataset对象
    for i, this_set in enumerate((training_set, test_set)):
        # appliance_var
        if appliance_original_name[0] is not None:
            appliance_df = next(this_set.select_using_appliances(
                original_name=appliance_original_name).meters[0].load(ac_type='active', sample_period=sample_period))
            appliance_df = appliance_df.droplevel('physical_quantity', axis=1)  # type: DataFrame
        else:
            if appliance_type_name == 'Lighting':
                appliance_df = this_set.select_using_appliances(
                    original_name=['B1E', 'B2E', 'OFE']).dataframe_of_meters(
                    ac_type='active', sample_period=sample_period)
                appliance_df['Lighting'] = appliance_df.sum(axis=1)
                appliance_df.drop(columns=appliance_df.columns[:-1], inplace=True)
            elif appliance_type_name == 'Heating':
                appliance_df = next(this_set.select_using_appliances(
                    original_name=['HPE']).meters[0].load(ac_type='active', sample_period=sample_period))
                appliance_df = appliance_df.droplevel('physical_quantity', axis=1)  # type: DataFrame
            else:
                raise Exception("Unsupported type. Only support 'Lighting' or 'Heating'")
        appliance_df.rename(columns={appliance_df.columns[0]: 'appliance_var'}, inplace=True)
        # mains_var
        mains_df = next(this_set.mains().load(
            ac_type='active', sample_period=sample_period)).droplevel('physical_quantity', axis=1)  # type: DataFrame
        mains_df.rename(columns={mains_df.columns[0]: 'mains_var'}, inplace=True)
        # weather_var
        ampds2_weather_df = load_ampds2_weather()
        mains_weather_df_merged = merge_two_time_series_df(mains_df, ampds2_weather_df)
        mains_weather_df_merged = mains_weather_df_merged.reindex(columns=mains_weather_df_merged.columns[1:].append(
            mains_weather_df_merged.columns[slice(1)]))
        # 组装成TorchDataset对象需要的data
        data = pd.concat((mains_weather_df_merged, appliance_df), axis=1)
        # 生成TorchDataset对象
        # 注意transform_args.pkl的命名方式，这样让training set和test set共用一组参数，只与用电器有关,
        # 由training set决定，因为loop中它在先
        torch_sets.append(NILMTorchDatasetForecast(data,
                                                   # 即：一天中的样本个数
                                                   sequence_length=int((3600 * 24) / sample_period),
                                                   transform_args_file_path=transform_args_file_path,
                                                   country=Canada()))
    return tuple(torch_sets)


def energies_paper_train_torch_model_for_ampds2_dataset(*, appliance_original_name: str = None,
                                                        appliance_type_name: str = None,
                                                        sample_period: int,
                                                        model_save_path: Path,
                                                        transform_args_file_path: Path) -> dict:
    training_time_path = model_save_path.parent / re.sub(r'_model', '_training_and_loss.pkl', model_save_path.stem)
    # TODO
    if True:
        # if not try_to_find_file(model_save_path):
        ############################################################
        epoch_num = 3000
        training_torch_set_dl_bs = 50
        hidden_size = 512
        learning_rate = 1e-4
        weight_decay = 0.00001
        dropout = 0.1
        #############################################################
        training_torch_set = energies_paper_prepare_dataset_for_torch_model_for_ampds2_dataset(
            appliance_original_name=appliance_original_name,
            appliance_type_name=appliance_type_name,
            sample_period=sample_period,
            transform_args_file_path=transform_args_file_path
        )[0]
        training_torch_set_dl = DataLoader(training_torch_set,
                                           batch_size=training_torch_set_dl_bs,
                                           shuffle=False)
        # %% 定义模型
        simple_lstm_model = StackedBiLSTM(input_size=training_torch_set[0][0].size()[-1],
                                          hidden_size=hidden_size,
                                          output_size=training_torch_set[0][1].size()[-1],
                                          dropout=dropout)
        # simple_lstm_model = torch.nn.DataParallel(simple_lstm_model, device_ids=[0]).cuda()  # 将模型转为cuda类型
        # %% 定义优化器
        opt = torch.optim.Adam(simple_lstm_model.parameters(),
                               lr=learning_rate,
                               weight_decay=weight_decay)  # weight_decay代表L2正则化
        # %% 定义loss函数
        loss_func = mse_loss
        # %% 开始train
        start_time = time.time()
        epoch_loss = []
        for i in range(epoch_num):
            simple_lstm_model.train()
            batch_loss = []
            for index, (xb, yb) in enumerate(training_torch_set_dl):
                pred = simple_lstm_model(xb)
                loss = loss_func(pred, yb).cuda()
                loss.backward()
                opt.step()
                opt.zero_grad()
                batch_loss.append(loss)
                print(f"第{i + 1: d}个epoch, 第{index + 1: d}个batch, loss={loss}")
            epoch_loss.append(batch_loss)
        # 保存整个模型
        torch.save(simple_lstm_model, model_save_path)
        # 保存训练时间和loss
        save_pkl_file(training_time_path, {'time': time.time() - start_time,
                                           'loss': epoch_loss})

    return {'model': torch.load(model_save_path),
            'training_time_and_loss': load_pkl_file(training_time_path)}


def energies_paper_test_torch_model_for_ampds2_dataset(*, appliance_original_name: str = None,
                                                       appliance_type_name: str = None,
                                                       sample_period: int,
                                                       model_save_path: Path,
                                                       transform_args_file_path: Path):
    model = torch.load(model_save_path)
    # 载入测试集
    test_torch_set = energies_paper_prepare_dataset_for_torch_model_for_ampds2_dataset(
        appliance_original_name=appliance_original_name,
        appliance_type_name=appliance_type_name,
        sample_period=sample_period,
        transform_args_file_path=transform_args_file_path)[0]
    test_torch_set_dl = DataLoader(test_torch_set,
                                   batch_size=1,
                                   shuffle=False)
    save_as_docx_buff = {(test_torch_set.data.index[0] + datetime.timedelta(days=i)).strftime("%Y_%m_%d"): [None, None]
                         for i in range(test_torch_set.__len__())}

    for index, (xb, yb) in enumerate(test_torch_set_dl):
        model.eval()
        pred = model(xb)
        ax = series(test_torch_set.data.index[
                    index * test_torch_set.sequence_length:(index + 1) * test_torch_set.sequence_length
                    ].to_pydatetime(),
                    pred[0,:,0].detach().cpu().numpy().flatten(), label='LSTM', figure_size=(10, 2.4))
        x_plot = test_torch_set.data.index[
                 index * test_torch_set.sequence_length:(index + 1) * test_torch_set.sequence_length].to_pydatetime()
        buf = series(
            x_plot,
            yb.cpu().numpy().flatten(),
            ax=ax,
            label='Truth',
            x_label='Time',
            y_label='Normlised active power (p.u.)',
            figure_size=(10, 2.4),
            title=appliance_original_name or appliance_type_name,
            save_to_buffer=True)
        save_as_docx_buff[(test_torch_set.data.index[0] + datetime.timedelta(days=index)).strftime("%Y_%m_%d")][0] = buf
    put_cached_png_into_a_docx(save_as_docx_buff,
                               model_save_path.parent / f'{appliance_original_name or appliance_type_name}_'
                                                        f'{sample_period}.docx',
                               1)
    # 画loss
    # _loss = tt['training_time_and_loss']['loss']
    # _loss = np.array(list(map(lambda x: np.array(x, dtype=float), _loss)))
    # _loss_mean = np.mean(_loss, axis=1)
    # series(_loss_mean)


def energies_paper_train_nilm_models_for_ampds2_dataset(top_n: int = 3):
    """
    只从ampds2_dataset中的training_set中选取top_n个（默认3个）appliance。分别是HPE, FRE and CDE。
    这里是符合nilmtk标准的算法。
    torch的算法单独处理
    :return:
    """
    # 准备训练数据
    training_set, test_set, _ = get_training_set_and_test_set_for_ampds2_dataset()
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
    # for _sample_period in (60, 60 * 30):
    # for this_type in ('Lighting',):
    #     # energies_paper_train_torch_model_for_ampds2_dataset(
    #     #     appliance_type_name=this_type,
    #     #     model_save_path=Path(project_path_) / f'Data/Results/Energies_paper/Ampds2/lstm/{this_type}/'
    #     #                                           f'{this_type}_{_sample_period}_lstm_model.pkl',
    #     #     sample_period=_sample_period)
    #     energies_paper_test_torch_model_for_ampds2_dataset(
    #         appliance_type_name=this_type,
    #         model_save_path=Path(project_path_) / f'Data/Results/Energies_paper/Ampds2/lstm/{this_type}/'
    #                                               f'{this_type}_{_sample_period}_lstm_model.pkl',
    #         sample_period=_sample_period)

    # for this_appliance in ('B1E', 'OFE', 'B2E', 'HPE'):
    #     if not ((_sample_period == 60) and (this_appliance == 'HPE')):
    #         continue
    #     energies_paper_train_torch_model_for_ampds2_dataset(
    #         appliance_original_name=this_appliance,
    #         model_save_path=Path(project_path_) / f'Data/Results/Energies_paper/Ampds2/lstm/{this_appliance}/'
    #                                               f'{this_appliance}_{_sample_period}_lstm_model.pkl',
    #         sample_period=_sample_period)
    #     # energies_paper_test_torch_model_for_ampds2_dataset(
    #     appliance_original_name=this_appliance,
    #     model_save_path=Path(project_path_) / f'Data/Results/Energies_paper/Ampds2/lstm/{this_appliance}/'
    #                                           f'{this_appliance}_{_sample_period}_lstm_model.pkl',
    #     sample_period=_sample_period)

    # energies_paper_prepare_dataset_for_torch_model_for_ampds2_dataset(appliance_original_name=['HPE'],
    #                                                                   sample_period=1800)
    _sample_period = 60
    energies_paper_train_torch_model_for_ampds2_dataset(
        appliance_original_name='HPE',
        model_save_path=Path(project_path_) / 'Data/Results/Energies_paper/Ampds2/lstm/forecast/input_week/'
                                              f'HPE_and_total_{_sample_period}.pkl',
        sample_period=_sample_period,
        transform_args_file_path=Path(project_path_) / 'Data/Results/Energies_paper/Ampds2/lstm/forecast/input_week/'
                                                       f'HPE_and_total_{_sample_period}_transform_args.pkl'
    )
    # energies_paper_test_torch_model_for_ampds2_dataset(
    #     appliance_original_name='HPE',
    #     model_save_path=Path(project_path_) / 'Data/Results/Energies_paper/Ampds2/lstm/forecast/input_week/'
    #                                           f'HPE_and_total_{_sample_period}.pkl',
    #     sample_period=_sample_period,
    #     transform_args_file_path=Path(project_path_) / 'Data/Results/Energies_paper/Ampds2/lstm/forecast/input_week/'
    #                                                    f'HPE_and_total_{_sample_period}_transform_args.pkl'
    # )
