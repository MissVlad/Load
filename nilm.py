from Regression_Analysis.DeepLearning_Class import StackedBiLSTM, GRUEncoderDecoderWrapper, GRUEncoder, \
    GRUDecoder, TensorFlowLSTMDecoder, TensorFlowLSTMEncoder, TensorFlowAttention
import pandas as pd
from pandas import DataFrame
from Ploting.fast_plot_Func import *
from project_utils import project_path_
from prepare_datasets import get_training_set_and_test_set_for_ampds2_dataset, NILMTorchDatasetForecast, \
    load_ampds2_weather, NILMTorchDataset
from nilmtk.legacy.disaggregate import CombinatorialOptimisation, FHMM
from pathlib import Path
from File_Management.path_and_file_management_Func import try_to_find_file, try_to_find_folder_path_otherwise_make_one
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
import os
import copy


# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# torch.backends.cudnn.enabled = False


# torch.backends.cudnn.benchmark = True
# torch.cuda.set_device(0)


def energies_paper_prepare_dataset_for_torch_model_for_ampds2_dataset(*, appliance_original_name: List[str] = None,
                                                                      appliance_type_name: str = None,
                                                                      sample_period: int,
                                                                      transform_args_file_path: Path,
                                                                      sets_save_path: Path = None):
    """
    返回专门针对pytorch的nn模型的dataset
    :param appliance_original_name: 可选['HPE', 'FRE', 'CDE']
    :param appliance_type_name: 可选['Lighting', 'Heating']:
                          'Lighting'是'B1E', 'B2E', 'OFE'的aggregate;
                          'Heating'只包含'HPE'
    :param sample_period: 必选。重采样。单位是秒
    :param transform_args_file_path
    :param sets_save_path
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
        tt1 = torch_sets[0][1]
        tt2 = torch_sets[0][2]
        tt3 = torch_sets[0][3]
        cc = 1
    if sets_save_path:
        train_np, test_np = [], []
        for i in range(torch_sets[0].__len__()):
            train_np.append(list(map(lambda x: x.cpu().numpy(), torch_sets[0][i])))
            test_np.append(list(map(lambda x: x.cpu().numpy(), torch_sets[1][i])))
        save_pkl_file(sets_save_path, {'train': train_np, 'test': test_np})
    torch_sets = tuple(torch_sets)
    return torch_sets


def energies_paper_load_ampds2_dataset_train_np_and_test_np(file_path: Path = None):
    file_path = file_path or (Path(project_path_) / 'Data/Results/Energies_paper/Ampds2/lstm/forecast/input_week/'
                                                    'sets_HPE_and_total_60_transform_args.pkl')
    return load_pkl_file(file_path)


def energies_paper_train_tf_model_for_ampds2_dataset(model_save_path: Path):
    import tensorflow as tf

    training_set = energies_paper_load_ampds2_dataset_train_np_and_test_np()['train']

    def training_set_gen():
        for i in range(training_set.__len__()):
            yield training_set[i][0], training_set[i][1]

    batch_size = 2
    steps_per_epoch = len(training_set) // batch_size
    tf_training_set = tf.data.Dataset.from_generator(training_set_gen, (np.float32, np.float32))
    tf_training_set = tf_training_set.batch(batch_size=batch_size, drop_remainder=False)

    tf_lstm_encoder = TensorFlowLSTMEncoder(hidden_size=32,
                                            training_mode=True)
    tf_lstm_decoder = TensorFlowLSTMDecoder(hidden_size=32,
                                            training_mode=True,
                                            output_feature_len=2)

    # # sample input
    # example_input_batch = next(iter(tf_training_set))[0]
    # sample_hidden = tf_lstm_encoder.initialize_h_0_c_0(batch_size)
    # sample_output, sample_hidden = tf_lstm_encoder(x=example_input_batch, h_0_c_0_list=sample_hidden)

    # attention_layer = TensorFlowAttention(10)
    # attention_result, attention_weights = attention_layer(sample_hidden[0], sample_output)
    #
    # sample_decoder_output, _, _ = tf_lstm_decoder(tf.random.uniform((batch_size, 1)),
    #                                               sample_hidden, sample_output)

    optimizer = tf.keras.optimizers.Adam()
    loss_func = tf.keras.losses.mean_squared_error

    # %% Check point
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=tf_lstm_encoder,
                                     decoder=tf_lstm_decoder)

    # %% train step
    @tf.function
    def train_step(_x, _y, _encoder_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            encoder_output, _encoder_hidden = tf_lstm_encoder(x=_x,
                                                              h_0_c_0_list=_encoder_hidden)
            decoder_hidden = encoder_hidden
            decoder_input = tf.zeros((batch_size, 2))

            # Teacher forcing - feeding the target as the next input
            for this_time_step in range(0, _y.shape[1]):
                print(this_time_step)
                # passing enc_output to the decoder
                predictions, decoder_hidden, _ = tf_lstm_decoder(
                    decoder_input,
                    decoder_hidden,
                    encoder_output
                )
                loss += loss_func(y[:, this_time_step, :], predictions)
        _batch_loss = (loss / int(_y.shape[1]))
        print(f"_batch_loss = {_batch_loss}")
        variables = tf_lstm_encoder.trainable_variables + tf_lstm_decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return _batch_loss

    # %% train
    epoch = 25000
    for epoch_idx in range(epoch):
        start_time = time.time()

        encoder_hidden = tf_lstm_encoder.initialize_h_0_c_0(batch_size)

        total_loss = 0
        for (batch_idx, (x, y)) in enumerate(tf_training_set.take(steps_per_epoch)):
            batch_loss = train_step(x, y, encoder_hidden)
            print(f"batch_idx = {batch_idx} finished")
            total_loss += batch_loss
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch_idx + 1,
                                                         batch_idx,
                                                         batch_loss.numpy()))
        if True:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Epoch {} Loss {:.4f}'.format(epoch_idx + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start_time))


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
        epoch_num = 25000
        training_torch_set_dl_bs = 90
        # training_torch_set_dl_bs = 25

        hidden_size = 1024
        learning_rate = 1e-4
        # weight_decay = 0.00000001
        weight_decay = 0.000001

        dropout = 0.1

        lstm_layer_num = 3
        #############################################################
        training_torch_set, test_torch_set = energies_paper_prepare_dataset_for_torch_model_for_ampds2_dataset(
            appliance_original_name=appliance_original_name,
            appliance_type_name=appliance_type_name,
            sample_period=sample_period,
            transform_args_file_path=transform_args_file_path
        )
        save_pkl_file(Path(r'.\training_torch_set'), training_torch_set)
        training_torch_set = load_pkl_file(Path(r'.\training_torch_set'))

        training_torch_set_dl = DataLoader(training_torch_set,
                                           batch_size=training_torch_set_dl_bs,
                                           shuffle=False)
        save_pkl_file(Path(r'.\training_torch_set_dl'), training_torch_set_dl)
        training_torch_set_dl = load_pkl_file(Path(r'.\training_torch_set_dl'))

        # %% 定义模型
        input_feature_len = training_torch_set[0][0].size()[-1]
        input_sequence_len = training_torch_set[0][0].size()[-2]

        output_feature_len = training_torch_set[0][1].size()[-1]
        output_sequence_len = training_torch_set[0][1].size()[-2]

        # lstm_encoder = GRUEncoder(
        #     gru_layer_num=lstm_layer_num,
        #     input_feature_len=input_feature_len,
        #     sequence_len=input_sequence_len,
        #     hidden_size=hidden_size,
        #     bidirectional=False,
        #     dropout=dropout
        # )
        #
        # lstm_decoder = GRUDecoder(
        #     gru_layer_num=lstm_layer_num,
        #     output_feature_len=output_feature_len,
        #     hidden_size=hidden_size,
        #     dropout=dropout,
        #     decoder_input_feature_len=hidden_size,
        #     attention_units=32,
        # )
        #
        # simple_lstm_model = GRUEncoderDecoderWrapper(
        #     gru_encoder=lstm_encoder,
        #     gru_decoder=lstm_decoder,
        #     output_sequence_len=output_sequence_len,
        #     output_feature_len=output_feature_len,
        #     teacher_forcing=0.01
        # )

        simple_lstm_model = StackedBiLSTM(lstm_layer_num=lstm_layer_num,
                                          input_feature_len=input_feature_len,
                                          hidden_size=hidden_size,
                                          output_feature_len=output_feature_len,
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
            epoch_start_time = time.time()

            simple_lstm_model.set_train()
            batch_loss = []
            for index, (xb, yb) in enumerate(training_torch_set_dl):
                pred = simple_lstm_model(xb)

                ##############################################################################
                # simple_lstm_model.set_eval()
                # pred = simple_lstm_model(xb)
                # series(xb[0, :, -1].detach().cpu().numpy().flatten(), label='X', figure_size=(10, 2.4))
                # ax = series(yb[0, :, -1].detach().cpu().numpy().flatten(), label='Truth', figure_size=(10, 2.4))
                # ax = series(pred[0, :, -1].detach().cpu().numpy().flatten(), ax=ax, label='LSTM', figure_size=(10, 2.4))
                ##############################################################################

                loss = loss_func(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                batch_loss.append(loss.item())
                print(f"第{i + 1: d}个epoch, 第{index + 1: d}个batch, loss={loss}")

            print(f"第{i + 1: d}个epoch结束, 平均loss={np.mean(batch_loss)}")
            print(f"第{i + 1: d}个epoch结束, 耗时{time.time() - epoch_start_time}")
            epoch_loss.append(batch_loss)
            if (i % 1000 == 0) and (i != 0):
                try_to_find_folder_path_otherwise_make_one(model_save_path.parent)
                torch.save(
                    simple_lstm_model,
                    model_save_path.parent / (model_save_path.stem + f"_epoch_{i}" + model_save_path.suffix)
                )

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
    model.set_eval()
    # 载入测试集
    test_torch_set = energies_paper_prepare_dataset_for_torch_model_for_ampds2_dataset(
        appliance_original_name=appliance_original_name,
        appliance_type_name=appliance_type_name,
        sample_period=sample_period,
        transform_args_file_path=transform_args_file_path)[1]
    test_torch_set_dl = DataLoader(test_torch_set,
                                   batch_size=1,
                                   shuffle=False)
    # save_as_docx_buff = {(test_torch_set.data.index[0] + datetime.timedelta(days=i)).strftime("%Y_%m_%d"): [None, None]
    #                      for i in range(test_torch_set.__len__())}
    # max
    ave = []
    for i in range(test_torch_set.__len__()):
        ave.append(np.mean(test_torch_set[i][1].cpu().numpy()[:, 0]))
    index = np.argmax(ave)
    for index in range(test_torch_set.__len__()):
        xb = test_torch_set[index][0].unsqueeze(0)
        yb = test_torch_set[index][1]
        xb_raw = test_torch_set.get_raw_data(index)[0]
        yb_raw = test_torch_set.get_raw_data(index)[1]
        pred = model(xb).squeeze(0)

        dim = 1
        title = (appliance_original_name or appliance_type_name) if dim == 1 else 'Total'
        x_plot = yb_raw.index.to_pydatetime()

        ax = series(x_plot,
                    pred[:, dim].detach().cpu().numpy().flatten(), label='LSTM', figure_size=(10, 2.4))

        ax = series(
            x_plot,
            yb[:, dim].cpu().numpy().flatten(),
            linestyle='--',
            ax=ax,
            label='Truth',
            x_label='Time',
            y_label='Normlised Active Power [p.u.]',
            figure_size=(10, 2.4),
            title=title,
        )

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
    # tt = energies_paper_load_ampds2_dataset_train_np_and_test_np()
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

    _sample_period = 300
    # energies_paper_train_torch_model_for_ampds2_dataset(
    #     appliance_original_name='HPE',
    #     model_save_path=Path(project_path_) / 'Data/Results/Energies_paper/Ampds2/lstm/forecast/input_21_days/HPE_IN/'
    #                                           f'HPE_and_total_{_sample_period}.pkl',
    #     sample_period=_sample_period,
    #     transform_args_file_path=Path(project_path_) / 'Data/Results/Energies_paper/Ampds2/forecast/input_21_days/'
    #                                                    f'HPE_IN/HPE_and_total_{_sample_period}_transform_args.pkl'
    # )

    energies_paper_test_torch_model_for_ampds2_dataset(
        appliance_original_name='HPE',
        model_save_path=Path(project_path_) / 'Data/Results/Energies_paper/Ampds2/lstm/forecast/input_21_days/'
                                              f'HPE_and_total_{_sample_period}.pkl',
        sample_period=_sample_period,
        transform_args_file_path=Path(project_path_) / 'Data/Results/Energies_paper/Ampds2/lstm/forecast/input_21_days/'
                                                       f'HPE_and_total_{_sample_period}_transform_args.pkl'
    )

    # energies_paper_train_tf_model_for_ampds2_dataset(
    #     Path(project_path_) / 'Data/Results/Energies_paper/Ampds2/lstm/forecast/input_week/'
    #                           'TensorFlow_HPE_and_total_60.pkl'
    # )
