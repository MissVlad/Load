import numpy as np
import torch
from pathlib import Path
import pickle

FILE_PATH = Path(
    "".join((r"C:\Users\SoapClancy\OneDrive\PhD\01-PhDProject\06-Load\MyProject\Data\Results\\",
             r"\Energies_paper\Ampds2\lstm\forecast\input_week\sets_HPE_and_total_60_transform_args.pkl"))
)


class LSTMEncoder(torch.nn.Module):
    def __init__(self, *, lstm_layer_num: int = 1,
                 input_feature_len: int,
                 output_feature_len: int,
                 sequence_len: int,
                 hidden_size: int,
                 bidirectional: bool = False,
                 dropout: float = 0.1,
                 device='cuda'):
        super().__init__()
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.input_feature_len = input_feature_len
        self.output_feature_len = output_feature_len
        self.lstm_layer_num = lstm_layer_num
        self.direction = 2 if bidirectional else 1

        self.lstm_layer = torch.nn.LSTM(input_feature_len,
                                        hidden_size,
                                        num_layers=lstm_layer_num,
                                        batch_first=True,
                                        bidirectional=bidirectional,
                                        dropout=dropout)
        self.device = device
        if "cuda" in device:
            self.cuda()

    def forward(self, x):
        lstm_output, (h_n, c_n) = self.lstm_layer(x)

        # lstm_output sum-reduced by direction
        lstm_output = lstm_output.view(x.size(0), self.sequence_len, self.direction, self.hidden_size)
        lstm_output = lstm_output.sum(2)

        # lstm_states sum-reduced by direction
        h_n = h_n.view(self.lstm_layer_num, self.direction, x.size(0), self.hidden_size)
        c_n = c_n.view(self.lstm_layer_num, self.direction, x.size(0), self.hidden_size)

        # Only use the information from the last layer
        h_n, c_n = h_n[-1], c_n[-1]
        h_n, c_n = h_n.sum(0), c_n.sum(0)

        return lstm_output, (h_n, c_n)

    def init_hidden(self, batch_size: int) -> tuple:
        h_0 = torch.zeros(self.lstm_layer_num * self.direction, batch_size, self.hidden_size, device=self.device)
        c_0 = torch.zeros(self.lstm_layer_num * self.direction, batch_size, self.hidden_size, device=self.device)

        return h_0, c_0


class Attention(torch.nn.Module):
    def __init__(self,
                 lstm_encoder_hidden_size,
                 units: int,
                 device="cuda"):
        super().__init__()
        self.W1 = torch.nn.Linear(lstm_encoder_hidden_size, units)
        self.W2 = torch.nn.Linear(lstm_encoder_hidden_size, units)
        self.V = torch.nn.Linear(units, 1)

        self.device = device
        if "cuda" in device:
            self.cuda()

    def forward(self, lstm_encoder_output, lstm_encoder_h_n):
        score = self.V(torch.tanh(self.W1(lstm_encoder_h_n.unsqueeze(1)) + self.W2(lstm_encoder_output)))
        attention_weights = torch.nn.functional.softmax(score, 1)

        context_vector = attention_weights * lstm_encoder_output
        context_vector = context_vector.sum(1)
        return context_vector, attention_weights


class LSTMDecoder(torch.nn.Module):
    def __init__(self, *,
                 lstm_num_layers: int = 1,
                 decoder_input_feature_len: int,
                 output_feature_len: int,
                 hidden_size: int,
                 lstm_hidden_size: int,
                 bidirectional: bool = False,
                 dropout: float = 0.1,
                 attention_units: int = 128,
                 device="cuda"):
        super().__init__()
        self.lstm_layer = torch.nn.LSTM(decoder_input_feature_len,
                                        hidden_size,
                                        num_layers=lstm_num_layers,
                                        batch_first=True,
                                        bidirectional=bidirectional,
                                        dropout=dropout)

        self.attention = Attention(lstm_hidden_size, attention_units)

        self.out = torch.nn.Linear(hidden_size, output_feature_len)

        self.device = device
        if "cuda" in device:
            self.cuda()

    def forward(self, y, lstm_encoder_output, lstm_encoder_h_n):
        context_vector, attention_weights = self.attention(lstm_encoder_output, lstm_encoder_h_n)
        y = torch.cat((context_vector.unsqueeze(1), y.unsqueeze(1)), -1)
        lstm_output, (h_n, c_n), = self.lstm_layer(y)
        output = self.out(lstm_output.squeeze(1))
        return output, (h_n.squeeze(0), c_n.squeeze(0)), attention_weights


class LSTMEncoderDecoderWrapper(torch.nn.Module):
    def __init__(self, *, lstm_encoder: LSTMEncoder,
                 lstm_decoder: LSTMDecoder,
                 output_sequence_len: int,
                 output_feature_len: int,
                 decoder_input=True,
                 teacher_forcing: float = 0.001,
                 device="cuda"):
        super().__init__()
        self.lstm_encoder = lstm_encoder  # type: LSTMEncoder
        self.lstm_decoder = lstm_decoder  # type: LSTMDecoder
        self.output_sequence_len = output_sequence_len
        self.output_feature_len = output_feature_len
        self.decoder_input = decoder_input
        self.teacher_forcing = teacher_forcing
        self.device = device
        if "cuda" in device:
            self.cuda()

    def forward(self, x, y=None):
        if y is None:
            y_teacher_forcing = torch.zeros((x.size(0), self.output_sequence_len, self.output_feature_len),
                                            device=self.device)
        else:
            y_teacher_forcing = y.clone()
            mask = torch.rand(y_teacher_forcing.size()) > self.teacher_forcing
            y_teacher_forcing[:, 1:, :] = y[:, :-1, :]
            y_teacher_forcing[mask] = 0
            y_teacher_forcing[:, 0, :] = 0

        encoder_output, (encoder_h_n, encoder_c_n) = self.lstm_encoder(x)
        decoder_h_n = encoder_h_n
        outputs = torch.zeros(y_teacher_forcing.size(), device=self.device)
        for i in range(y_teacher_forcing.size(1)):
            this_time_step_output, (decoder_h_n, _), _ = self.lstm_decoder(y_teacher_forcing[:, i, :],
                                                                           encoder_output,
                                                                           decoder_h_n)
            outputs[:, i, :] = this_time_step_output
        return outputs

    def set_train(self):
        self.lstm_encoder.train()
        self.lstm_decoder.train()
        self.train()

    def set_eval(self):
        self.lstm_encoder.eval()
        self.lstm_decoder.eval()
        self.eval()


def load_data_sets():
    with open(str(FILE_PATH), "rb") as f:
        return pickle.load(f)['train']


def do_training(_model_save_path: Path):
    training_set = load_data_sets()
    tt = 1


if __name__ == "__main__":
    model_save_path = Path(r"./seq2seq_forecast/torch_model")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    do_training(model_save_path)
