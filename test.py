import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
lstm = nn.LSTM(1, 1)  # Input dim is 3, output dim is 3
cell = nn.LSTMCell(1, 1)
# inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5
#
# # initialize the hidden state.
# hidden = (torch.randn(1, 1, 3),
#           torch.randn(1, 1, 3))
