import torch.nn as nn
from torch.autograd import Variable
import torch


class LSTMClass(nn.Module):
    def __init__(self, hp, input_dim, output_dim):
        super(LSTMClass, self).__init__()
        self.lstm = nn.LSTM(input_dim, hp.lstm_hid_size,
                            hp.n_lstm_layers, batch_first=True)
        self.output = nn.Linear(hp.lstm_hid_size, output_dim)
        self.hp = hp
        self.init_hidden_states()

    def forward(self, x):

        # out: batch_size x seq_len x hidden_size, final_hid_state: num_layers x batch_size x hidden_size (same for final_cell_state). Out[-1] is the last hidden state of the sequence, which is the same for final_hidden_state
        out, (final_hidden_state, final_cell_state) = self.lstm(
            x, (self.h_0, self.c_0))
        # take only last time step
        out = self.output(out[:, -1])
        return out

    def init_hidden_states(self):
        # Initialize hidden states and we make them trainable.
        # initial hiideen state: num_layers x batch_size x  hidden_size
        self.h_0 = nn.Parameter(torch.zeros(
            self.hp.n_lstm_layers, self.hp.batch_size, self.hp.lstm_hid_size))
        # initial cell state: num_layers x batch_size x hidden_size
        self.c_0 = nn.Parameter(torch.zeros(
            self.hp.n_lstm_layers, self.hp.batch_size, self.hp.lstm_hid_size))
