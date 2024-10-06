import torch
import torch.nn as nn


class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wi = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.Wh = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bi = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.bh = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.init_weights()

    def init_weights(self):
        std_v = 1.0 / ((self.hidden_size + self.input_size) ** 0.5)
        for weight in self.parameters():
            weight.data.uniform_(-std_v, std_v)

    def forward(self, x, hidden):
        # both should have size (batch, hidden)
        h_prev, c_prev = hidden
        gates = torch.mm(x, self.Wi) + self.bi + torch.mm(h_prev, self.Wh) + self.bh
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)
        c_cur = (forget_gate * c_prev) + (input_gate * cell_gate)
        h_cur = output_gate * torch.tanh(c_cur)
        return h_cur, c_cur


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cells = nn.ModuleList(
            [CustomLSTMCell(input_size, hidden_size) if i == 0 else CustomLSTMCell(hidden_size, hidden_size) for i in
             range(num_layers)])
        self.fc = nn.Linear(hidden_size, output_size)

        self.device = device

    def forward(self, x, hidden):
        # both should have shape (layer_index, batch, hidden), x should be a single timestamp x
        h_prev, c_prev = hidden
        h_out, c_out = [], []

        x_in = x
        for layer in range(self.num_layers):
            h_cur, c_cur = self.lstm_cells[layer](x_in, (h_prev[layer], c_prev[layer]))
            h_out.append(h_cur)
            c_out.append(c_cur)

            # x for next layer will be h_cur from previous layer
            x_in = h_cur

        h_out = torch.stack(h_out, dim=0)
        c_out = torch.stack(c_out, dim=0)
        y = self.fc(h_out[-1])
        return y, h_out, c_out

    def init_hidden(self, batch_size):
        # Initializing hidden state with zeros
        h_prev = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c_prev = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return h_prev, c_prev
