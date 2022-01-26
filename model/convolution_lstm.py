import torch
import torch.nn as nn
from torch.autograd import Variable

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, internal_state):
        h, c = internal_state

        x_b, x_s, x_c, x_w, x_h = x.shape
        x = x.reshape(-1, x_c, x_w, x_h)

        _, _, h_c, h_w, h_h = h.shape
        h = h.reshape(-1, h_c, h_w, h_h)

        _, _, c_c, c_w, c_h = c.shape
        c = c.reshape(-1, c_c, c_w, c_h)

        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        
        _, cc_c, cc_w, cc_h = cc.shape
        cc = cc.reshape(x_b, x_s, cc_c, cc_w, cc_h)

        _, ch_c, ch_w, ch_h = ch.shape
        ch = ch.reshape(x_b, x_s, ch_c, ch_w, ch_h)

        return ch, cc

    def init_hidden(self, x, hidden):
        b, s, _, w, h = x.shape
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, w, h)).to(x.device)
            self.Wcf = Variable(torch.zeros(1, hidden, w, h)).to(x.device)
            self.Wco = Variable(torch.zeros(1, hidden, w, h)).to(x.device)
        else:
            assert w == self.Wci.size()[2], 'Input Height Mismatched!'
            assert h == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(b, s, hidden, w, h)).to(x.device),
                Variable(torch.zeros(b, s, hidden, w, h)).to(x.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self.conv_lstm_cells = nn.ModuleList([
            ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            for i in range(self.num_layers)
        ])

    def forward(self, x_):
        internal_states = []
        outputs = []

        for i in range(self.num_layers):
            internal_state = self.conv_lstm_cells[i].init_hidden(x=x_, hidden=self.hidden_channels[i])
            internal_states.append(internal_state)

        for step in range(self.step):
            x = x_
    
            for i in range(self.num_layers):
                internal_state = internal_states[i]
                x, new_c = self.conv_lstm_cells[i](x, internal_state)
                internal_states[i] = (x, new_c)

            if step in self.effective_step:
                outputs.append(x)
    
        return outputs, (x, new_c)