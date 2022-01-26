import torch
import torch.nn as nn
import numpy as np
from model.convolution_lstm import ConvLSTM

def attention(out):
    t_out = out.transpose(0, 1)
    s, b, _, _ ,_ = t_out.shape
    
    att = []

    for k in range(s):
        inner = torch.mul(t_out[k], t_out[-1]).reshape(b, -1)
        inner = inner.sum(1) / s
        att.append(inner)

    att = torch.softmax(torch.stack(att).T, 1)

    b, s, c, w, h = out.shape

    att = att.reshape(b, s, 1)
    out = out.reshape(b, s, -1)

    out = (out * att).sum(1).reshape(b, c, w, h)
    
    return out

class CnnEncoder(nn.Module):
    def __init__(self, in_channels_encoder):
        super(CnnEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels_encoder, 32, 3, (1, 1), 1),
            nn.SELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, (2, 2), 1),
            nn.SELU()
        )    
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 2, (2, 2), 1),
            nn.SELU()
        )   
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 2, (2, 2), 0),
            nn.SELU()
        )


    def forward(self, X):
        b, s, c, w, h = X.shape
        X = X.reshape(-1, c, w, h)

        conv1_out = self.conv1(X)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)

        def reshape_out(x):
            _, x_c, x_w, x_h = x.shape
            return x.reshape(b, s, x_c, x_w, x_h)

        conv1_out, conv2_out, conv3_out, conv4_out = map(
            reshape_out, [conv1_out, conv2_out, conv3_out, conv4_out]
        )

        return conv1_out, conv2_out, conv3_out, conv4_out


class Conv_LSTM(nn.Module):
    def __init__(self):
        super(Conv_LSTM, self).__init__()
        self.conv1_lstm = ConvLSTM(input_channels=32, hidden_channels=[32], 
                                   kernel_size=3, step=5, effective_step=[4])
        self.conv2_lstm = ConvLSTM(input_channels=64, hidden_channels=[64], 
                                   kernel_size=3, step=5, effective_step=[4])
        self.conv3_lstm = ConvLSTM(input_channels=128, hidden_channels=[128], 
                                   kernel_size=3, step=5, effective_step=[4])
        self.conv4_lstm = ConvLSTM(input_channels=256, hidden_channels=[256], 
                                   kernel_size=3, step=5, effective_step=[4])

    def forward(self, out1, out2, out3, out4):
        (l_out1,), _ = self.conv1_lstm(out1)
        l_out1 = attention(l_out1)
        (l_out2,), _ = self.conv2_lstm(out2)
        l_out2 = attention(l_out2)
        (l_out3,), _ = self.conv3_lstm(out3)
        l_out3 = attention(l_out3)
        (l_out4,), _ = self.conv4_lstm(out4)
        l_out4 = attention(l_out4)
        return l_out1, l_out2, l_out3, l_out4

class CnnDecoder(nn.Module):
    def __init__(self, in_channels):
        super(CnnDecoder, self).__init__()
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, 2, 2, 0, 0),
            nn.SELU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 2, 2, 1, 1),
            nn.SELU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 3, 2, 1, 1),
            nn.SELU()
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 3, 1, 1, 0),
            nn.SELU()
        )
    
    def forward(self, out1, out2, out3, out4):
        deconv4 = self.deconv4(out4)
        deconv4_concat = torch.cat((deconv4, out3), dim = 1)
        deconv3 = self.deconv3(deconv4_concat)
        deconv3_concat = torch.cat((deconv3, out2), dim = 1)
        deconv2 = self.deconv2(deconv3_concat)
        deconv2_concat = torch.cat((deconv2, out1), dim = 1)
        deconv1 = self.deconv1(deconv2_concat)
        return deconv1


class MSCRED(nn.Module):
    def __init__(self, in_channels_encoder, in_channels_decoder):
        super(MSCRED, self).__init__()
        self.cnn_encoder = CnnEncoder(in_channels_encoder)
        self.conv_lstm = Conv_LSTM()
        self.cnn_decoder = CnnDecoder(in_channels_decoder)
    
    def forward(self, x):
        encoding = self.cnn_encoder(x)
        lstm_encoding = self.conv_lstm(*encoding)
        decoding = self.cnn_decoder(*lstm_encoding)
        return decoding
