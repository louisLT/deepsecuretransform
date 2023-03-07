"""
Author : Louis Le Tarnec
"""

import torch.nn as nn

class EncoderModule(nn.Module):

    def __init__(self, input_channels, output_channels, stride, kernel, pad):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel, padding=pad, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DecoderModule(nn.Module):

    def __init__(self, input_channels, output_channels, stride, activation="relu", apply_bn=True):
        super().__init__()
        self.convt = nn.ConvTranspose2d(input_channels,
                                        output_channels,
                                        kernel_size=(1, stride),
                                        stride=(1, stride))
        self.bn = nn.BatchNorm2d(output_channels)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            assert False, "unknown activation : %s" % activation
        self.apply_bn = apply_bn

    def forward(self, x):
        return self.activation(self.bn(self.convt(x)) if self.apply_bn else self.convt(x))

class Encoder(nn.Module):

    def __init__(self, color_channels, bottleneck_nb_channels, encoder_output_size, n_latent_features):
        self.bottleneck_nb_channels = bottleneck_nb_channels
        self.n_neurons_in_middle_layer = bottleneck_nb_channels * encoder_output_size
        super().__init__()
        self.bottle = EncoderModule(color_channels, 4, stride=1, kernel=1, pad=0)
        self.m1 = EncoderModule(4, 8, stride=1, kernel=3, pad=1)
        self.m2 = EncoderModule(8, 16, stride=2, kernel=3, pad=1)
        self.m3 = EncoderModule(16, self.bottleneck_nb_channels, stride=2, kernel=3, pad=1)
        self.fc = nn.Linear(self.n_neurons_in_middle_layer, n_latent_features)
    def forward(self, x):
        out = self.m3(self.m2(self.m1(self.bottle(x))))
        return self.fc(out.view(-1, self.n_neurons_in_middle_layer))

class Decoder(nn.Module):

    def __init__(self, color_channels, bottleneck_nb_channels, encoder_output_size, n_latent_features):
        self.encoder_output_size = encoder_output_size
        self.bottleneck_nb_channels = bottleneck_nb_channels
        self.n_neurons_in_middle_layer = bottleneck_nb_channels * encoder_output_size
        super().__init__()
        self.m1 = DecoderModule(32, 16, stride=1)
        self.m2 = DecoderModule(16, 8, stride=2)
        self.m3 = DecoderModule(8, 4, stride=2)
        self.bottle = DecoderModule(4, color_channels, stride=1, activation="sigmoid")
        self.fc = nn.Linear(n_latent_features, self.n_neurons_in_middle_layer)

    def forward(self, x):
        x = self.fc(x)
        out = x.view(-1, self.bottleneck_nb_channels, 1, self.encoder_output_size)
        out = self.m3(self.m2(self.m1(out)))
        return self.bottle(out)