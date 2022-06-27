import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):

    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='batchnorm', activation="ReLU",
                 kernel_size=3, padding=1,
                 expand_chan=False):
        super(ResidualConvBlock, self).__init__()

        activation = getattr(nn.modules.activation, activation)

        self.expand_chan = expand_chan
        if self.expand_chan:
            ops = [nn.Conv3d(n_filters_in, n_filters_out, 1)]
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            ops.append(activation())
            self.conv_expan = nn.Sequential(*ops)

        ops = []
        for i in range(n_stages):
            if normalization != 'none':
                ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size, padding=padding))
                if normalization == 'batchnorm':
                    ops.append(nn.BatchNorm3d(n_filters_out))
            else:
                ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size, padding=padding))

            ops.append(activation(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        if self.expand_chan:
            x = self.conv(x) + self.conv_expan(x)
        else:
            x = (self.conv(x) + x)
        return x


class DownsamplingConvBlock(nn.Module):

    def __init__(self, n_filters_in, n_filters_out, normalization='batchnorm', activation="ReLU", stride=2, padding=0):
        super(DownsamplingConvBlock, self).__init__()

        activation = getattr(nn.modules.activation, activation)

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=padding, stride=stride))

        ops.append(activation(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):

    def __init__(self, n_filters_in, n_filters_out, normalization='batchnorm', activation="ReLU", stride=2):
        super(UpsamplingDeconvBlock, self).__init__()

        activation = getattr(nn.modules.activation, activation)

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(activation(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class VNETEncoder(nn.Module):

    def __init__(self, n_channels, n_filters=16, depth=5, normalization='batchnorm', activation="ReLU", kernel_size=3,
                 padding=1):

        super(VNETEncoder, self).__init__()

        self.depth = depth

        def _pow(l):
            return 2 ** l

        for level in range(self.depth):
            n_repeats = max(min(level + 1, 3), 1)

            if level == 0:
                temp = ResidualConvBlock(n_repeats, n_channels, n_filters, normalization, activation, kernel_size,
                                         padding, expand_chan=n_channels > 1)
                setattr(self, "block_{}_enc".format(level + 1), temp)
            else:
                temp = ResidualConvBlock(n_repeats, n_filters * _pow(level), n_filters * _pow(level), normalization,
                                         activation, kernel_size, padding)
                setattr(self, "block_{}_enc".format(level + 1), temp)

            if level < self.depth - 1:
                temp = DownsamplingConvBlock(n_filters * _pow(level), n_filters * _pow(level + 1), normalization,
                                             activation)
                setattr(self, "block_{}_dw".format(level + 1), temp)

    def forward(self, x):
        encoder = dict()
        for level in range(1, self.depth + 1):
            x = x if level == 1 else encoder["x{}_dw".format(level - 1)]
            encoder["x{}_enc".format(level)] = getattr(self, "block_{}_enc".format(level))(x)
            if level < self.depth:
                encoder["x{}_dw".format(level)] = getattr(self, "block_{}_dw".format(level))(
                    encoder["x{}_enc".format(level)])
        return encoder


class VNETDecoder(nn.Module):

    def __init__(self, n_classes, n_filters=16, depth=5, normalization='batchnorm', activation="ReLU", kernel_size=3,
                 padding=1):
        super(VNETDecoder, self).__init__()

        self.depth = depth

        def _pow(l):
            return 2 ** l

        for level in range(self.depth, 0, -1):
            n_repeats = max(min(level, 3), 1)

            if level < self.depth:
                temp = ResidualConvBlock(n_repeats, n_filters * _pow(level - 1), n_filters * _pow(level - 1),
                                         normalization, activation, kernel_size, padding)
                setattr(self, "block_{}_dec".format(level), temp)

            if level > 1:
                temp = UpsamplingDeconvBlock(n_filters * _pow(level - 1), n_filters * _pow(level - 2), normalization,
                                             activation)
                setattr(self, "block_{}_up".format(level), temp)

        self.out_conv = nn.Conv3d(n_filters, n_classes, kernel_size, padding=padding)

    def forward(self, encoder):
        decoder = dict()
        x = None
        for level in range(self.depth, 0, -1):
            x = encoder["x{}_enc".format(level)] if level == self.depth else decoder["x{}_up".format(level + 1)]
            if level < self.depth:
                x = getattr(self, "block_{}_dec".format(level))(x)

            if level > 1:
                x = getattr(self, "block_{}_up".format(level))(x)
                decoder["x{}_up".format(level)] = x + encoder["x{}_enc".format(level - 1)]

        out_logits = self.out_conv(x)
        return out_logits


class VNet(nn.Module):

    def __init__(self, n_channels, n_classes, n_filters=16, depth=5, normalization='batchnorm', activation="ReLU",
                 kernel_size=3, padding=1):
        super(VNet, self).__init__()
        self.encoder = VNETEncoder(n_channels, n_filters, depth, normalization, activation, kernel_size, padding)
        self.decoder = VNETDecoder(n_classes, n_filters, depth, normalization, activation, kernel_size, padding)

    def forward(self, x):
        encoder = self.encoder(x)
        return self.decoder(encoder)
