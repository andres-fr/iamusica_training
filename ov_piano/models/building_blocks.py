#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
PyTorch building blocks.
"""


import torch
#
from ..train import init_weights


# ##############################################################################
# # WRAPPERS
# ##############################################################################
class Permuter(torch.nn.Module):
    """
    Module wrapper to permute tensor axes.
    """

    def __init__(self, *permutation):
        """
        :param permutation: Sequence of axis IDs to permute. Usage example:
          ``Permuter(0, 2, 1, 3)``
        """
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        """
        """
        return x.permute(self.permutation)


def get_relu(leaky_slope=None):
    """
    :returns: a ReLU if ``leaky_slope`` is ``None``, otherwise a leaky ReLU
      with the given slope (must be a float).
    """
    if leaky_slope is None:
        result = torch.nn.ReLU(inplace=True)
    else:
        result = torch.nn.LeakyReLU(leaky_slope, inplace=True)
    return result


# ##############################################################################
# # POST-PROCESSING
# ##############################################################################
class GaussianBlur1d(torch.nn.Module):
    """
    Performs 1D gaussian convolution along last dimension of ``(b, c, t)``
    tensors.
    """

    @staticmethod
    def gaussian_1d_kernel(ksize=15, stddev=3.0, mean_offset=0, rbf=False,
                           dtype=torch.float32):
        """
        :param mean_offset: If 0, the mean of the gaussian is at the center of
          the kernel. So peaks at index ``t`` will appear at idx ``t+offset``
          when convolved
        :param rbf: If true, ``max(kernel)=1`` instead of ``sum(kernel)=1``
        """
        x = torch.linspace(-(ksize - 1) / 2., (ksize - 1) / 2., ksize,
                           dtype=dtype)
        x += mean_offset
        x.div_(stddev).pow_(2).mul_(-0.5).exp_()
        if not rbf:
            x.div_(x.sum())
        return x

    def __init__(self, num_chans, ksize=15, stddev=3.0, mean_offset=0,
                 rbf=False):
        """
        """
        assert ksize % 2 == 1, "Only odd ksize supported!"
        super().__init__()
        self.ksize = ksize
        self.stddev = stddev
        self.blur_fn = torch.nn.Conv1d(
            num_chans, num_chans, ksize, padding=ksize // 2,
            groups=num_chans, bias=False)
        #
        with torch.no_grad():
            # create 1d gaussian kernel and reshape to match the conv1d weight
            kernel = self.gaussian_1d_kernel(
                ksize, stddev, rbf=rbf, mean_offset=mean_offset,
                dtype=self.blur_fn.weight.dtype)
            kernel = kernel[None, :].repeat(num_chans, 1).unsqueeze(1)
            # assign kernel to weight
            self.blur_fn.weight[:] = kernel

    def forward(self, x):
        """
        """
        x = self.blur_fn(x)
        return x


class Nms1d(torch.nn.Module):
    """
    PT-compatible NMS, 1-dimensional along the last axis. Note that any
    non-zero entry that equals the maximum among the ``pool_ksize`` vicinity
    is considered a maximum. This includes if multiple maxima are present in
    a vicinity (even if disconnected), and particularly if all values are equal
    and non-zero
    """

    def __init__(self, pool_ksize=3):
        """
        """
        super().__init__()
        self.nms_pool = torch.nn.MaxPool1d(
            pool_ksize, stride=1, padding=pool_ksize // 2, ceil_mode=False)

    def forward(self, onset_preds, thresh=None):
        """
        :param onset_preds: Batch of shape ``(b, chans, t)``
        :param thresh: Any values below this will also be zeroed out
        """
        x = self.nms_pool(onset_preds)
        x = (onset_preds == x)
        x = onset_preds * x
        if thresh is not None:
            x = x * (x >= thresh)
        return x


# ##############################################################################
# # NORM LAYERS
# ##############################################################################
class SubSpectralNorm(torch.nn.Module):
    """
    Modified from https://arxiv.org/pdf/2103.13620.pdf
    This torch module reshapes the ``(b, c, f, t)`` batch into frequency
    sub-bands and applies batch normalization to those, resulting in frequency-
    specific BNs. See reference above for more details
    """

    def __init__(self, C, F, S, momentum=0.1, eps=1e-5):
        """
        :param C: Channels in batch ``(N, C, F, T)``
        :param S: Number of subbands such that ``(N, C*S, F//S, T)``
        """
        super().__init__()
        self.S = S
        self.eps = eps
        self.bn = torch.nn.BatchNorm2d(C * S, momentum=momentum)
        assert divmod(F, S)[1] == 0, "S must divide F exactly!"

    def forward(self, x):
        """
        """
        # x: input features with shape {N, C, F, T}
        # S: number of sub-bands
        N, C, F, T = x.size()
        x = x.reshape(N, C * self.S, F // self.S, T)
        x = self.bn(x)
        return x.reshape(N, C, F, T)


# #############################################################################
# # CONTEXT-AWARE MODULE
# #############################################################################
class SELayer(torch.nn.Module):
    """
    Implementation of the squeeze-excitation module from
    https://arxiv.org/pdf/1709.01507.pdf

    Since convolutions work with channels, this class is thought as an adaptive
    channel-attention mechanism. First, per-channel global pool is performed,
    then, results are sent through a single-hidden layer MLP ending with a
    sigmoid, then each channel is multiplied by the sigmoid output.
    """
    def __init__(self, in_chans, hidden_chans=None, out_chans=None,
                 bn_momentum=0.1):
        super().__init__()
        if hidden_chans is None:
            hidden_chans = in_chans // 4
        if out_chans is None:
            out_chans = in_chans
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)  # output a scalar per ch
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_chans, hidden_chans, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_chans, out_chans, bias=True),
            torch.nn.Sigmoid()
        )

    def set_biases(self, val=0):
        """
        """
        self.apply(lambda module: init_weights(
            module, init_fn=None, bias_val=val))

    def forward(self, x):
        """
        :param x: Input batch of shape ``(b, ch, h, w)``
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x)[:, :, 0, 0]  # (b, c)
        y = self.fc(y)[:, :, None, None]
        return y  # (b, c, 1, 1)


class ContextAwareModule(torch.nn.Module):
    """
    Context-Aware Module from https://arxiv.org/pdf/1910.12223.pdf adapted for
    the processing of spectrograms.
    """
    def __init__(self,
                 in_chans, hdc_chans=None, se_bottleneck=None,
                 ksizes=((3, 5), (3, 5), (3, 5), (3, 5)),
                 dilations=((1, 1), (1, 2), (1, 3), (1, 4)),
                 paddings=((1, 2), (1, 4), (1, 6), (1, 8)),
                 bn_momentum=0.1):
        """
        """
        super().__init__()
        #
        assert len(ksizes) == len(dilations) == len(paddings), \
            "ksizes, dilations and paddings must have same length!"
        num_convs = len(ksizes)
        #
        if hdc_chans is None:
            hdc_chans = in_chans // num_convs
        hdc_out_chans = hdc_chans * num_convs
        if se_bottleneck is None:
            se_bottleneck = in_chans // 4
        #
        self.se = SELayer(in_chans, se_bottleneck, hdc_out_chans, bn_momentum)
        #
        self.hdcs = torch.nn.ModuleList(
            [torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_chans, hdc_chans, stride=1,
                    kernel_size=ks, dilation=dil, padding=pad,
                    bias=False),
                torch.nn.BatchNorm2d(hdc_chans, momentum=bn_momentum),
                torch.nn.ReLU(inplace=True))
             for ks, dil, pad in zip(ksizes, dilations, paddings)])
        #
        self.skip = torch.nn.Sequential(
            torch.nn.Conv2d(in_chans, hdc_out_chans, kernel_size=1,
                            bias=False),
            torch.nn.BatchNorm2d(hdc_out_chans, momentum=bn_momentum),
            torch.nn.ReLU(inplace=True))

    def forward(self, x):
        """
        :param x: Input batch of shape ``(b, ch, h, w)``
        """
        se_att = self.se(x)  # (b, hdc, 1, 1)
        skip = self.skip(x)  # (b, hdc, h, w)
        hdc = torch.cat([hdc(x) for hdc in self.hdcs], dim=1)  # (b, hdc, h, w)
        #
        x = skip + (hdc * se_att)
        return x


# ##############################################################################
# # RESHAPER CONVS
# ##############################################################################
def conv1x1net(hid_chans, bn_momentum=0.1, last_layer_bn_relu=False,
               dropout_drop_p=None, leaky_relu_slope=None,
               kernel_width=1):
    """
    :param hid_chans: List of integers in the form ``(200, 100, ...)``,
      providing the successive hidden dimensionalities.
    :param last_layer_bn_relu: Whether to append BN and ReLU after last layer.
    :param kernel_width: Width of the convolutions across dimension ``t``. Must
      be an odd integer, and the padding will be adjusted so that ``t`` is
      preserved.
    :returns: A ``torch.nn.Sequential`` module that expects a batch of shape
      ``(b, hid_chans[0], 1, t)`` and returns a batch of shape
      ``(b, hid_chans[-1], 1, t)``. Effectively, it acts as a MLP applied at
      each timepoint ``t``, with a fully connected layer followed by BN,
      (leaky) ReLU and optionally dropout.
    """
    assert (kernel_width % 2) == 1, "Only odd kwidth supported!"
    wpad = kernel_width // 2
    #
    result = torch.nn.Sequential()
    n_layers = len(hid_chans) - 1
    for i, (h_in, h_out) in enumerate(zip(hid_chans[:-1],
                                          hid_chans[1:]), 1):
        if (i < n_layers) or ((i == n_layers) and last_layer_bn_relu):
            result.append(torch.nn.Conv2d(h_in, h_out, (1, kernel_width),
                                          padding=(0, wpad), bias=False))
            result.append(torch.nn.BatchNorm2d(h_out, momentum=bn_momentum))
            result.append(get_relu(leaky_relu_slope))
            if dropout_drop_p is not None:
                result.append(torch.nn.Dropout(dropout_drop_p, inplace=False))
        else:
            result.append(torch.nn.Conv2d(h_in, h_out, (1, kernel_width),
                                          padding=(0, wpad), bias=True))
    #
    return result


class DepthwiseConv2d(torch.nn.Module):
    """
    For input spectrogram ``(b, ch_in, h_in, t)``, the DepthwiseConv can be
    implemented as follows (assuming ``same`` padding):
    ``torch.nn.Conv2d(ch_in, ch_out*K, (h_in, kt), groups=ch_in)`` followed
    by ``squeeze(2)`` and ``reshape(b, K, ch_out, t)``. This is so because the
    convolution yields``(b, K*ch_out, 1, t)``.

    Note that the conv filter has dimensionality ``(K*ch_out, 1, h_in, 1)``,
    but each of the ``K`` segments is applied to a separate channel, due to
    having ``groups=ch_in``.

    More info: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """

    def __init__(self, ch_in, ch_out, h_in, h_out, kernel_width=1, bias=True):
        """
        """
        super().__init__()
        self.ch_in, self.ch_out = ch_in, ch_out
        self.h_in, self.h_out = h_in, h_out
        #
        assert (kernel_width % 2) == 1, "Only odd kwidth supported!"
        self.conv = torch.nn.Conv2d(
            ch_in, ch_out * h_out,
            (h_in, kernel_width), padding=(0, kernel_width // 2), groups=ch_in,
            bias=bias)

    def forward(self, x):
        """
        :param x: Batch tensor of shape ``(b, ch_in, h_in, t)``.
        :returns: Batch tensor of shape ``(b, ch_out, h_out, t)``
        """
        b, ch_in, h_in, t = x.shape
        x = self.conv(x)
        x = x.squeeze(2).reshape(b, self.ch_out, self.h_out, -1)
        return x
