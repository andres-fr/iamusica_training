#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This module hosts the main DNNs, making use of PyTorch built-ins and parts from
``building_blocks``.
"""


import torch
from .building_blocks import get_relu, SubSpectralNorm, Permuter
from .building_blocks import ContextAwareModule, DepthwiseConv2d, conv1x1net
from ..train import init_weights


# ##############################################################################
# # MAIN MODEL
# ##############################################################################
class OnsetVelocityNet(torch.nn.Module):
    """
    A DNN to perform onset and velocity detection. Main features:
    * Fully convolutional
    * Dilated+residual convolutional blocks
    * Multiple residual stages
    * Batch+SubSpectral normalization, dropout and leaky ReLU
    * Depthwise convolution to convert from spectrogram to key
    """

    @staticmethod
    def get_cam_stage(in_chans, out_bins, conv1x1=(400, 200, 100),
                      num_cam_bottlenecks=3, cam_hdc_chans=3,
                      cam_se_bottleneck=8,
                      cam_ksizes=((3, 21), (3, 17), (3, 13), (3, 9)),
                      cam_dilations=((1, 1), (1, 2), (1, 3), (1, 4)),
                      cam_paddings=((1, 10), (1, 16), (1, 18), (1, 16)),
                      bn_momentum=0.1, leaky_relu_slope=0.1, dropout_p=0.1,
                      summary_width=3, conv1x1_kw=1):
        """
        Retrieve a CAM stage, which is a ``torch.nn.Sequential``. Given a
        tensor of shape ``(b, c, h, t)``, returns ``(b, 1, out_bins, t)``
        performing the following operations:

        1. Conv2D (with BN and lReLU) to expand channels
        2. A sequence of ``num_cam_bottlenecks`` CAMs (with BN and lReLU)
        3. Conv2D (with BN and lReLU) to collapse height and channels into
          ``out_bins`` channels
        4. conv1x1net (with BN, dropout and lReLU among layers), to perform
          MLP-alike operations for each entry in dimension ``t``.
        5. Swap channels with height, and return result
        """
        cam_out_chans = cam_hdc_chans * len(cam_ksizes)
        cam = torch.nn.Sequential(
            # from (b, in, h, t) to (b, cam_out, h, t)
            torch.nn.Conv2d(in_chans, cam_out_chans, (1, 1),
                            padding=(0, 0), bias=False),
            torch.nn.BatchNorm2d(cam_out_chans, momentum=bn_momentum),
            get_relu(leaky_relu_slope),
            *[torch.nn.Sequential(
                # shape-preserving
                ContextAwareModule(
                    cam_out_chans, cam_hdc_chans, cam_se_bottleneck,
                    cam_ksizes, cam_dilations, cam_paddings, bn_momentum),
                torch.nn.BatchNorm2d(cam_out_chans, momentum=bn_momentum),
                get_relu(leaky_relu_slope))
              for _ in range(num_cam_bottlenecks)],
            # from (b, cam_out, h, t) to (b, first_hid, 1, t)
            torch.nn.Conv2d(
                cam_out_chans, conv1x1[0], (out_bins, summary_width),
                padding=(0, 1), bias=False),
            torch.nn.BatchNorm2d(conv1x1[0], momentum=bn_momentum),
            get_relu(leaky_relu_slope),
            # from (b, first_hid, 1, t) to (b, out_bins, 1, t)
            conv1x1net((*conv1x1, out_bins), bn_momentum,
                       last_layer_bn_relu=False,
                       dropout_drop_p=dropout_p,
                       leaky_relu_slope=leaky_relu_slope,
                       kernel_width=conv1x1_kw),
            # reshape to (b, 1, out_bins, t)
            Permuter(0, 2, 1, 3))
        #
        return cam  # (b, 1, out_bins, t)

    def __init__(self, in_bins, out_bins, bn_momentum=0.1,
                 conv1x1=(400, 200, 100),
                 init_fn=torch.nn.init.kaiming_normal_,
                 se_init_bias=1.0, dropout_drop_p=0.1,
                 leaky_relu_slope=0.1):
        """
        """
        super().__init__()

        #
        in_chans = 2
        #
        stem_num_cam_bottlenecks = 3
        stem_hdc_chans = 4
        stem_se_bottleneck = 8
        #
        stem_ksizes = ((3, 5), (3, 5), (3, 5), (3, 5), (3, 5))
        stem_dilations = ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5))
        stem_paddings = ((1, 2), (1, 4), (1, 6), (1, 8), (1, 10))
        stem_inner_chans = stem_hdc_chans * len(stem_ksizes)
        stem_out_chans = stem_inner_chans
        # #
        cam_num_bottlenecks = 3
        cam_hdc_chans = 4
        cam_se_bottleneck = 8
        #
        cam_ksizes = ((1, 21), (1, 17), (1, 13), (1, 9))
        cam_dilations = ((1, 1), (1, 2), (1, 3), (1, 4))
        cam_paddings = ((0, 10), (0, 16), (0, 18), (0, 16))
        # #
        num_refiner_stages = 3
        refiner_num_bottlenecks = 2
        refiner_hdc_chans = 3
        refiner_se_bottleneck = 8
        #
        refiner_ksizes = ((1, 9), (1, 7), (1, 5))
        refiner_dilations = ((1, 1), (1, 2), (1, 3))
        refiner_paddings = ((0, 4), (0, 6), (0, 6))
        #
        self.specnorm = SubSpectralNorm(2, in_bins, in_bins, bn_momentum)
        self.stem = torch.nn.Sequential(
            # lift in chans into stem chans
            torch.nn.Conv2d(in_chans, stem_inner_chans, (3, 3),
                            padding=(1, 1), bias=False),
            SubSpectralNorm(stem_inner_chans, in_bins, in_bins, bn_momentum),
            get_relu(leaky_relu_slope),
            # series of stem CAMs. Output: (b, stem_inner_chans, mels, t)
            *[torch.nn.Sequential(
                ContextAwareModule(
                    stem_inner_chans, stem_hdc_chans, stem_se_bottleneck,
                    stem_ksizes, stem_dilations, stem_paddings, bn_momentum),
                SubSpectralNorm(
                    stem_inner_chans, in_bins, in_bins, bn_momentum),
                get_relu(leaky_relu_slope))
              for _ in range(stem_num_cam_bottlenecks)],
            # reshape to ``(b, stem_inner_chans, keys, t)``
            DepthwiseConv2d(
                stem_inner_chans, stem_out_chans, in_bins, out_bins,
                kernel_width=1, bias=False),
            torch.nn.BatchNorm2d(stem_out_chans, momentum=bn_momentum),
            get_relu(leaky_relu_slope))
        #
        self.first_stage = self.get_cam_stage(
             stem_out_chans, out_bins, conv1x1,
             cam_num_bottlenecks, cam_hdc_chans, cam_se_bottleneck,
             cam_ksizes, cam_dilations, cam_paddings,
             bn_momentum, leaky_relu_slope, dropout_drop_p)
        #
        self.refiner_stages = torch.nn.ModuleList(
            [self.get_cam_stage(
                stem_out_chans, out_bins, conv1x1,
                refiner_num_bottlenecks, refiner_hdc_chans,
                refiner_se_bottleneck,
                refiner_ksizes, refiner_dilations, refiner_paddings,
                bn_momentum, leaky_relu_slope, dropout_drop_p)
             for _ in range(num_refiner_stages)])

        self.velocity_stage = torch.nn.Sequential(
            SubSpectralNorm(
                stem_out_chans + 1, out_bins, out_bins, bn_momentum),
            self.get_cam_stage(
                stem_out_chans + 1, out_bins, [out_bins * 2, out_bins],
                num_cam_bottlenecks=1, cam_hdc_chans=4,
                cam_ksizes=((1, 3), (1, 3), (1, 3), (1, 3)),
                cam_dilations=((1, 1), (1, 2), (1, 3), (1, 4)),
                cam_paddings=((0, 1), (0, 2), (0, 3), (0, 4)),
                bn_momentum=bn_momentum, leaky_relu_slope=leaky_relu_slope,
                dropout_p=dropout_drop_p, summary_width=3),
            SubSpectralNorm(1, out_bins, out_bins, bn_momentum))

        # initialize weights
        if init_fn is not None:
            self.apply(lambda module: init_weights(
                module, init_fn, bias_val=0.0))
        self.apply(lambda module: self.set_se_biases(module, se_init_bias))

    @staticmethod
    def set_se_biases(module, bias_val):
        """
        Wrapper to recursively call ``set_biases`` for the CAM submodules, and
        ignore otherwise. Used in constructor.
        """
        try:
            module.se.set_biases(bias_val)
        except AttributeError:
            pass  # ignore: not a CAM module

    def forward_onsets(self, x):
        """
        Given a log-mel spectrogram of shape ``(b, melbins, t)``, performs
        forward pass through the NN stem, and then multi-residual-stage
        onset probability detection. Used in ``forward``.

        :returns: ``(x_stages, stem_out)``, where ``stem_out`` is a tensor of
          shape ``(b, stem_chans, keys, t-1)`` and ``x_stages`` is a list with
          one onset prediction per stage, each of shape ``(b, keys, t-1)``.
        """
        xdiff = x.diff(dim=-1)  # (b, melbins, t-1)
        # x+xdiff has shape (b, 2, melbins, t-1)
        x = torch.stack([x[:, :, 1:], xdiff]).permute(1, 0, 2, 3)
        x = self.specnorm(x)
        #
        stem_out = self.stem(x)  # (b, stem_ch, keys, t-1)
        x = self.first_stage(stem_out)  # (b, 1, keys, t-1)
        #
        x_stages = [x]
        for ref in self.refiner_stages:
            x = ref(stem_out) + x_stages[-1]  # residual instead of concat
            x_stages.append(x)
        for st in x_stages:
            st.squeeze_(1)
        #
        return x_stages, stem_out

    def forward(self, x, trainable_onsets=True):
        """
        :param x: Logmel batch of shape ``(b, melbins, t)``
        :returns: ``(x_stages, velocities)``. See ``forward_onsets`` for
          a description of ``x_stages``. The ``velocities`` tensor has shape
          ``(b, 1, keys, t-1)``, and is the result of processing ``stem_out``
          concatenated with the last ``x_stage`` output.
        """
        if trainable_onsets:
            x_stages, stem_out = self.forward_onsets(x)
            stem_out = torch.cat([stem_out, x_stages[-1].unsqueeze(1)], dim=1)
        else:
            with torch.no_grad():
                x_stages, stem_out = self.forward_onsets(x)
                stem_out = torch.cat([stem_out, x_stages[-1].unsqueeze(1)],
                                     dim=1)
        #
        velocities = self.velocity_stage(stem_out).squeeze(1)
        #
        return x_stages, velocities
