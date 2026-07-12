from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from .hr_decoder import DepthDecoder
from .mpvit import mpvit_small


class DeepNet(nn.Module):
    def __init__(self, type, weights_init="pretrained", num_layers=18, num_pose_frames=2, scales=range(4)):
        super(DeepNet, self).__init__()
        self.type = type
        self.num_layers = num_layers
        self.weights_init = weights_init
        self.num_pose_frames = num_pose_frames
        self.scales = scales

        if self.type == 'mpvitnet':
            self.encoder = mpvit_small()
            self.encoder.num_ch_enc = [64, 128, 216, 288, 288]
            self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, ch_enc=self.encoder.num_ch_enc)
        else:
            print("wrong type of the networks, only depthnet and posenet")

    def forward(self, inputs):
        self.outputs = self.decoder(self.encoder(inputs))
        return self.outputs
