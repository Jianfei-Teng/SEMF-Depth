from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from .hr_layers import *


class DepthDecoder(nn.Module):
    def __init__(self, ch_enc=[64, 128, 216, 288, 288], scales=range(4), num_ch_enc=[64, 64, 128, 256, 512],
                 num_output_channels=1):
        super(DepthDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.ch_enc = ch_enc
        self.scales = scales
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = nn.ModuleDict()

        # [Refactored for Dynamic Scale Support]
        # MPViT has 4 scales: [64, 128, 216, 288] (Indices 0, 1, 2, 3)
        # ResNet/Original has 5 scales: [64, 64, 128, 256, 512] (Indices 0, 1, 2, 3, 4)
        self.num_scales = len(ch_enc)

        # Feature Fusion Modules ("fX")
        # We process from the deepest scale up to the second scale (index 1).
        # For 5 scales (0..4), we have f4, f3, f2, f1.
        # For 4 scales (0..3), we have f3, f2, f1.
        for i in range(self.num_scales - 1, 0, -1):
            self.convs[f"f{i}"] = Attention_Module(self.ch_enc[i], num_ch_enc[i])

        # Dense Connection Tables (Triangle Shape)
        if self.num_scales == 5:
            # Original Depth 5 Triangle
            # j=1: 01, 11, 21, 31
            # j=2: 02, 12, 22
            # j=3: 03, 13
            # j=4: 04
            self.all_position = ["01", "11", "21", "31", "02", "12", "22", "03", "13", "04"]
            self.attention_position = ["31", "22", "13", "04"]  # Diagonal i+j=4
            self.non_attention_position = ["01", "11", "21", "02", "12", "03"]
        elif self.num_scales == 4:
            # Reduced Depth 4 Triangle
            # j=1: 01, 11, 21
            # j=2: 02, 12
            # j=3: 03
            self.all_position = ["01", "11", "21", "02", "12", "03"]
            self.attention_position = ["21", "12", "03"]  # Diagonal i+j=3
            self.non_attention_position = ["01", "11", "02"]
        else:
            raise ValueError(f"Unsupported number of encoder scales: {self.num_scales}. Expected 4 or 5.")

        # Construct Dense Block Convolutions
        for j in range(self.num_scales):
            for i in range(self.num_scales - j):
                # upconv 0
                if j == 0:
                    num_ch_in = num_ch_enc[i]
                else:
                    num_ch_in = self.num_ch_dec[i + 1]
                num_ch_out = num_ch_in // 2
                self.convs["X_{}{}_Conv_0".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

                # Last Column upconv 1 (Output Column -> 1/2 Resolution)
                if i == 0 and j == (self.num_scales - 1):
                    num_ch_in = num_ch_out
                    num_ch_out = self.num_ch_dec[i]
                    self.convs["X_{}{}_Conv_1".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

        # Declare fSEModule and original module
        for index in self.attention_position:
            row = int(index[0])
            col = int(index[1])
            # The output of fSEModule feeds directly into the next X_{row}{col-1}_Conv_0
            # The output of fSEModule feeds directly into the next X_{row}{col-1}_Conv_0
            expected_out_ch = self.num_ch_dec[row + 1]

            if col == 1:
                high_ch = num_ch_enc[row + 1] // 2
            else:
                high_ch = self.num_ch_dec[row + 2] // 2

            self.convs["X_" + index + "_attention"] = fSEModule(
                high_ch,
                self.num_ch_enc[row] + self.num_ch_dec[row + 1] * (col - 1),
                output_channel=expected_out_ch)

        for index in self.non_attention_position:
            row = int(index[0])
            col = int(index[1])
            if col == 1:
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(
                    num_ch_enc[row + 1] // 2 + self.num_ch_enc[row],
                    self.num_ch_dec[row + 1])
            else:
                self.convs["X_" + index + "_downsample"] = Conv1x1(
                    self.num_ch_dec[row + 2] // 2 + self.num_ch_enc[row] + self.num_ch_dec[row + 1] * (col - 1),
                    self.num_ch_dec[row + 1] * 2)
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(
                    self.num_ch_dec[row + 1] * 2,
                    self.num_ch_dec[row + 1])

        # Disparity Output Convolutions
        # We maintain 4 scales of disparity output if possible, or reduce to self.num_scales
        # Original code hardcoded range(4).
        for i in range(4):
            self.convs["dispconv{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)

        # Finalization: Ensure HRDecoder is correctly indexed for ModuleList
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def nestConv(self, conv, high_feature, low_features):
        conv_0 = conv[0]
        conv_1 = conv[1]
        assert isinstance(low_features, list)
        high_features = [upsample(conv_0(high_feature))]
        for feature in low_features:
            high_features.append(feature)
        high_features = torch.cat(high_features, 1)
        if len(conv) == 3:
            high_features = conv[2](high_features)
        return conv_1(high_features)

    def forward(self, input_features):
        outputs = {}
        feat = {}

        # Adaptive Feature Extraction
        # For 5 scales: feat[4], feat[3], feat[2], feat[1], feat[0]
        # For 4 scales: feat[3], feat[2], feat[1], feat[0]
        # Note: feat[0] is always input_features[0]
        for i in range(self.num_scales - 1, 0, -1):
            feat[i] = self.convs[f"f{i}"](input_features[i])
        feat[0] = input_features[0]

        features = {}
        for i in range(self.num_scales):
            features["X_{}0".format(i)] = feat[i]

        # Network architecture (Dense Connections)
        for index in self.all_position:
            row = int(index[0])
            col = int(index[1])

            low_features = []
            for i in range(col):
                low_features.append(features["X_{}{}".format(row, i)])

            # add fSE block to decoder
            if index in self.attention_position:
                features["X_" + index] = self.convs["X_" + index + "_attention"](
                    self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)](features["X_{}{}".format(row + 1, col - 1)]),
                    low_features)
            elif index in self.non_attention_position:
                conv = [self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)],
                        self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)]]
                if col != 1:
                    conv.append(self.convs["X_" + index + "_downsample"])
                features["X_" + index] = self.nestConv(conv, features["X_{}{}".format(row + 1, col - 1)], low_features)

        # Output Extraction
        # Top-Right Node: X_0{num_scales-1} (e.g., X_04 or X_03)
        top_node_idx = self.num_scales - 1
        x = features[f"X_0{top_node_idx}"]
        x = self.convs[f"X_0{top_node_idx}_Conv_0"](x)

        # [V23 MULTI-RESOLUTION GUIDANCE] Expose intermediate features at 1/2 and 1/1 resolutions
        # 1. Provide features at 1/2 resolution (same as X_0{top} node)
        outputs[("F_dec", 1)] = x  # 1/2 resolution for auxiliary heads (e.g., EdgeHead)

        # 2. Final upsample to reach 1/1 (Native) resolution
        # We use Conv_1 (already defined in nodes) to generate the final full-res feature map
        x_full = self.convs[f"X_0{top_node_idx}_Conv_1"](upsample(x))  # 1/1 resolution

        # Expose F_dec: the TRUE full-resolution decoded feature (matching input image size)
        outputs[("F_dec", 0)] = x_full  # 16-channel 1/1 resolution feature

        # Disparity Outputs (Multiscale)
        # disp0 is full res (from x_full)
        outputs[("disp", 0)] = self.sigmoid(self.convs["dispconv0"](x_full))

        # disp1, disp2, disp3 from lower resolutions
        # 5-scale: disp1 (X_04), disp2 (X_13), disp3 (X_22)
        # 4-scale: disp1 (X_03), disp2 (X_12), disp3 (X_21) (Adjust indices by -1 relative to loop)

        # Helper lambda to safe get feature
        def get_feat(r, c):
            return features.get(f"X_{r}{c}", None)

        # disp1: From the node just before the final upsample block (X_0{top})
        if get_feat(0, top_node_idx) is not None:
            outputs[("disp", 1)] = self.sigmoid(self.convs["dispconv1"](features[f"X_0{top_node_idx}"]))

        # disp2: From X_1{top-1}
        if get_feat(1, top_node_idx - 1) is not None:
            outputs[("disp", 2)] = self.sigmoid(self.convs["dispconv2"](features[f"X_1{top_node_idx - 1}"]))

        # disp3: From X_2{top-2}
        if get_feat(2, top_node_idx - 2) is not None:
            outputs[("disp", 3)] = self.sigmoid(self.convs["dispconv3"](features[f"X_2{top_node_idx - 2}"]))

        return outputs
