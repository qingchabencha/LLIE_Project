import torch
import torch.nn as nn
from model import retinex_extractor_utils


class RetinexExtractor(nn.Module):
    def __init__(self, n_resblocks = 16, n_feats = 64, rgb_range = 1.0, scale = [4], conv=retinex_extractor_utils.default_conv):
        super().__init__()

        #print("initialized a retinex extractor")
        kernel_size = 3
        self.scale_idx = 0

        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = retinex_extractor_utils.MeanShift(rgb_range, rgb_mean, rgb_std)

        m_head = [conv(3, n_feats, kernel_size)]

        self.pre_process = nn.ModuleList([
            nn.Sequential(
                retinex_extractor_utils.ResBlock(conv, n_feats, 5, act=act),
                retinex_extractor_utils.ResBlock(conv, n_feats, 5, act=act)
            ) for _ in scale
        ])

        m_body = [
            retinex_extractor_utils.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.upsample = nn.ModuleList([
            retinex_extractor_utils.Upsampler(
                conv, s, n_feats, act=False
            ) for s in scale
        ])

        m_tail = [conv(n_feats, 3, kernel_size)]

        self.add_mean = retinex_extractor_utils.MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        x = self.pre_process[self.scale_idx](x)

        res = self.body(x)
        res += x

        x = self.upsample[self.scale_idx](res)
        x = self.tail(x)
        x = self.add_mean(x)
        #print("all positive")
        x = nn.ReLU()(x)

        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx