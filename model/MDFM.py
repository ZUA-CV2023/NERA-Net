import torch
import torch.nn as nn
import torch.nn.functional as F


class MDFM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(MDFM, self).__init__()
        # self.features = []
        # for bins in bins:  # 遍历缩放因子
        #     self.features.append(
        #         nn.Sequential(
        #             nn.AdaptiveAvgPool2d(bins),
        #             nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
        #             nn.BatchNorm2d(reduction_dim),
        #             nn.ReLU(inplace=True)
        #         ))
        # self.features = nn.ModuleList(self.features)  # 将 Python 列表转换为 PyTorch 的模块列表
        self.bins = bins
        self.features = nn.ModuleList()

        # 主金字塔分支
        for bin0 in bins:
            self.features.append(self._build_level(in_dim, reduction_dim, bin0))

        # 子金字塔分支
        self.sub_pyramids = nn.ModuleList()
        for bin1 in [1, 2, 3, 4]:
            self.sub_pyramids.append(self._build_level(reduction_dim, int(reduction_dim / 4), bin1))

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim * 3, in_dim * 2, 1),
            nn.BatchNorm2d(in_dim * 2),
            nn.ReLU(inplace=True)
        )

    def _build_level(self, in_dim, out_dim, bin_size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(bin_size),
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x_size = x.size()  # 获取图像的形状
        # out = [x]
        # for f in self.features:  # 获取各缩放因子尺度下的特征图
        #     out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        x_size = x.size()[2:]
        out = [x]

        # 处理主金字塔分支
        for i, f in enumerate(self.features):
            feat = f(x)
            # 上采样并收集特征
            up_feat = F.interpolate(feat, x_size, mode='bilinear', align_corners=True)
            out.append(up_feat)

            # 递归处理子金字塔
            for j, f1 in enumerate(self.sub_pyramids):
                sub_feat = f1(up_feat)
                sub_feat = F.interpolate(sub_feat, x_size, mode='bilinear', align_corners=True)
                out.append(sub_feat)
        # return torch.cat(out, 1)  # 拼接后返回
        return self.fusion(torch.cat(out, dim=1))
