import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


# ASPP模块，用于多尺度特征提取
class GEASPP(nn.Module):
    def __init__(self, out_channels=256):
        super(GEASPP, self).__init__()

        dim = out_channels  # 保留原始通道数

        self.layer6_0 = nn.Sequential(      # 1x1卷积分支，用于全局特征融合
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            GradientModulationLayer(),  # 梯度调制增强
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=False),
            GradientModulationLayer(),  # 梯度调制增强
        )

        self.layer6_1 = nn.Sequential(      # 1x1卷积分支，原始分辨率特征提取
            GradientModulationLayer(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=False),
            GradientModulationLayer()
            )

        self.layer6_2 = nn.Sequential(      # 3x3卷积分支，空洞卷积在不增加参数量的前提下扩大感受野
            GradientModulationLayer(),  # 梯度调制增强
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, bias=True),
            nn.ReLU(inplace=False),
            GradientModulationLayer(),  # 梯度调制增强
            )

        self.layer6_3 = nn.Sequential(      # 3x3卷积分支，空洞卷积在不增加参数量的前提下扩大感受野
            GradientModulationLayer(),  # 梯度调制增强
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=True),
            nn.ReLU(inplace=False),
            GradientModulationLayer(),  # 梯度调制增强
            )

        self.layer6_4 = nn.Sequential(      # 3x3卷积分支，空洞卷积在不增加参数量的前提下扩大感受野
            GradientModulationLayer(),  # 梯度调制增强
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18, bias=True),
            nn.ReLU(inplace=False),
            GradientModulationLayer(),  # 梯度调制增强
            )

        # 6. 梯度引导的特征融合层
        # 取代简单的拼接操作
        self.gradient_fusion = GradientFusionBlock(out_channels, 5)  # 5个分支输入

        self._init_weight()     # 权重初始化

    # 实现了神经网络参数的初始化方法，专门用于初始化ASPP模块中的卷积层和批归一化层
    def _init_weight(self):     # 自定义权重初始化
        for m in self.modules():        # 遍历所有子模块
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)     # 何凯明正态分布初始化，适用于ReLU激活函数
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)      # 初始化为全1，保持原始输入分布
                m.bias.data.zero_()     # 初始化为全0，初始时不进行偏移
                
    def forward(self, x):
        feature_size = x.shape[-2:]     # 获取图像的尺寸大小

        global_feature = self.layer6_0(x)      # 全局特征融合    [4, 256, 1, 1]

        # 将全局特征扩展到原本尺寸
        global_feature = F.interpolate(global_feature, size=feature_size,  mode='bilinear', align_corners=True) + x

        x1 = self.layer6_1(x)
        x2 = self.layer6_2(x)
        x3 = self.layer6_3(x)
        x4 = self.layer6_4(x)

        all_features = [global_feature, x1, x2, x3, x4]     # 将所有分支特征和全局特征组合

        out = self.gradient_fusion(all_features, x)      # 6. 应用梯度引导的特征融合
        return out


# 梯度调制层：增强特征中的梯度信息
class GradientModulationLayer(nn.Module):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha  # 调制强度因子

    def forward(self, x):
        if self.training:
            # 训练时应用梯度增强
            return GradientModulationFunction.apply(x, self.alpha)
        # 推理时直接返回原始特征
        return x


# 梯度调制函数（自定义反向传播）
class GradientModulationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, alpha_value):
        """前向传播：避免所有原地操作"""
        # 关键：创建输入张量的副本，避免污染原始数据
        input_clone = input_tensor.clone()

        # 保存克隆后的张量（确保原始张量不受影响）
        ctx.save_for_backward(input_clone)

        # 保存alpha并确保是标量张量
        if not isinstance(alpha_value, torch.Tensor):
            ctx.alpha = torch.tensor(
                alpha_value,
                dtype=input_tensor.dtype,
                device=input_tensor.device
            )
        else:
            ctx.alpha = alpha_value.detach().clone()

        return input_tensor  # 返回原始输入（未修改）

    @staticmethod
    def backward(ctx, *grad_outputs):
        """反向传播：修正所有潜在问题点"""
        # 1. 安全获取保存的张量（使用正确的属性名）
        input_clone, = ctx.saved_tensors
        alpha = ctx.alpha.to(input_clone.device)

        # 2. 确保不使用原地操作
        with torch.set_grad_enabled(True):
            # 安全设置梯度需求（创建新张量）
            input_with_grad = input_clone.detach().clone().requires_grad_(True)

            # 3. 计算梯度幅值（避免任何原地操作）
            grad_map = torch.autograd.grad(
                outputs=input_with_grad.sum(),  # 使用新张量
                inputs=input_with_grad,
                create_graph=False,  # 暂时禁用高阶梯度
                retain_graph=False
            )[0]

        # 4. 确保所有操作都是非原地的
        energy = grad_map.abs().mean(dim=1, keepdim=True).detach()

        # 5. 获取当前梯度输出
        grad_output = grad_outputs[0]

        # 6. 应用安全的梯度调制（避免类型错误）
        modulation_factor = 1 + alpha * energy
        modulated_grad = grad_output * modulation_factor

        return modulated_grad, None  # 返回两个值（匹配输入）


# 梯度引导的特征融合块
class GradientFusionBlock(nn.Module):
    def __init__(self, channels, num_branches):     # 64 5
        super().__init__()
        # 可学习的权重参数
        self.gradient_weights = nn.Parameter(torch.ones(num_branches, 1, 1, 1))
        # 特征融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(num_branches * channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, branch_outputs, origin_feat):
        if self.training:
            # 训练时 - 动态计算梯度重要性
            gradients = []
            with torch.enable_grad():
                # 开启原始特征的梯度计算
                origin_feat.requires_grad_(True)

                # 计算每个分支特征的梯度重要性
                for feat in branch_outputs:
                    # 计算特征对原始输入的梯度
                    grad = torch.autograd.grad(
                        outputs=feat,
                        inputs=origin_feat,
                        grad_outputs=torch.ones_like(feat),
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    # 平均梯度幅值作为重要性指标
                    gradients.append(grad.abs().mean(dim=[1, 2, 3]))

            # 计算各分支的权重（基于梯度重要性）
            gradient_importance = torch.stack(gradients, dim=1).softmax(dim=1)
        else:
            # 推理时 - 使用训练好的固定权重
            gradient_importance = self.gradient_weights

        # 应用权重融合各分支特征
        weighted_outputs = []
        for i in range(len(branch_outputs)):
            # 使用取模运算确保索引安全
            idx = i % gradient_importance.size(1)
            weight = gradient_importance[:, idx].view(-1, 1, 1, 1)
            if not self.training:
                weight = weight.mean(dim=0, keepdim=True)
            weighted_outputs.append(weight * branch_outputs[i])

        # 拼接加权后的特征
        concatenated = torch.cat(weighted_outputs, dim=1)

        # 通过融合卷积产生最终输出
        return self.fusion_conv(concatenated)
