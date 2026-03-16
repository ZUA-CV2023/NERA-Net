reduce_dim = 256
self.bg_loss = nn.CrossEntropyLoss(reduction='none')  # 损失函数
# self.avgpool_list = [60, 30, 15, 8]
# 初始化一个可学习的背景原型（prototype），用于对比学习或背景建模
self.bg_prototype = nn.Parameter(torch.zeros(1, reduce_dim, 1, 1))

self.down_bg = nn.Sequential(
    nn.Conv2d(reduce_dim * 3, reduce_dim, kernel_size=1, padding=0, bias=False),
    nn.ReLU(inplace=True),
    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
    nn.ReLU(inplace=True),
)

# 对背景特征进行进一步提取，数据增强
self.bg_res1 = nn.Sequential(
    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
    nn.ReLU(inplace=True),
    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
    nn.ReLU(inplace=True),
)

self.bg_cls = nn.Sequential(
    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
    nn.ReLU(inplace=True),
    nn.Dropout2d(p=0.1),
    nn.Conv2d(reduce_dim, 2, kernel_size=1)
)

# self.feat_out = nn.Sequential(
#     nn.Conv2d(reduce_dim * 4, reduce_dim, kernel_size=1, padding=0, bias=False),
#     nn.ReLU(inplace=True),
#     nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
#     nn.ReLU(inplace=True),
# )


def multi_scale_optimization(img_init):
    """
    对输入的 img_cam 进行多尺度处理，并返回优化后的图像。

    参数:
    img_cam (Tensor): 输入张量，形状为 (1, 2, 60, 60)

    返回:
    Tensor: 优化后的张量，形状为 (1, 2, 60, 60)
    """
    # 进行下采样到不同尺度
    img_init_15 = F.interpolate(img_init, size=(15, 15), mode='bilinear', align_corners=False)
    img_init_30 = F.interpolate(img_init, size=(30, 30), mode='bilinear', align_corners=False)
    img_init_120 = F.interpolate(img_init, size=(120, 120), mode='bilinear', align_corners=False)

    # 将所有图像上采样回原始尺寸 (60, 60)
    img_init_15_up = F.interpolate(img_init_15, size=(60, 60), mode='bilinear', align_corners=False)
    img_init_30_up = F.interpolate(img_init_30, size=(60, 60), mode='bilinear', align_corners=False)
    img_init_120_up = F.interpolate(img_init_120, size=(60, 60), mode='bilinear', align_corners=False)

    # 融合不同尺度的图像 (简单加权平均)
    img_fused = (img_init + img_init_15_up + img_init_30_up + img_init_120_up) / 4

    # 注意：这里 img_fused 已经是 (1, 2, 60, 60)，所以不需要再次上采样
    img_optimized = img_fused

    return img_optimized


def process_multi_scale(x, in_channels, out_channels):
    """
    对输入特征进行多尺度处理，并返回与原始输入大小一致的输出特征。

    Args:
        x (torch.Tensor): 输入特征图，形状为 (B, C, H, W)。
        in_channels (int): 输入特征图的通道数。
        out_channels (int): 输出特征图的通道数。

    Returns:
        torch.Tensor: 经过多尺度处理后的特征图，形状为 (B, out_channels, H, W)。
    """
    # # 定义卷积层
    # conv = nn.Sequential(
    #     nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
    #     nn.ReLU(inplace=True),
    # )

    # 原始尺度处理
    x_60 = x

    # 多尺度变换
    x_15 = F.interpolate(x, size=(15, 15), mode='bilinear', align_corners=False)
    # x_15 = conv(x_15)
    x_15 = F.interpolate(x_15, size=(60, 60), mode='bilinear', align_corners=False)

    x_30 = F.interpolate(x, size=(30, 30), mode='bilinear', align_corners=False)
    # x_30 = conv(x_30)
    x_30 = F.interpolate(x_30, size=(60, 60), mode='bilinear', align_corners=False)

    x_120 = F.interpolate(x, size=(120, 120), mode='bilinear', align_corners=False)
    # x_120 = conv(x_120)
    x_120 = F.interpolate(x_120, size=(60, 60), mode='bilinear', align_corners=False)

    x_15_up = F.interpolate(x_15, size=(60, 60), mode='bilinear', align_corners=False)
    x_30_up = F.interpolate(x_30, size=(60, 60), mode='bilinear', align_corners=False)
    x_120_up = F.interpolate(x_120, size=(60, 60), mode='bilinear', align_corners=False)
    # 合并多尺度特征
    x_combined = x_60 + x_15_up + x_30_up + x_120_up
    return x_combined


def process_features(query_feat_cnn, supp_feat_bin, img_cam, clip_similarity):
    """
    对多组输入特征分别进行多尺度处理，并返回处理后的结果。

    Args:
        query_feat_cnn (torch.Tensor): 主干网络提取的特征，形状为 (B, 256, 60, 60)。
        supp_feat_bin (torch.Tensor): 支持特征，形状为 (B, 256, 60, 60)。
        img_cam (torch.Tensor): 图像相关特征，形状为 (B, 2, 60, 60)。
        clip_similarity (torch.Tensor): CLIP 相似度特征，形状为 (B, 2, 60, 60)。

    Returns:
        tuple: 处理后的特征图，分别为：
            - query_feat_cnn_processed: (B, 256, 60, 60)
            - supp_feat_bin_processed: (B, 256, 60, 60)
            - img_cam_processed: (B, 2, 60, 60)
            - clip_similarity_processed: (B, 2, 60, 60)
    """
    query_feat_cnn_processed = process_multi_scale(query_feat_cnn, in_channels=256, out_channels=256)
    supp_feat_bin_processed = process_multi_scale(supp_feat_bin, in_channels=256, out_channels=256)
    img_cam_processed = process_multi_scale(img_cam, in_channels=2, out_channels=2)
    clip_similarity_processed = process_multi_scale(clip_similarity, in_channels=2, out_channels=2)

    return query_feat_cnn_processed, supp_feat_bin_processed, img_cam_processed, clip_similarity_processed





# --------------------------------------- BGMM背景挖掘模块 ------------------------------------------ ###
        # PBG 和 PBG‘  调用初始化中的构建BG原型，并扩展到对应尺寸     [8, 512, 60, 60]
        bg = self.bg_prototype.expand(query_feat_cnn.size(0), -1, query_feat_cnn.size(2), query_feat_cnn.size(3))

        # PBG' + Xq
        qrybg_feat = torch.cat((query_feat_cnn, bg), dim=1)     # [8, 768, 60, 60]

        # 调用初始化中定义的BG降维模块进行特征融合
        qrybg_feat1 = self.down_bg(qrybg_feat)  # [8, 256, 60, 60]

        # XqBG 调用初始化中对BG进行残差的两个3*3卷积进行背景挖掘并残差连接
        query_feat_cnn = self.bg_res1(qrybg_feat1) + qrybg_feat1  # [8, 256, 60, 60]

        query_bg_out = self.bg_cls(query_feat_cnn)  # [8, 2, 60, 60]

        #
        # query_bg_out_list = []
        # query_bg_feat_list = []
        # # 实现多尺度特征金字塔，通过不同尺度的特征融合、残差连接和混淆区域处理逐步优化分割结果
        # for idx, tmp_bin in enumerate(self.avgpool_list):
        #     # 如果输入的金字塔尺度为比例而不是尺寸
        #     if tmp_bin <= 1.0:
        #         bins = int(query_feat_cnn.shape[2] * tmp_bin)  # 获取和比例对应的尺寸
        #         query_feat_bin = nn.AdaptiveAvgPool2d(bins)(query_feat_cnn)  # 直接调用自适应平均池化   [8, 256, bins, bins]
        #     else:
        #         bins = tmp_bin
        #         # 调用初始化中的全局平均池化对查询特征进行特征提取
        #         query_feat_bin = nn.AdaptiveAvgPool2d(bins)(query_feat_cnn)  # [8, 256, bins, bins]
        #
        #     # PBG 和 PBG‘  调用初始化中的构建BG原型，并扩展到对应尺寸     [8, 512, bins, bins]
        #     bg = self.bg_prototype.expand(query_feat_cnn.size(0), -1, bins, bins)
        #
        #     # PBG' + Xq
        #     qrybg_feat = torch.cat((query_feat_bin, bg), dim=1)  # [8, 768, bins, bins]
        #
        #     # 调用初始化中定义的BG降维模块进行特征融合
        #     qrybg_feat1 = self.down_bg(qrybg_feat)  # [8, 256, bins, bins]
        #
        #     # XqBG 调用初始化中对BG进行残差的两个3*3卷积进行背景挖掘并残差连接
        #     qrybg_feat2 = self.bg_res1(qrybg_feat1) + qrybg_feat1  # [8, 256, bins, bins]
        #     query_feat = F.interpolate(qrybg_feat2, size=(query_feat_cnn.size(2), query_feat_cnn.size(3)),
        #                                mode='bilinear', align_corners=True)
        #     query_bg_out = self.bg_cls(qrybg_feat2)  # [8, 2, bins, bins]
        #     query_bg_out = F.interpolate(query_bg_out, size=(query_feat_cnn.size(2), query_feat_cnn.size(3)),
        #                                  mode='bilinear', align_corners=True)
        #
        #     query_bg_out_list.append(query_bg_out)
        #     query_bg_feat_list.append(query_feat)
        #
        # query_bg_out = (query_bg_out_list[0] + query_bg_out_list[1] + query_bg_out_list[2] +
        #                 query_bg_out_list[3]) / len(query_bg_out_list)
        #
        # query_feat_cnn = torch.cat((query_bg_feat_list[0], query_bg_feat_list[1], query_bg_feat_list[2],
        #                             query_bg_feat_list[3]), dim=1)
        # query_feat_cnn = self.feat_out(query_feat_cnn)

        supp_bg_out_list = []  # 支持背景输出列表
        # 如果在训练中
        if self.training:
            # bg = self.bg_prototype.expand(query_feat_cnn.size(0), -1, query_feat_cnn.size(2), query_feat_cnn.size(3))
            # 遍历无掩码融合支持特征列表
            for supp_feat_nomask in supp_feat_list_ori:
                # Xs + PBG’ 对特征进行维度上的拼接
                suppbg_feat = torch.cat((supp_feat_nomask, bg), dim=1)  # [8, 768, 60, 60]
                # 调用初始化中定义的降维模块进行特征融合
                suppbg_feat = self.down_bg(suppbg_feat)  # [8, 256, 60, 60]
                # 调用初始化中对BG进行残差的两个3*3卷积进行背景挖掘并残差连接
                suppbg_feat = self.bg_res1(suppbg_feat) + suppbg_feat  # [8, 256, 60, 60]
                # ysBG  调用初始化中的背景分类块对特征处理并得到消除背景的支持图像预测
                supp_bg_out = self.bg_cls(suppbg_feat)  # [8, 2, 60, 60]
                supp_bg_out_list.append(supp_bg_out)  # 将支持背景输出加入支持背景输出列表中




# 将消除背景的查询图像恢复到原始输入图像的尺寸    [8, 2, 473, 473]
query_bg_out = F.interpolate(query_bg_out, size=(h, w), mode='bilinear', align_corners=True)

mygt0 = torch.zeros(query_bg_out.size(0), h, w).cuda()  # 初始化非背景标签
mygt1 = torch.ones(query_bg_out.size(0), h, w).cuda()  # 初始化背景标签
query_bg_loss = self.weighted_BCE(query_bg_out, mygt0, y_m) + 0.5 * self.criterion(query_bg_out,  # 查询图像背景损失
                                                                                   mygt1.long())
# 支持图像背景损失
# 遍历消除背景的支持预测列表
for j, supp_bg_out in enumerate(supp_bg_out_list):
    # 将消除背景的支持预测结果恢复到原始输入图像的尺寸  [8, 2, 473, 473]
    supp_bg_out = F.interpolate(supp_bg_out, size=(h, w), mode='bilinear', align_corners=True)
    supp_bg_loss = self.weighted_BCE(supp_bg_out, mygt0, s_y[:, j, :, :]) + 0.5 * self.criterion(
        supp_bg_out, mygt1.long())  # 查询图像背景损失
    aux_bg_loss = aux_bg_loss + supp_bg_loss  # 支持的损失叠加

bg_loss = (query_bg_loss + aux_bg_loss) / (len(supp_bg_out_list) + 1)  # 平均背景损失





def weighted_BCE(self, image, target, mask):
    loss_list = []
    # 若mask的某个位置为 1，则cmask在该位置取 1。否则，cmask 取 target 的值（即真实标签）。
    cmask = torch.where(mask.long() == 1, mask.long(), target.long())

    # 遍历每个样本
    for x, y, z in zip(image, target, cmask):
        loss = self.bg_loss(x.unsqueeze(0), y.unsqueeze(0).long())  # 计算单个样本的损失     [1, 473, 473]
        area = torch.sum(z) + 1e-5  # 计算有效区域面积
        Loss = torch.sum(z.unsqueeze(0) * loss) / area  # 加权损失计算
        loss_list.append(Loss.unsqueeze(0))  # 将损失添加到损失列表
    LOSS = torch.cat(loss_list, dim=0)  # 将每个样本的损失拼接成张量

    # 计算批次平均损失并返回
    return torch.mean(LOSS)





# 背景原型相关参数
{'params': self.bg_prototype},  # 背景原型参数
{'params': self.down_bg.parameters()},  # 背景特征降维层的参数
{'params': self.bg_res1.parameters()},  # 背景特征提取层的参数
{'params': self.bg_cls.parameters()},  # 背景分类层的参数
# {'params': self.feat_out.parameters(), "lr": LR * 10},