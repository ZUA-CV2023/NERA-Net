import cv2
import numpy as np
import argparse
import os.path as osp
from tqdm import tqdm
# from .util import get_train_val_set, check_makedirs
from util import get_train_val_set, check_makedirs

# Get the annotations of base categories

# root_path
# ├── BAM/
# │   ├── util/
# │   ├── config/
# │   ├── model/
# │   ├── README.md
# │   ├── train.py
# │   ├── train_base.py
# │   └── test.py
# └── data/
#     ├── base_annotation/   # the scripts to create THIS folder
#     │   ├── pascal/
#     │   │   ├── train/   
#     │   │   │   ├── 0/     # annotations of PASCAL-5^0
#     │   │   │   ├── 1/
#     │   │   │   ├── 2/
#     │   │   │   └── 3/
#     │   │   └── val/      
#     │   └── coco/          # the same file structure for COCO
#     ├── VOCdevkit2012/
#     └── MSCOCO2014/

parser = argparse.ArgumentParser()
args = parser.parse_args()

args.data_set = 'coco'    # 选择要处理的数据集     pascal coco
args.use_split_coco = False      # 是否使用COCO的分割方法

num_classes = 0

for sp in [0, 1, 2, 3]:     # 遍历所有分割方式
    for mm in ['train', 'val']:     # 遍历训练和验证
        args.mode = mm      # 获取当前处理集   train val
        args.split = sp     # 获取当前处理块   0 1 2 3
        if args.data_set == 'pascal':       # 根据当前集获取总类别数
            num_classes = 20
        elif args.data_set == 'coco':
            num_classes = 80

        # -------------------------- 路径配置 ----------------------------------
        # root_path = '/mnt/home/bhpeng22/githubProjects/fewshot_segmentation/data'
        root_path = 'E:/FSS/PI_CLIP_me/'     # 绝对路径
        data_path = osp.join(root_path, 'data/base_annotation/')    # 相对路径
        save_path = osp.join(data_path, args.data_set, args.mode, str(args.split))      # 拼接得到保存路径
        check_makedirs(save_path)

        # -------------------------- 数据列表获取 ----------------------------------
        sub_list, sub_val_list = get_train_val_set(args)

        # get data_list
        # fss_list_root = '../lists/{}/fss_list/{}'.format(args.data_set, args.mode)
        fss_list_root = '../lists/{}/fss_list/{}/'.format(args.data_set, args.mode)     # 文件存放路径
        fss_data_list_path = fss_list_root + 'data_list_{}.txt'.format(args.split)      # 文件路径
        with open(fss_data_list_path, 'r') as f:        # 对文件逐行读取
            f_str = f.readlines()
        data_list = []
        for line in f_str:
            img, mask = line.split(' ')     # 按空格分割图像和标注路径
            data_list.append((img, mask.strip()))       # 移除mask路径的换行符

        # -------------------------- 标注重映射处理 ----------------------------------
        for index in tqdm(range(len(data_list))):       # 路径处理
            image_path, label_path = data_list[index]       # 读取图像和标签路径
            # image_path, label_path = root_path + image_path[3:], root_path+ label_path[3:] 
            # print(">>>>>>>>>>>>>>>>>>>>>>>")
            # print(image_path)
            # print(label_path)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)    # 读取标注（单通道灰度图）
            label_tmp = label.copy()        # 创建副本用于查询
            # 类别重映射
            for cls in range(1, num_classes + 1):       # 遍历所有可能类别（跳过背景类0）
                select_pix = np.where(label_tmp == cls)     # 找到当前类别像素坐标
                if cls in sub_list:     # 是目标类别
                    label[select_pix[0], select_pix[1]] = sub_list.index(cls) + 1   # 重新编号(从1开始)  无论原本是第几类，都要从1开始重编为15类
                else:
                    label[select_pix[0], select_pix[1]] = 0     # 设为背景

            # for pix in np.nditer(label, op_flags=['readwrite']):
            #     if pix == 255:
            #         pass
            #     elif pix not in sub_list: 
            #         pix[...] = 0
            #     else:
            #         pix[...] = sub_list.index(pix) + 1

            save_item_path = osp.join(save_path, label_path.split('/')[-1])     # 获取文件保存路径
            cv2.imwrite(save_item_path, label)      # 保存文件

        print('end')
