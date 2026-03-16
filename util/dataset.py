import os
import os.path
import cv2
import numpy as np
import copy

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random
import time
from tqdm import tqdm
from PIL import Image

from .get_weak_anns import transform_anns

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split=0, data_root=None, data_list=None, sub_list=None, filter_intersection=False):
    assert split in [0, 1, 2, 3]
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    # Shaban uses these lines to remove small objects:
    # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
    #    filtered_item.append(item)
    # which means the mask will be downsampled to 1/32 of the original size and the valid area should be larger than 2,
    # therefore the area in original size should be accordingly larger than 2 * 32 * 32
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Processing data...".format(sub_list))
    sub_class_file_list = {}
    for sub_c in sub_list:
        sub_class_file_list[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')
        image_name = os.path.join(data_root, line_split[0])
        label_name = os.path.join(data_root, line_split[1])
        item = (image_name, label_name)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        label_class = np.unique(label).tolist()

        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)

        new_label_class = []

        if filter_intersection:  # filter images containing objects of novel categories during meta-training
            if set(label_class).issubset(set(sub_list)):
                for c in label_class:
                    if c in sub_list:
                        tmp_label = np.zeros_like(label)
                        target_pix = np.where(label == c)
                        tmp_label[target_pix[0], target_pix[1]] = 1
                        if tmp_label.sum() >= 2 * 32 * 32:
                            new_label_class.append(c)
        else:
            for c in label_class:
                if c in sub_list:
                    tmp_label = np.zeros_like(label)
                    target_pix = np.where(label == c)
                    tmp_label[target_pix[0], target_pix[1]] = 1
                    if tmp_label.sum() >= 2 * 32 * 32:
                        new_label_class.append(c)

        label_class = new_label_class

        if len(label_class) > 0:
            image_label_list.append(item)
            for c in label_class:
                if c in sub_list:
                    sub_class_file_list[c].append(item)

    print("Checking image&label pair {} list done! ".format(split))
    return image_label_list, sub_class_file_list


class SemData(Dataset):
    def __init__(self, split=3, shot=1, data_root=None, base_data_root=None, data_list=None, data_set=None,
                 use_split_coco=False,
                 transform=None, transform_tri=None, mode='train', ann_type='mask',
                 ft_transform=None, ft_aug_size=None,
                 ms_transform=None):

        assert mode in ['train', 'val', 'demo', 'finetune']     # 检测处于什么模式
        assert data_set in ['pascal', 'coco']       # 检测数据集是否正确
        if mode == 'finetune':
            assert ft_transform is not None
            assert ft_aug_size is not None

        if data_set == 'pascal':        # 不同数据集的总类别数
            self.num_classes = 20
        elif data_set == 'coco':
            self.num_classes = 80

        self.mode = mode  # 获取当前模式
        self.split = split  # 获取分割方法
        self.shot = shot  # 获取支持样本量
        self.data_root = data_root  # 获取数据位置
        self.base_data_root = base_data_root  # 获取基类掩码数据地址
        self.ann_type = ann_type  # 获取数据类别

        if data_set == 'pascal':
            self.class_list = list(range(1, 21))    # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            if self.split == 3:
                self.sub_list = list(range(1, 16))      # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                self.sub_val_list = list(range(16, 21))     # [16,17,18,19,20]
            elif self.split == 2:
                self.sub_list = list(range(1, 11)) + list(range(16, 21))    # [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
                self.sub_val_list = list(range(11, 16))     # [11,12,13,14,15]
            elif self.split == 1:
                self.sub_list = list(range(1, 6)) + list(range(11, 21))     # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(6, 11))      # [6,7,8,9,10]
            elif self.split == 0:
                self.sub_list = list(range(6, 21))      # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(1, 6))   # [1,2,3,4,5]

        elif data_set == 'coco':        # coco数据集根据分割方式对类进行分割，并判断是否使用coco专门的分割方式
            if use_split_coco:
                print('INFO: using SPLIT COCO (FWB)')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_val_list = list(range(4, 81, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
                elif self.split == 2:
                    self.sub_val_list = list(range(3, 80, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
                elif self.split == 1:
                    self.sub_val_list = list(range(2, 79, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
                elif self.split == 0:
                    self.sub_val_list = list(range(1, 78, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
            else:
                print('INFO: using COCO (PANet)')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_list = list(range(1, 61))
                    self.sub_val_list = list(range(61, 81))
                elif self.split == 2:
                    self.sub_list = list(range(1, 41)) + list(range(61, 81))
                    self.sub_val_list = list(range(41, 61))
                elif self.split == 1:
                    self.sub_list = list(range(1, 21)) + list(range(41, 81))
                    self.sub_val_list = list(range(21, 41))
                elif self.split == 0:
                    self.sub_list = list(range(21, 81))
                    self.sub_val_list = list(range(1, 21))

        print('sub_list: ', self.sub_list)
        print('sub_val_list: ', self.sub_val_list)

        # @@@ For convenience, we skip the step of building datasets and instead use the pre-generated lists @@@ if
        # self.mode == 'train': self.data_list, self.sub_class_file_list = make_dataset(split, data_root, data_list,
        # self.sub_list, True) assert len(self.sub_class_file_list.keys()) == len(self.sub_list) elif self.mode ==
        # 'val' or self.mode == 'demo' or self.mode == 'finetune': self.data_list, self.sub_class_file_list =
        # make_dataset(split, data_root, data_list, self.sub_val_list, False) assert len(
        # self.sub_class_file_list.keys()) == len(self.sub_val_list)

        mode = 'train' if self.mode == 'train' else 'val'       # 根据当前的模式，调整为训练或者验证，用于拼接文件路径
        self.base_path = os.path.join(self.base_data_root, mode, str(self.split))       # 获取基类掩码路径

        fss_list_root = './lists/{}/fss_list/{}/'.format(data_set, mode)        # 获取记录数据的文件地址
        fss_data_list_path = fss_list_root + 'data_list_{}.txt'.format(split)       # 获取训练数据列表文件的详细地址
        # 获取分类数据列表文件的详细地址
        fss_sub_class_file_list_path = fss_list_root + 'sub_class_file_list_{}.txt'.format(split)

        # Write FSS Data
        # with open(fss_data_list_path, 'w') as f:
        #     for item in self.data_list:
        #         img, label = item
        #         f.write(img + ' ')
        #         f.write(label + '\n')
        # with open(fss_sub_class_file_list_path, 'w') as f:
        #     f.write(str(self.sub_class_file_list))

        # 读取图片和标签的数据
        with open(fss_data_list_path, 'r') as f:
            f_str = f.readlines()       # 逐行读取文件内容
        self.data_list = []
        for line in f_str:      # 遍历每行内容
            img, mask = line.split(' ')     # 按空格进行分割
            img, mask = str(img).replace('..', '.'), str(mask).replace('..', '.')  # 路径转换
            self.data_list.append((img, mask.strip()))      # 将数据地址保存在列表中

        with open(fss_sub_class_file_list_path, 'r') as f:
            f_str = f.read()
            f_str = str(f_str).replace('..', '.')  # 路径转换
        self.sub_class_file_list = eval(f_str)      # 将数据地址保存在列表中

        self.transform = transform  # 获取图像处理方法
        self.transform_tri = transform_tri
        self.ft_transform = ft_transform  # 微调阶段的数据增强流水线
        self.ft_aug_size = ft_aug_size  # 微调时的增强输出尺寸
        self.ms_transform_list = ms_transform  # 多尺度训练的增强策略列表

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]      # 获取图片和标签的路径
        tmp_sperate = image_path.split('/')     # 将文件路径字符串按照 / 分隔符进行拆分
        if 'VOC' in image_path:     # 获取查询数据的名字     '2008_006549.jpg'
            query_name = tmp_sperate[5]
        else:
            query_name = tmp_sperate[4]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)    # 读取BGR格式图像

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # 转换为RGB格式
        images = image
        image = np.float32(image)       # 转换为float32类型
        img_cv2 = image.copy()      # 获取查询图像的原始备份
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)        # 读取原始灰度标注
        label_b = cv2.imread(os.path.join(str(self.base_path).replace('\\', '/'), label_path.split('/')[-1]),
                             cv2.IMREAD_GRAYSCALE)      # 读取处理过的标签的灰度标注

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:    # 如果尺寸大小不匹配，则抛出异常
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        label_class = np.unique(label).tolist()     # 获取查询标签中包含的的所有类别
        if 0 in label_class:        # 如果列表中有背景，删除背景
            label_class.remove(0)
        if 255 in label_class:      # 删除忽视标签
            label_class.remove(255)
        new_label_class = []
        for c in label_class:       # 遍历标签列表，筛选出满足当前模式的列表
            if c in self.sub_val_list:
                if self.mode == 'val' or self.mode == 'demo' or self.mode == 'finetune':
                    new_label_class.append(c)       # 如果该标签属于未知类，且处于验证等模式时，将其加入新的类列表
            if c in self.sub_list:
                if self.mode == 'train':
                    new_label_class.append(c)       # 如果该标签属于基类，且处于训练模式时，将其加入新的类列表
        label_class = new_label_class       # 新的类列表更新列表
        assert len(label_class) > 0, f"{image_path}, {label_path}"     # 确保类别不为0

        class_chosen = label_class[random.randint(1, len(label_class)) - 1]     # 在类别列表随机选择一个类别

        target_pix = np.where(label == class_chosen)    # 获取标签中选择类别的坐标
        ignore_pix = np.where(label == 255)     # 获取标签中忽视类别的坐标
        label[:, :] = 0     # 初始化列表
        if target_pix[0].shape[0] > 0:      # 如果该类别的像素数量不为0
            label[target_pix[0], target_pix[1]] = 1     # 将对应的类别全部设为1
        label[ignore_pix[0], ignore_pix[1]] = 255       # 将忽视的地方全部设为255

        # for cls in range(1,self.num_classes+1):
        #     select_pix = np.where(label_b_tmp == cls)
        #     if cls in self.sub_list:
        #         label_b[select_pix[0],select_pix[1]] = self.sub_list.index(cls) + 1
        #     else:
        #         label_b[select_pix[0],select_pix[1]] = 0

        file_class_chosen = self.sub_class_file_list[class_chosen]      # 获取包含该标签的所有图片列表
        num_file = len(file_class_chosen)       # 获取图片数量

        support_image_path_list = []        # 支持图像列表
        support_label_path_list = []        # 支持标签列表
        support_idx_list = []       # 支持下标列表
        for k in range(self.shot):
            support_idx = random.randint(1, num_file) - 1       # 随机获取一个图像的下标
            support_image_path = image_path     # 初始化查询图像和标签的地址
            support_label_path = label_path

            while (support_image_path == image_path and support_label_path == label_path) or support_idx in \
                    support_idx_list:       # 当查询和支持完全一样时，或支持图像已使用时，循环查找新的支持数据
                support_idx = random.randint(1, num_file) - 1       # 随机获取一个图像的下标
                support_image_path, support_label_path = file_class_chosen[support_idx]     # 获取图像和标签的地址
            support_idx_list.append(support_idx)        # 将信息加入对应的列表中
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        support_image_list_ori = []
        support_label_list_ori = []
        support_label_list_ori_mask = []
        subcls_list = []
        if self.mode == 'train':        # 如果在训练模式
            subcls_list.append(self.sub_list.index(class_chosen))       # 获取所选类别在基类列表中的下标
        else:
            subcls_list.append(self.sub_val_list.index(class_chosen))       # 否则获取所选类别在未知类列表中的下标
        for k in range(self.shot):      # 逐个加载支持的图像和标签
            support_image_path = support_image_path_list[k]     # 读取图像和标签的路径
            support_label_path = support_label_path_list[k]
            support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)        # 读取BGR格式图像
            support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)      # 转换为RGB格式
            support_image = np.float32(support_image)        # 转换为float32类型
            support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)        # 读取原始灰度标注
            target_pix = np.where(support_label == class_chosen)         # 获取对应标签所有像素的坐标
            ignore_pix = np.where(support_label == 255)     # 获取所有忽视像素的坐标
            support_label[:, :] = 0     # 初始化支持标签
            support_label[target_pix[0], target_pix[1]] = 1     # 将所有对应标签的像素设为1
            # 根据不同类型（mask/bbox），获取标签掩码
            support_label, support_label_mask = transform_anns(support_label, self.ann_type)  # mask/bbox
            support_label[ignore_pix[0], ignore_pix[1]] = 255       # 将标签掩码的忽视像素设置为255
            support_label_mask[ignore_pix[0], ignore_pix[1]] = 255      # 将原始标签掩码的忽视像素设置为255
            if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                raise (RuntimeError(        # 如果图像和标签尺寸大小不匹配，则抛出异常
                    "Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))
            support_image_list_ori.append(support_image)        # 将对应原始数据加入对应列表
            support_label_list_ori.append(support_label)
            support_label_list_ori_mask.append(support_label_mask)
        assert len(support_label_list_ori) == self.shot and len(support_image_list_ori) == self.shot    # 判断是否和支持量数量相同

        raw_image = image.copy()        # 复制查询图像和其标签的副本
        raw_label = label.copy()
        raw_label_b = label_b.copy()
        support_image_list = [[] for _ in range(self.shot)]     # 根据支持量初始化对应大小的双层列表
        support_label_list = [[] for _ in range(self.shot)]
        if self.transform is not None:
            # 同时对查询的图像和两个标签，以及cv2图像同时transform
            image, label, label_b, img_cv2 = self.transform_tri(image, label, label_b, img_cv2)
            for k in range(self.shot):      # 遍历支持集，对支持数据同时进行transform
                support_image_list[k], support_label_list[k] = self.transform(support_image_list_ori[k],
                                                                              support_label_list_ori[k])

        s_xs = support_image_list       # 最终支持图像
        s_ys = support_label_list       # 最终支持标签
        s_x = s_xs[0].unsqueeze(0)      # 提取图像中第一个数据并扩充一维      [1, 3, h, w]
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)     # 如果有多个支持图片，则拼接
        s_y = s_ys[0].unsqueeze(0)      # 提取图像中第一个数据并扩充一维
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)     # 如果有多个支持标签，则拼接

        # Return
        if self.mode == 'train':        # 如果在训练中，返回查询的图片、名称、标签、原始标签；支持的图像、标签、类别下标、获取的类别和cv2图像
            return image, query_name, label, label_b, s_x, s_y, subcls_list, class_chosen - 1, img_cv2
        elif self.mode == 'val':
            return image, query_name, label, label_b, s_x, s_y, subcls_list, class_chosen - 1, raw_label, \
                raw_label_b, img_cv2, images
        elif self.mode == 'demo':
            total_image_list = support_image_list_ori.copy()
            total_image_list.append(raw_image)
            return image, label, label_b, s_x, s_y, subcls_list, total_image_list, support_label_list_ori, \
                support_label_list_ori_mask, raw_label, raw_label_b


# 基础加载器
class BaseData(Dataset):
    def __init__(self, split=3, mode=None, data_root=None, data_list=None, data_set=None, use_split_coco=False,
                 transform=None, main_process=False, batch_size=None):

        assert data_set in ['pascal', 'coco']  # 错误验证
        assert mode in ['train', 'val']

        if data_set == 'pascal':  # 获取类别数量
            self.num_classes = 20
        elif data_set == 'coco':
            self.num_classes = 80

        self.mode = mode
        self.split = split
        self.data_root = data_root
        self.batch_size = batch_size

        if data_set == 'pascal':
            self.class_list = list(range(1, 21))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            if self.split == 3:
                self.sub_list = list(range(1, 16))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                self.sub_val_list = list(range(16, 21))  # [16,17,18,19,20]
            elif self.split == 2:
                self.sub_list = list(range(1, 11)) + list(range(16, 21))  # [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
                self.sub_val_list = list(range(11, 16))  # [11,12,13,14,15]
            elif self.split == 1:
                self.sub_list = list(range(1, 6)) + list(range(11, 21))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(6, 11))  # [6,7,8,9,10]
            elif self.split == 0:
                self.sub_list = list(range(6, 21))  # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(1, 6))  # [1,2,3,4,5]

        elif data_set == 'coco':
            if use_split_coco:
                print('INFO: using SPLIT COCO (FWB)')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_val_list = list(range(4, 81, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
                elif self.split == 2:
                    self.sub_val_list = list(range(3, 80, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
                elif self.split == 1:
                    self.sub_val_list = list(range(2, 79, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
                elif self.split == 0:
                    self.sub_val_list = list(range(1, 78, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
            else:
                print('INFO: using COCO (PANet)')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_list = list(range(1, 61))
                    self.sub_val_list = list(range(61, 81))
                elif self.split == 2:
                    self.sub_list = list(range(1, 41)) + list(range(61, 81))
                    self.sub_val_list = list(range(41, 61))
                elif self.split == 1:
                    self.sub_list = list(range(1, 21)) + list(range(41, 81))
                    self.sub_val_list = list(range(21, 41))
                elif self.split == 0:
                    self.sub_list = list(range(21, 81))
                    self.sub_val_list = list(range(1, 21))

        print('sub_list: ', self.sub_list)
        print('sub_val_list: ', self.sub_val_list)

        self.data_list = []
        list_read = open(data_list).readlines()  # 读取数据
        print("Processing data...")

        for l_idx in tqdm(range(len(list_read))):  # 从文本文件中读取图像和标签的路径对
            line = list_read[l_idx]  # 获取当前行文本
            line = line.strip()  # 去除首尾空白字符
            line_split = line.split(' ')  # 按空格分割行内容
            image_name = os.path.join(self.data_root, line_split[0])  # 拼接绝对路径
            label_name = os.path.join(self.data_root, line_split[1])
            item = (image_name, label_name)
            self.data_list.append(item)

        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]  # 获取图片和掩码的路径
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 以BGR格式加载彩色图像
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # 将图像从BGR颜色空间转换为RGB颜色空间
        image = np.float32(image)   # 将图像像素值的数据类型转换为32位浮点数
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)    # 以灰度模式读取标签图像
        label_tmp = label.copy()    # 创建标签数据的深拷贝副本

        for cls in range(1, self.num_classes + 1):      # 遍历所有可能的类别
            select_pix = np.where(label_tmp == cls)     # 获取标签副本中所有属于当前类别的像素坐标
            if cls in self.sub_list:    # 检查当前类别是否是需要保留的子类别
                label[select_pix[0], select_pix[1]] = self.sub_list.index(cls) + 1
            else:   # 处理不属于子类别的像素（背景或noval类），将其设为0
                label[select_pix[0], select_pix[1]] = 0

        raw_label = label.copy()    # 创建基础类别标签的深拷贝副本

        if self.transform is not None:      # 对图像和标签进行转换
            image, label = self.transform(image, label)

        # Return
        if self.mode == 'val' and self.batch_size == 1:
            return image, label, raw_label
        else:
            return image, label
