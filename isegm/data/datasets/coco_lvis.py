from pathlib import Path
import pickle
import random
import numpy as np
import json
import cv2
import os
from copy import deepcopy
from isegm.data.base import ISDataset
from isegm.data.sample import DSample

class CocoLvisDataset(ISDataset):
    def __init__(self, dataset_path, split='train', stuff_prob=0.0,
                 allow_list_name=None, anno_file='hannotation.pickle', **kwargs):
        super(CocoLvisDataset, self).__init__(**kwargs)
        dataset_path = Path(dataset_path)
        # 修改为正确的路径
        self._split_path = dataset_path / 'cocolvis_annotation' / split
        self.split = split
        # 修改为实际的图片文件夹路径
        self._images_path = dataset_path / 'train2017'
        self._masks_path = self._split_path / 'masks'
        self.stuff_prob = stuff_prob

        # 打印文件路径进行调试
        file_path = self._split_path / anno_file
        print(f"尝试打开的文件路径: {file_path}")

        try:
            with open(file_path, 'rb') as f:
                self.dataset_samples = sorted(pickle.load(f).items())
        except FileNotFoundError:
            print(f"文件 {file_path} 不存在，请检查路径和文件是否正确。")
            raise

        if allow_list_name is not None:
            allow_list_path = self._split_path / allow_list_name
            with open(allow_list_path, 'r') as f:
                allow_images_ids = json.load(f)
            allow_images_ids = set(allow_images_ids)

            self.dataset_samples = [sample for sample in self.dataset_samples
                                    if sample[0] in allow_images_ids]

    def get_sample(self, index) -> DSample:
        image_id, sample = self.dataset_samples[index]
        image_path = self._images_path / f'{image_id}.jpg'
        # print(f"尝试读取的图像文件路径: {image_path}")  # 添加调试信息

        if not os.path.exists(image_path):
            print(f"文件 {image_path} 不存在，请检查路径。")
            return None  # 可以选择跳过该样本或者进行其他处理

        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"无法读取文件 {image_path}，可能文件损坏。")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"读取文件 {image_path} 时出错: {e}")
            return None  # 可以选择跳过该样本或者进行其他处理

        packed_masks_path = self._masks_path / f'{image_id}.pickle'
        try:
            with open(packed_masks_path, 'rb') as f:
                encoded_layers, objs_mapping = pickle.load(f)
        except FileNotFoundError:
            print(f"文件 {packed_masks_path} 不存在，请检查路径。")
            return None  # 可以选择跳过该样本或者进行其他处理
        except Exception as e:
            print(f"读取文件 {packed_masks_path} 时出错: {e}")
            return None  # 可以选择跳过该样本或者进行其他处理

        layers = [cv2.imdecode(x, cv2.IMREAD_UNCHANGED) for x in encoded_layers]
        layers = np.stack(layers, axis=2)

        instances_info = deepcopy(sample['hierarchy'])
        for inst_id, inst_info in list(instances_info.items()):
            if inst_info is None:
                inst_info = {'children': [], 'parent': None, 'node_level': 0}
                instances_info[inst_id] = inst_info
            inst_info['mapping'] = objs_mapping[inst_id]

        if self.stuff_prob > 0 and random.random() < self.stuff_prob:
            for inst_id in range(sample['num_instance_masks'], len(objs_mapping)):
                instances_info[inst_id] = {
                    'mapping': objs_mapping[inst_id],
                    'parent': None,
                    'children': []
                }
        else:
            for inst_id in range(sample['num_instance_masks'], len(objs_mapping)):
                layer_indx, mask_id = objs_mapping[inst_id]
                layers[:, :, layer_indx][layers[:, :, layer_indx] == mask_id] = 0

        return DSample(image, layers, objects=instances_info)
