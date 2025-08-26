import pickle as pkl
from pathlib import Path
import cv2
import numpy as np
from pycocotools.coco import COCO
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class COCO2017Dataset(ISDataset):
    def __init__(self, dataset_path, split='train', **kwargs):
        super(COCO2017Dataset, self).__init__(**kwargs)
        assert split in {'val'}
        # assert split in {'train', 'val', 'test'}

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = self.dataset_path / f'{split}2017'
        self._anno_path = self.dataset_path / 'annotations' / f'instances_{split}2017.json'

        self.coco = COCO(str(self._anno_path))
        self.img_ids = self.coco.getImgIds()
        self.ann_ids = self.coco.getAnnIds(imgIds=self.img_ids)

        # 初始化 self.dataset_samples
        self.dataset_samples = self.img_ids

    def get_sample(self, index) -> DSample:
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = str(self._images_path / img_info['file_name'])

        # 检查文件是否存在
        if not Path(image_path).exists():
            print(f"Error: Image file {image_path} does not exist.")
            return None

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Failed to read image file {image_path}.")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        instances_mask = np.zeros((img_info['height'], img_info['width']), dtype=np.int32)
        objects_ids = []
        for i, ann in enumerate(anns):
            mask = self.coco.annToMask(ann)
            instances_mask[mask > 0] = i + 1
            objects_ids.append(i + 1)

        return DSample(image, instances_mask, objects_ids=objects_ids, sample_id=index)
