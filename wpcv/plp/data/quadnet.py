# coding: utf-8
from __future__ import division

import sys
import os
import config as cfg

sys.path.append(os.path.abspath('..'))
import cv2
import random, os, glob, json
from utils.centernet_utils import draw_points_heatmap
from torch.utils.data import dataset
import numpy as np

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from matplotlib import pyplot as plt
from utils.myutils import normalize
import wpcv
from wpcv.utils.augmentations import object_detection as transforms
from wpcv.utils.augmentations import base as BT
from wpcv.utils.transforms import pil as IMG

from torch.utils.data import dataset,dataloader

class Dataset(dataset.Dataset):
    def __init__(self, imgs_dir, labels_dir, batch_size=8, classes=['0'], shuffle=True,input_size=cfg.TRAIN_INPUT_SIZE):
        self.classes = classes
        self.label2id = dict(zip(classes, range(0, len(classes))))
        self.imgs_dir = imgs_dir
        self.labels_dir = labels_dir
        self.input_size=input_size
        self.data_pairs = self._load_annotations()
        self.shuffle = shuffle
        if shuffle:
            random.shuffle(self.data_pairs)
        self.batch_size = batch_size
        self.num_batches = len(self.data_pairs) // batch_size
        self.currect_batch_index = 0
        # random.seed(0)
        self.transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Limitsize(600),
            transforms.RandomMultiChoice([
                transforms.RandomRotate(30),
                transforms.RandomShear(15,15),
                transforms.RandomTranslate((100, 100)),
                # transforms.RandomVerticalFlip(),
                # transforms.RandomHorizontalFlip(),
            ],[0.2,0.2,0.2,0.5,0.5]
            ),
            transforms.Zip([
                BT.Compose([
                    BT.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.2),
                    BT.RandomApply([IMG.blur],p=0.3),
                    BT.RandomApply([IMG.edge_enhance],p=0.3)
                ]),
                transforms.Identical(),
            ]),
            transforms.Resize(self.input_size,keep_ratio=True,fillcolor=(0,0,0)),
        ])

        # print(self.num_batches,self.data_pairs)

    def _load_annotations(self):
        '''
        {shapes:[{points:[],label:""}]}
        labeled using labeme
        :return:
        '''
        fs = glob.glob(self.labels_dir + '/*.json')
        annotations = []
        for i, f in enumerate(fs):
            img_path = self.imgs_dir + '/' + os.path.basename(f).replace('.json', '.jpg')
            with open(f, 'r') as fp:
                dic = json.load(fp)
            # objects=[(obj['points'],obj['label']) for obj in dic['shapes']]
            objects = [[*obj['points'], self.label2id[obj['label']]] for obj in dic['shapes']]
            annotations.append((img_path, objects))
        return annotations

    def _preprocess(self, img, polygon):
        # print(polygon)
        img,[points]=self.transform(img,[polygon])
        # wpcv.draw_polygon(img,points,width=5).show()
        # raise
        # print(img.size)
        img = BT.cv2img(img)


        # h,w=img.shape[:2]
        # dst_w,dst_h=self.input_size
        # scaleX,scaleY=dst_w/w,dst_h/h
        # print(scaleX,scaleY)
        # img = cv2.resize(img, (512, 512))
        # points=np.array(points)*np.array([scaleX,scaleY])

        img = img / 255
        img = normalize(img)
        # plt.matshow(img)
        img = np.transpose(img, (2, 0, 1))
        # print(img.shape)

        points=(np.array(points)/4).astype(np.int)
        heatmap = draw_points_heatmap(points, (128,128),radius=3)

        # plt.matshow(heatmap)
        # plt.show()
        heatmap=np.expand_dims(heatmap,0)
        # raise
        return img, heatmap

    def __iter__(self):
        return self
    def __getitem__(self, item):
        return self.data_pairs[item]
    def __next__(self):
        start_point = self.batch_size * self.currect_batch_index
        end_point = start_point + self.batch_size
        data_pairs = self.data_pairs[start_point:end_point]
        batch_image = []
        batch_heatmap = []
        for i, (img_path, objects) in enumerate(data_pairs):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            points = []
            for obj in objects:
                points += obj[:-1]
            img, heatmap = self._preprocess(img, points)
            # print(img.shape)
            batch_image.append(img)
            batch_heatmap.append(heatmap)
        # print(batch_image)
        batch_image = np.array(batch_image).astype(np.float32)
        batch_heatmap = np.array(batch_heatmap).astype(np.float32)
        if self.currect_batch_index >= self.num_batches - 1:
            self.currect_batch_index = 0
            random.shuffle(self.data_pairs)
            raise StopIteration
        else:
            self.currect_batch_index += 1
            return batch_image, batch_heatmap

    def __len__(self):
        return self.num_batches
