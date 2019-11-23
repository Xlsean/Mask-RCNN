########################################################################

# Constructing a customized dataset with Pytorch "utils.data.Dataset"
# module.

########################################################################

import numpy as np
from PIL import Image
from xml.etree import ElementTree as ET

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as fn

from utils_ops import get_file_path

########################################################################

# Total class number = 20 + 1 (background is counted as 0)
VOC_LABELS = {
    'none': (20, 'Background'), 20: ('none', 'Background'),
    'aeroplane': (0, 'Vehicle'), 0: ('aeroplane', 'Vehicle'),
    'bicycle': (1, 'Vehicle'), 1: ('bicycle', 'Vehicle'),
    'bird': (2, 'Animal'), 2: ('bird', 'Animal'),
    'boat': (3, 'Vehicle'), 3: ('boat', 'Vehicle'),
    'bottle': (4, 'Indoor'), 4: ('bottle', 'Indoor'),
    'bus': (5, 'Vehicle'), 5: ('bus', 'Vehicle'),
    'car': (6, 'Vehicle'), 6: ('car', 'Vehicle'),
    'cat': (7, 'Animal'), 7: ('cat', 'Animal'),
    'chair': (8, 'Indoor'), 8: ('chair', 'Indoor'),
    'cow': (9, 'Animal'), 9: ('cow', 'Animal'),
    'diningtable': (10, 'Indoor'), 10: ('diningtable', 'Indoor'),
    'dog': (11, 'Animal'), 11: ('dog', 'Animal'),
    'horse': (12, 'Animal'), 12: ('horse', 'Animal'),
    'motorbike': (13, 'Vehicle'), 13: ('motorbike', 'Vehicle'),
    'person': (14, 'Person'), 14: ('person', 'Person'),
    'pottedplant': (15, 'Indoor'), 15: ('pottedplant', 'Indoor'),
    'sheep': (16, 'Animal'), 16: ('sheep', 'Animal'),
    'sofa': (17, 'Indoor'), 17: ('sofa', 'Indoor'),
    'train': (18, 'Vehicle'), 18: ('train', 'Vehicle'),
    'tvmonitor': (19, 'Indoor'), 19: ('tvmonitor', 'Indoor')
}

########################################################################


class CustomDataset(Dataset):
    def __init__(self, data_dir, label_dir, extension,
                 train=True, img_size=416):
        data_list = get_file_path(1, data_dir, extension=extension)[0]
        label_list = get_file_path(1, label_dir, extension=extension)[0]

        self.data_list = np.sort(data_list)
        self.label_list = np.sort(label_list)
        self.img_size = img_size
        self.train = train
        self.total = self.data_list.shape[0]

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        data_path = self.data_list[index % self.total]
        xml_path = self.label_list[index % self.total]

        xml_tree = ET.parse(xml_path)
        xml_root = xml_tree.getroot()

        filename = xml_root.find('filename').text
        hwc = np.array([
            int(xml_root.find('size').find('height').text),
            int(xml_root.find('size').find('width').text),
            int(xml_root.find('size').find('depth').text)
        ])

        labels = []
        bboxes = []
        for obj in xml_root.findall('object'):
            labels.append(VOC_LABELS[obj.find('name').text][0])

            bbox = []
            bbox.append(float(obj.find('bndbox').find('xmin').text))
            bbox.append(float(obj.find('bndbox').find('ymin').text))
            bbox.append(float(obj.find('bndbox').find('xmax').text))
            bbox.append(float(obj.find('bndbox').find('ymax').text))
            bboxes.append(bbox)

        labels = np.array(labels).astype(np.int)
        bboxes = np.array(bboxes).astype(np.int)

        # --------------------------------------------------------------

        rand_scale = np.random.uniform(0.85, 1.0)
        crop_size = int(np.min([hwc[0], hwc[1]]) * rand_scale)

        rand_i = np.random.randint(
            0, hwc[0] - crop_size) if hwc[0] > crop_size else 0
        rand_j = np.random.randint(
            0, hwc[1] - crop_size) if hwc[1] > crop_size else 0

        rand_hue = np.random.uniform(-0.3, 0.3)
        rand_num = [np.random.uniform(0.6, 1.5) for i in range(4)]

        img = Image.open(data_path)
        img = fn.resized_crop(img, rand_i, rand_j,
                              crop_size, crop_size,
                              (self.img_size, self.img_size),
                              interpolation=2)

        ratio = self.img_size / crop_size
        bboxes = np.concatenate([
            np.maximum(bboxes[:, 0:1], rand_j) - rand_j,
            np.maximum(bboxes[:, 1:2], rand_i) - rand_i,
            np.minimum(bboxes[:, 2:3], rand_j + crop_size) - rand_j,
            np.minimum(bboxes[:, 3:], rand_i + crop_size) - rand_i
        ], axis=-1) * ratio

        if self.train:
            img = fn.adjust_hue(img, rand_hue)
            img = fn.adjust_contrast(img, rand_num[0])
            img = fn.adjust_brightness(img, rand_num[1])
            img = fn.adjust_saturation(img, rand_num[2])

            if rand_num[3] > 1.05:
                img = fn.hflip(img)
                bboxes = np.concatenate([
                    (bboxes[:, 2:3] - self.img_size) * (-1), bboxes[:, 1:2],
                    (bboxes[:, 0:1] - self.img_size) * (-1), bboxes[:, 3:]
                ], axis=-1)

        img = np.array(img).transpose(2, 0, 1)
        bboxes = bboxes.astype(np.int)

        return img, filename, hwc, labels, bboxes


########################################################################


def collate_fn(batch):
    imgs = torch.tensor([batch[i][0] for i in range(len(batch))])
    filenames = [batch[i][1] for i in range(len(batch))]
    hwc = torch.tensor([batch[i][2] for i in range(len(batch))])

    max_lab = np.max([len(batch[i][3]) for i in range(len(batch))])
    max_bbox = np.max([len(batch[i][4]) for i in range(len(batch))])

    labels = np.zeros([len(batch), max_lab])
    bboxes = np.zeros([len(batch), max_bbox, 4])

    for i in range(len(batch)):
        labels[i, :len(batch[i][3])] = batch[i][3]
        bboxes[i, :len(batch[i][4]), :] = batch[i][4]

    bboxes = bbox_filter(bboxes)
    bboxes = torch.tensor(bboxes)
    labels = torch.tensor(labels)

    return imgs, filenames, hwc, labels, bboxes


def bbox_filter(bbox):
    idx_x = (bbox[:, :, 2] - bbox[:, :, 0] <= 0)
    idx_y = (bbox[:, :, 3] - bbox[:, :, 1] <= 0)
    bbox[idx_x] = [0, 0, 0, 0]
    bbox[idx_y] = [0, 0, 0, 0]
    return bbox.astype(np.int32)

########################################################################
# For testing the functions defined above.


import matplotlib.pyplot as plt
import matplotlib.patches as patches


VOC_TRAIN = '/Users/kcl/Documents/Python_Projects/VOCROOT/VOC2007_train/JPEGImages'
VOC_TEST = '/Users/kcl/Documents/Python_Projects/VOCROOT/VOC2007_test/JPEGImages'
xml_test = '/Users/kcl/Documents/Python_Projects/VOCROOT/VOC2007_test/Annotations'
VOC2012 = '/Users/kcl/Documents/Python_Projects/VOCROOT/VOC2012/JPEGImages'

batch_size = 3
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]


if __name__ == '__main__':
    custom_dataset = CustomDataset(VOC_TEST, xml_test,
                                   train=False,
                                   extension='.xml|.jpg')
    train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               collate_fn=collate_fn)

    for img, filename, hwc, labels, bboxes in train_loader:
        images = img.numpy() / 255.0
        lbs = labels.numpy().astype(np.int)
        bboxes = bboxes.numpy().astype(np.int)
        print(filename)

        # print('img: ', img.shape)
        # print('filename: ', filename)
        # print('hwc: ', hwc)
        print('labels: ', labels)
        print('bboxes: ', bboxes)

        fig, ax = plt.subplots(1)

        frame = {}
        for ID in range(batch_size):
            for color, bbox, lb in zip(colors, bboxes[ID], lbs[ID]):
                tl = (bbox[0], bbox[1])
                W = bbox[2] - bbox[0]
                H = bbox[3] - bbox[1]

                rec = patches.Rectangle(tl, W, H,
                                        linewidth=2,
                                        edgecolor='r',
                                        facecolor='none')
                ax.add_patch(rec)
            break

        # cv2.imshow('frame', frame)
        images = images.transpose(0, 2, 3, 1)
        ax.imshow(images[0])
        plt.show()

#         break
