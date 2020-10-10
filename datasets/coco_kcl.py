########################################################################

########################################################################

import os
import numpy as np
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as fn

from lib.utils.tools import get_file_path

########################################################################


class ImageFolder(Dataset):
    def __init__(self, data_dir, extension='.jpg', img_size=416):
        self.data_list = get_file_path(1, data_dir, extension=extension)[0]
        self.img_size = img_size
        self.total = self.data_list.shape[0]

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        img_path = self.data_list[index % len(self.data_list)]
        img = Image.open(img_path)
        img = img.convert('RGB')

        w, h = img.size

        pad1 = np.abs(w - h) // 2
        pad2 = np.abs(w - h) - pad1

        # The input image should be a square shape. Pad the image with 0
        # if the "h" and "w" of a image are not equal.
        border = (0, pad1, 0, pad2) if w >= h else (pad1, 0, pad2, 0)
        img = ImageOps.expand(img, border, fill=0)

        # Resize the image shape into the default size of yolov3.
        img = img.resize((self.img_size, self.img_size), Image.ANTIALIAS)

        # Fit into pytorch format (C, H, W)
        img = np.array(img).transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).float()

        return img_path, img_tensor

########################################################################
# For testing the functions defined above.


# import matplotlib.pyplot as plt
# img_path = '/Users/kcl/Desktop/_test_img'

# if __name__ == '__main__':
#     dataset = ImageFolder(img_path, extension='.jpg', img_size=416)
#     data_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)

#     for path, img in data_loader:
#         img = img.numpy().transpose(0, 2, 3, 1) / 255.0
#         batch_size = img.shape[0]

#         fig, axes = plt.subplots(1, 2)
#         fig.subplots_adjust(hspace=0.6, wspace=0.6)

#         for n, ax in enumerate(axes.flat):
#             try:
#                 ax.imshow(img[n], interpolation='spline16')
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#             except:
#                 continue

#         plt.show()
#         break

########################################################################

class ImageLabel(Dataset):
    def __init__(self, img_dir, lab_dir, train=True,
                 extension='.jpg|.txt', img_size=416):
        img_list = get_file_path(1, img_dir, extension=extension)[0]
        lab_list = get_file_path(1, lab_dir, extension=extension)[0]
        self.img_list = np.sort(img_list)
        self.lab_list = np.sort(lab_list)

        # If the provided label files and image files are not match,
        # only pick up those file names contained in both files.
        if self.img_list.shape != self.lab_list.shape:
            im_ext = os.path.splitext(os.path.basename(self.img_list[0]))[1]
            im = np.array([os.path.splitext(os.path.basename(i))[0]
                           for i in self.img_list])

            lb_ext = os.path.splitext(os.path.basename(self.lab_list[0]))[1]
            lb = np.array([os.path.splitext(os.path.basename(i))[0]
                           for i in self.lab_list])

            # Get the intersections of 2 lists.
            inter = np.intersect1d(im, lb)
            self.img_list = np.array([os.path.join(
                img_dir, i + im_ext) for i in inter])
            self.lab_list = np.array([os.path.join(
                lab_dir, i + lb_ext) for i in inter])

        self.train = train
        self.img_size = img_size
        self.total = self.img_list.shape[0]

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        img_path = self.img_list[index % len(self.img_list)]

        img = Image.open(img_path)
        img = img.convert('RGB')
        w, h = img.size

        pad1 = np.abs(w - h) // 2
        pad2 = np.abs(w - h) - pad1

        # The input image should be a square shape. Pad the image with 0
        # if the "h" and "w" of a image are not equal.
        border = (0, pad1, 0, pad2) if w >= h else (pad1, 0, pad2, 0)
        img = ImageOps.expand(img, border, fill=0)

        # Resize the image shape into the default size of yolov3.
        img = img.resize((self.img_size, self.img_size), Image.ANTIALIAS)

        # --------------------------------------------------------------

        lab_path = self.lab_list[index % len(self.lab_list)]
        labs = np.loadtxt(lab_path).reshape(-1, 5)

        # It's crutial to be fully aware of what is the "1" total length
        # when adjusting relative coordinates.
        if w >= h:
            pad1 = pad1 / w
            labs[:, 2] = labs[:, 2] * h / w + pad1
            labs[:, 4] = labs[:, 4] * h / w
        else:
            pad1 = pad1 / h
            labs[:, 1] = labs[:, 1] * w / h + pad1
            labs[:, 3] = labs[:, 3] * w / h

        rand_hue = np.random.uniform(-0.3, 0.3)
        rand_num = [np.random.uniform(0.8, 1.2) for i in range(4)]

        if self.train:
            img = fn.adjust_hue(img, rand_hue)
            img = fn.adjust_contrast(img, rand_num[0])
            img = fn.adjust_brightness(img, rand_num[1])
            img = fn.adjust_saturation(img, rand_num[2])

            if rand_num[3] > 1.0:
                img = fn.hflip(img)
                labs[:, 1] = 1.0 - labs[:, 1]

        # Fit into pytorch format (C, H, W)
        img = np.array(img).transpose(2, 0, 1)
        return img, labs

########################################################################


def collate_fn(batch):
    imgs = torch.tensor([batch[i][0] for i in range(len(batch))])

    max_labs = np.max([len(batch[i][1]) for i in range(len(batch))])
    labels = np.zeros([len(batch), max_labs, 5])

    for i in range(len(batch)):
        labels[i, :len(batch[i][1])] = batch[i][1]

    labels = torch.tensor(labels)
    return imgs, labels


########################################################################
# For testing the functions defined above.

import matplotlib.pyplot as plt
import matplotlib.patches as patches

img_path = '/Users/kcl/Documents/Python_Projects/coco/images/train2014'
lab_path = '/Users/kcl/Documents/Python_Projects/coco/labels/train2014'
colors = [tuple(1.0 * np.random.rand(3)) for _ in range(80)]
B = 3

if __name__ == '__main__':
    dataset = ImageLabel(img_path, lab_path)
    data_loader = DataLoader(dataset=dataset, batch_size=B,
                             shuffle=False, collate_fn=collate_fn)

    n = 0
    for img, lbs in data_loader:
        img = img.numpy().transpose(0, 2, 3, 1) / 255.0
        labs = lbs.numpy()[:, :, 0].astype(np.int)
        bboxes = lbs.numpy()[:, :, 1:].astype(np.float)
        print(labs)

        fig, ax = plt.subplots(1)
        for ID in range(B):
            for bbox, lb in zip(bboxes[ID], labs[ID]):
                xmin = bbox[0] - bbox[2] / 2
                ymin = bbox[1] - bbox[3] / 2
                rec = patches.Rectangle((xmin * 416, ymin * 416),
                                        bbox[2] * 416, bbox[3] * 416,
                                        edgecolor=colors[lb],
                                        facecolor='none')
                ax.add_patch(rec)
            break

        ax.imshow(img[0])
        plt.show()

        n += 1
        if n == 5:
            break
