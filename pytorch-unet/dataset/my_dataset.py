from torch.utils.data import Dataset
import cv2
import numpy as np


def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = np.array(img).astype(np.int64)
    else:
        img = cv2.imread(path)
        img = np.array(img, dtype=np.float32) / 255.0
    return img


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        with open(txt_path, 'r') as fh:
            imgs_file = []
            for line in fh:
                line = line.rstrip()
                words = line.split()
                imgs_file.append(words)
        self.imgs_file = imgs_file
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        img_file, label_file = self.imgs_file[item]

        img = load_img(img_file)
        img = img.transpose([2, 0, 1])

        label = load_img(label_file, grayscale=True)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.imgs_file)
