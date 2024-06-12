import glob
import os

import numpy as np
from PIL import Image
import torch


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.img_paths = img_paths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx, 0]
        image = Image.open(img_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def get_paths_labels_classes(dataset_dir):
    folder_paths = glob.glob(os.path.join(dataset_dir, "*"), recursive=True)
    classes = [x.split(os.path.sep)[-1] for x in folder_paths]
    img_paths = []
    labels = []
    for i, folder_path in enumerate(folder_paths):
        jpg_paths = glob.glob(os.path.join(folder_path, "*.jpg"), recursive=True)
        img_paths.extend(jpg_paths)
        labels.extend([i] * len(jpg_paths))
    img_paths = np.array(img_paths, dtype=object)
    labels = np.array(labels)

    return img_paths.reshape((-1, 1)), labels, classes
