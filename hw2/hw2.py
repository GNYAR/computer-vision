import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torchvision.transforms import v2


def gaussian_kernel(size, sigma):
    """generate 2D gaussian kernel"""
    x = np.zeros((size, size))
    x[size // 2, size // 2] = 1
    return gaussian_filter(x, sigma)


def build_model(n):
    """2D depthwise convolution with gaussian kernel"""
    k_size = 5
    conv = torch.nn.Conv2d(n, n, k_size, padding="same", bias=False, groups=n)
    k = gaussian_kernel(k_size, 1)
    with torch.no_grad():
        conv.weight.copy_(torch.from_numpy(k))
    conv.train(False)
    return conv


def to_tensor(x):
    """convert cv2 image to tensor (channels-last to channels-first)"""
    return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])(x)


def to_img(x):
    """convert tensor to cv2 image (channels-first to channels-last)"""
    return x.detach().numpy().transpose((1, 2, 0)).clip(0, 255).astype(np.uint8)


def fn1(img, k):
    """version 1: apply operations to RGB channels"""
    blur = build_model(3)

    img_t = to_tensor(img)
    x = to_img(img_t + k * (img_t - blur(img_t)))
    return x


def fn2(img, k):
    """version 2: apply operations to Y channel (YUV color space)"""
    blur = build_model(1)

    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv_t = to_tensor(yuv)
    i = yuv_t[0].unsqueeze(0)
    yuv_t[0] = (i + k * (i - blur(i)))[0, ...]
    x = cv2.cvtColor(to_img(yuv_t), cv2.COLOR_YUV2BGR)
    return x


def add_image(img, idx, title):
    plt.subplot(4, 3, idx)
    plt.imshow(img)
    plt.axis(False)
    plt.title(title)


for v, fn in enumerate([fn1, fn2]):
    plt.figure(figsize=(9, 12))  # 3:4
    for i in [1, 2, 3]:
        img = cv2.imread(f"test_image_{i}.jpg")[:, :, [2, 1, 0]]
        add_image(img, i, "Original")
        for j, val in enumerate([1, 5, 10]):
            add_image(fn(img, val), i + (j + 1) * 3, f"k={val}")
    plt.tight_layout()
    plt.savefig(f"result_{v + 1}.png")
    plt.show()
