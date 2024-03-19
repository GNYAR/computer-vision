import math
import os

import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

# get image
URL = "https://avatars.githubusercontent.com/u/48669636?v=4"
FILE_NAME = "dog"
EXT_NAME = "png"
torch.hub.download_url_to_file(URL, f"{FILE_NAME}.{EXT_NAME}")
IMG = cv2.imread(f"{FILE_NAME}.{EXT_NAME}")

if IMG is not None:
    # cv2.imshow
    cv2.imshow("image", IMG)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # plt.imshow
    # openCV: B, G, R | plt: R, G, B
    plt.imshow(IMG[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()

    for i in range(1, 7):
        plt.subplot(2, 3, i)
        plt.imshow(IMG[:, :, [2, 1, 0]])
        plt.title(str(i))
        plt.axis("off")
    plt.show()

    # image rotate and save
    cy, cx = np.array(IMG.shape[0:2]) // 2
    rotation = np.zeros((2, 3))
    org_corners = (
        np.array(
            [
                [0, 0, 1],
                [IMG.shape[1], 0, 1],
                [IMG.shape[1], IMG.shape[0], 1],
                [0, IMG.shape[0], 1],
            ]
        )
        .reshape(-1, 3)
        .T
    )

    for i in range(0, 360, 45):
        # setup a rotation matrix
        rotation[0, 0] = math.cos(i / 180 * math.pi)
        rotation[0, 1] = -math.sin(i / 180 * math.pi)
        rotation[1, 1] = rotation[0, 0]
        rotation[1, 0] = -rotation[0, 1]

        # determine the size of the rotated image
        cpos = rotation.dot(org_corners)
        size = np.asarray(np.max(cpos, axis=1) - np.min(cpos, axis=1), dtype=int)

        # setup the affine transformation matrix
        rotation[:, 2] = (
            rotation[0:2, 0:2].dot(np.array([-cx, -cy]).reshape(2, -1)).ravel()
            + size // 2
        )

        # generate the transformed image
        dst = cv2.warpAffine(IMG, rotation, tuple(size))
        PATH = "result"
        output_name = f"rotation-{str(i)}"

        if not os.path.exists(PATH):
            os.mkdir(PATH)
        if not cv2.imwrite(f"{PATH}/{output_name}.{EXT_NAME}", dst):
            print("could not write image")

        plt.subplot(2, 4, i // 45 + 1)
        plt.imshow(dst[:, :, [2, 1, 0]])
        plt.title(output_name)
        plt.axis("off")
    plt.show()
else:
    print("image file is not found")
