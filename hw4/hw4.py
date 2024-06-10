import glob
import os
import time

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.backends import cudnn
from torchvision import transforms

from dataset import ImageDataset
from dataset import get_paths_labels_classes
from training import build_resnet
from training import evaluate
from training import train_model

DATASET_PATH = "D:\\CCSN_v2"
paths, labels, classes = get_paths_labels_classes(DATASET_PATH)
# dataset = ImageDataset(paths, labels)

device = torch.device("cpu")
if torch.cuda.is_available():
    print("gpu is available")
    device = torch.device("cuda:0")
else:
    print("cpu only")

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "validate": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

# 80 % for training and validation sets, 20 % for the test set
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

batch_size = 64
epochs = 100
patience = epochs // 5

test_accuracy = []
test_loss = []
histories = []

criterion = nn.CrossEntropyLoss()

for i, (train_idx, test_idx) in enumerate(skf.split(paths, labels)):
    print("-" * 50)
    x_train, x_valid, y_train, y_valid = train_test_split(
        paths[train_idx], labels[train_idx], test_size=0.2, random_state=i
    )

    testloader = torch.utils.data.DataLoader(
        ImageDataset(
            paths[test_idx],
            labels[test_idx],
            data_transforms["train"],
            lambda x: torch.tensor(x, dtype=torch.long),
        ),
        batch_size=1,
        shuffle=True,
    )
    trainloader = torch.utils.data.DataLoader(
        ImageDataset(
            x_train,
            y_train,
            data_transforms["train"],
            lambda x: torch.tensor(x, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    validloader = torch.utils.data.DataLoader(
        ImageDataset(
            x_valid,
            y_valid,
            data_transforms["validate"],
            lambda x: torch.tensor(x, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    resnet = build_resnet(len(classes), True)
    resnet = resnet.to(device)
    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(resnet.parameters())

    trained_model, history = train_model(
        resnet,
        criterion,
        optimizer_ft,
        dataloaders={"train": trainloader, "validate": validloader},
        dataset_sizes={"train": x_train.shape[0], "validate": x_valid.shape[0]},
        patience=patience,
        num_epochs=epochs,
        scheduler=None,
    )

    histories.append(history)
    print(f"Fold {i+1:2d}", end=" ")
    loss, acc = evaluate(trained_model, criterion, testloader)

    test_loss.append(loss)
    test_accuracy.append(acc)

# display result
plt.figure(figsize=(8, 3 * len(histories)))

max_loss = 0
max_acc = 0
for i, history in enumerate(histories):
    max_loss = max(max_loss, np.max(history["loss"]), np.max(history["val_loss"]))

max_loss *= 1.05
for i, history in enumerate(histories):
    plt.subplot(len(histories), 2, i * 2 + 1)
    plt.title(f"fold:{i+1}")
    plt.plot(history["loss"], label="train")
    plt.plot(history["val_loss"], label="valid")
    plt.xlabel("epoch")
    step = int(np.ceil(len(history["loss"]) / 5))
    plt.xticks(
        np.arange(0, len(history["loss"]), step),
        [str(u + 1) for u in np.arange(0, len(history["loss"]), step)],
    )
    plt.ylabel("loss")
    plt.ylim([0, max_loss])
    plt.grid(True)
    plt.legend()

    plt.subplot(len(histories), 2, i * 2 + 2)
    plt.title(f"fold:{i+1}")
    plt.plot(history["accuracy"], label="train")
    plt.plot(history["val_accuracy"], label="valid")
    plt.xticks(
        np.arange(0, len(history["accuracy"]), step),
        [str(u + 1) for u in np.arange(0, len(history["accuracy"]), step)],
    )
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.ylim([0, 1.0])
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()

plt.figure()
plt.bar(x=np.arange(len(test_accuracy)), height=np.array(test_accuracy))
plt.xlabel("fold")
plt.xticks(
    np.arange(len(test_accuracy)), [str(i + 1) for i in np.arange(len(test_accuracy))]
)
plt.ylabel("accuracy")
plt.title(
    f"average accuracy rate:{np.mean(np.array(test_accuracy)):.3f}+/-{np.std(np.array(test_accuracy)):.3f}"
)
plt.grid(True)
plt.ylim([0, 1.0])
plt.show()
print(
    f"average accuracy rate:{np.mean(np.array(test_accuracy)):.3f}+/-{np.std(np.array(test_accuracy)):.3f}"
)
