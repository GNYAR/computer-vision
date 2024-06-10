import os
from tempfile import TemporaryDirectory
import time

import torch
import torchvision


def build_resnet(class_number, trainable=True):
    resnet = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.DEFAULT
    )

    for param in resnet.parameters():
        param.requires_grad = trainable

    num_ftrs = resnet.fc.in_features
    # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    resnet.fc = torch.nn.Linear(num_ftrs, class_number)

    return resnet


def evaluate(model, criterion, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    dataset_size = 0
    # Iterate over data.
    for inputs, labels in dataloader:

        dataset_size += inputs.size(0)

        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size

    print(f"test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc.cpu().numpy()


def train_model(
    model,
    criterion,
    optimizer,
    dataloaders,
    dataset_sizes,
    num_epochs=25,
    patience=0,
    scheduler=None,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    since = time.time()
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as dir:
        best_model_params_path = os.path.join(dir, "best_model_params.pt")
        torch.save(model.state_dict(), best_model_params_path)

        best_loss = None
        best_acc = 0
        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        patience_cnt = 0
        for epoch in range(num_epochs):
            print(f"Epoch {epoch:3d}/{num_epochs - 1}", end=" ")
            # Each epoch has a training and validation phase
            for phase in ["train", "validate"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == "train" and scheduler:
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(
                    f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}",
                    end=" " if phase == "train" else "\n",
                )
                acc = epoch_acc.cpu().numpy()
                if phase == "train":
                    history["loss"].append(epoch_loss)
                    history["accuracy"].append(acc)
                else:
                    # deep copy the model
                    history["val_loss"].append(epoch_loss)
                    history["val_accuracy"].append(acc)
                    if acc > best_acc:
                        best_acc = acc
                    if epoch == 0 or epoch_loss < best_loss:
                        best_loss = epoch_loss
                        patience_cnt = 0
                        torch.save(model.state_dict(), best_model_params_path)
                    else:
                        patience_cnt += 1
            if patience_cnt > patience:
                break

        time_elapsed = time.time() - since
        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best validate Acc: {best_acc:4f}")

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model, history
