import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import DiceLoss
from timeit import default_timer as timer
from copy import deepcopy

from networks.vision_transformer import SwinUnet


# Setup our own args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_dir",
    type=str,
    default="../datasets/preprocessed/train",
    help="dir for training slices",
)
parser.add_argument(
    "--val_dir",
    type=str,
    default="../datasets/preprocessed/val",
    help="dir for validation slices",
)
parser.add_argument(
    "--snapshot_dir",
    type=str,
    default="snapshots",
    help="dir for model and data snapshots",
)
parser.add_argument(
    "--base_lr", type=float, default=0.05, help="segmentation network learning rate"
)
parser.add_argument(
    "--max_epoch", type=int, default=300, help="maximum epoch number to train"
)
parser.add_argument(
    "--patience", type=int, default=10, help="num epochs before early stopping"
)
parser.add_argument("--clip", type=int, default=-1, help="number of slices to clip")
parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
args = parser.parse_args()


# Setup swinunet config
# I have no idea if this is actually needed but might as well
from config import get_config

args2 = argparse.ArgumentParser()
args2.cfg = "configs/swin_tiny_patch4_window7_224_lite.yaml"
args2.batch_size = args.batch_size
args2.cache_mode = "no"

config = get_config(args2)


# Define Dataset
class NiftiDataset(Dataset):
    def __init__(self, slices_dir, clip=None, transform=None):
        self.slices_dir = slices_dir
        self.transform = transform
        self.filenames = [f for f in os.listdir(slices_dir) if f.endswith(".npz")]
        if clip > 0:
            self.filenames = self.filenames[:clip]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        slice_path = os.path.join(self.slices_dir, self.filenames[idx])

        # Load image
        data = np.load(slice_path)
        image, label = data["image"], data["label"]
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        sample = {"image": image, "label": label}
        if self.transform:
            sample = self.transform(sample)
        sample["case_name"] = self.filenames[idx].strip("\n")
        return sample


train_dir = args.train_dir
val_dir = args.val_dir
snapshot_dir = args.snapshot_dir
base_lr = args.base_lr
max_epoch = args.max_epoch
patience = args.patience

# Initialize dataloaders
db_train = NiftiDataset(train_dir, args.clip)
db_val = NiftiDataset(val_dir, args.clip)
print("The length of train set is: {}".format(len(db_train)))
print("The length of val set is: {}".format(len(db_val)))

trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(db_val, batch_size=args.batch_size, shuffle=True)


# Initialize model
model = SwinUnet(config, img_size=224, num_classes=2)
model.load_from(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))

model = nn.DataParallel(model)
model.to(device)

ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(2)
optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)


# Make snapshots dir
if not os.path.exists(snapshot_dir):
    os.makedirs(snapshot_dir)


max_iterations = max_epoch * len(trainloader)
iter_num = 0
best_performance = 0.0
overall_start = timer()
history = []
best_val_metric = float("inf")
wait = 0
best_model = None
# For each epoch
iterator = tqdm(range(max_epoch), ncols=100)
for epoch_num in iterator:
    start = timer()
    history_ce = 0.0
    history_dice = 0.0
    history_loss = 0.0

    # Training loop
    model.train()
    for i_batch, sampled_batch in enumerate(trainloader):
        image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
        image_batch, label_batch = image_batch.to(device), label_batch.to(device)

        outputs = model(image_batch)

        loss_ce = ce_loss(outputs, label_batch.squeeze(1).long())
        loss_dice = dice_loss(outputs, label_batch.squeeze(1), softmax=True)
        loss = 0.4 * loss_ce + 0.6 * loss_dice

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_

        history_ce += loss_ce.item()
        history_dice += loss_dice.item()
        history_loss += loss.item()
        iter_num += 1
        # Track training progress
        print(
            f"Epoch: {epoch_num}\t{100 * (i_batch + 1) / len(trainloader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.",
            end="\r",
        )

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # No need to track gradients during validation
        for i_batch, sampled_batch in enumerate(valloader):
            image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)

            outputs = model(image_batch)

            loss_dice = dice_loss(outputs, label_batch.squeeze(1), softmax=True)

            val_loss += loss_dice.item()

    val_loss /= len(valloader)
    history_ce /= len(trainloader)
    history_dice /= len(trainloader)
    history_loss /= len(trainloader)
    history.append([history_ce, history_dice, history_loss, val_loss])

    # Early Stopping Check
    if val_loss < best_val_metric:
        best_val_metric = val_loss
        best_model = deepcopy(
            model.state_dict()
        )  # Save a copy of the current best model
        wait = 0  # Reset wait counter
        print(f"Validation loss improved to {val_loss:.4f}. Saving model...")
    else:
        wait += 1

    if wait >= patience:
        print("Stopping early due to lack of improvement in validation loss.")
        # Save
        save_history_path = os.path.join(
            snapshot_dir, "epoch_" + str(epoch_num) + "_history.npz"
        )
        np.savez_compressed(save_history_path, history=history)
        save_model_path = os.path.join(snapshot_dir, "epoch_" + str(epoch_num) + ".pth")
        torch.save(model.state_dict(), save_model_path)
        best_model_path = os.path.join(
            snapshot_dir, "epoch_" + str(epoch_num) + "_best.pth"
        )
        torch.save(best_model, best_model_path)
        print(f"Saved model to {save_model_path}, {best_model_path}")
        break

    # Save occasionally
    if epoch_num + 1 % 50 == 0:
        save_history_path = os.path.join(
            snapshot_dir, "epoch_" + str(epoch_num) + "_history.npz"
        )
        np.savez_compressed(save_history_path, history=history)
        save_model_path = os.path.join(snapshot_dir, "epoch_" + str(epoch_num) + ".pth")
        torch.save(model.state_dict(), save_model_path)
        print(f"Saved model to {save_model_path}")
iterator.close()

total_time = timer() - overall_start
print(
    f"{total_time:.2f} total seconds elapsed. {total_time / (iter_num+1):.2f} seconds per epoch."
)
