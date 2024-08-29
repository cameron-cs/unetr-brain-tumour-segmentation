import argparse
import os
import yaml
from dataclasses import dataclass, fields
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split

from conf.unetr_conf import UNETR2DConfig
from dataset.brain_tumour_dataset import BrainTumorDataset
from model.unetr2d import UNETR2D


def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true = y_true.contiguous().view(-1)
    y_pred = y_pred.contiguous().view(-1)

    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum()

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


# parse YAML into a dictionary
def parse_yaml_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict

# convert a dictionary to a dataclass instance
def dict_to_dataclass(cls, dict_obj):
    field_set = {f.name for f in fields(cls) if f.init}
    filtered_dict = {k: v for k, v in dict_obj.items() if k in field_set}
    return cls(**filtered_dict)

# argparse for command-line interface
def get_args():
    parser = argparse.ArgumentParser(description="UNETR2D Model Configuration")
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file.")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    config_dict = parse_yaml_config(args.config)
    config = dict_to_dataclass(UNETR2DConfig, config_dict)

    path = config.dataset_path

    # all image and mask files
    image_paths = sorted(os.listdir(f'{path}/images'))
    mask_paths = sorted(os.listdir(f'{path}/masks'))

    # splitting dataset into train, validation, and test sets
    train_images, val_images, train_masks, val_masks = train_test_split(image_paths, mask_paths, test_size=0.2,
                                                                        random_state=42)
    val_images, test_images, val_masks, test_masks = train_test_split(val_images, val_masks, test_size=0.5,
                                                                      random_state=42)
    # transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # datasets
    train_dataset = BrainTumorDataset(image_dir=f'{path}/images', mask_dir=f'{path}/masks', image_paths=train_images,
                                      mask_paths=train_masks, transform=transform)
    val_dataset = BrainTumorDataset(image_dir=f'{path}/images', mask_dir=f'{path}/masks', image_paths=val_images,
                                    mask_paths=val_masks, transform=transform)
    test_dataset = BrainTumorDataset(image_dir=f'{path}/images', mask_dir=f'{path}/masks', image_paths=test_images,
                                     mask_paths=test_masks, transform=transform)

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNETR2D(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        img_size=config.img_size,
        feature_size=config.feature_size,
        hidden_size=config.hidden_size,
        mlp_dim=config.mlp_dim,
        num_heads=config.num_heads,
        pos_embed=config.pos_embed,
        norm_name=config.norm_name,
        conv_block=config.conv_block,
        res_block=config.res_block,
        dropout_rate=config.dropout_rate
    ).to(device)

    # loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = config.num_epochs
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        with tqdm(train_loader, unit="batch") as tepoch:
            for images, masks in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")

                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, masks)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                tepoch.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.model_save_path)
            print(f"Model saved at epoch {epoch + 1}")


