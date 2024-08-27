import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class BrainTumorDataset(Dataset):
    def __init__(self, images, labels):
        self.X = images
        self.y = labels

        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        self.rotation_angles = [45, 90, 120, 180, 270, 300, 330]

        # transforms with rotations
        self.transforms = [self.base_transform] + [
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(angle),
                transforms.ToTensor()
            ]) for angle in self.rotation_angles
        ]

    def __len__(self):
        # length of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # transformations
        transformed_images = [transform(self.X[idx]) for transform in self.transforms]

        # one-hot encode the label
        labels = torch.zeros(4, dtype=torch.float32)
        labels[int(self.y[idx])] = 1.0

        # repeat the label for each transformed image
        labels_batch = [labels] * len(transformed_images)

        # batch of images and labels
        return torch.stack(labels_batch), torch.stack(transformed_images)
