import torch
import torchvision
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Pixel statistics of all (train + test) CIFAR-10 images
# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
AVG = (0.4914, 0.4822, 0.4465) # Mean
STD = (0.2023, 0.1994, 0.2010) # Standard deviation
# https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py#L55
# AVG = (125.31, 122.95, 113.87) # Mean
# STD = (62.99, 62.09, 66.70) # Standard deviation
# # https://github.com/darshanvjani/torchcraft/blob/main/utils/helper.py#L55
# AVG = (0.49139968, 0.48215841, 0.44653091) # Mean
# STD = (0.24703223, 0.24348513, 0.26158784) # Standard deviation
CHW = (3, 32, 32) # Channel, height, width
CLASSES = [ # Class labels (list index = class value)
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]


class TransformedDataset(torch.utils.data.Dataset):
    """Pytorch dataset + custom data augmentation (image transformation)
    https://github.com/parrotletml/era_session_seven/blob/main/mnist/dataset.py
    https://albumentations.ai/docs/examples/migrating_from_torchvision_to_albumentations/
    """
    def __init__(self,
        dataset:torchvision.datasets,
        transform:A.Compose|None=None,
    ) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx:int) -> tuple[torch.Tensor, int]:
        image, label = self.dataset[idx]
        if self.transform: image = self.transform(image=np.array(image))['image']
        return image, label


def get_transform(
    padding:int=40, # size after padding before cropping (unit: pixels)
    crop:int=32, # size after cropping (unit: pixels)
    cutout:int=16, # size of cutout box (unit: pixels)
) -> dict[str, A.Compose]:
    """Create image transformation pipeline for training and test datasets.
    https://github.com/kuangliu/pytorch-cifar/blob/master/main.py#L30-L34
    https://github.com/davidcpage/cifar10-fast/blob/master/core.py#L98
    """
    return {
        'train': A.Compose([
            A.Normalize(mean=AVG, std=STD, always_apply=True), # Cutout boxes should be grey, not black
            A.PadIfNeeded(min_height=padding, min_width=padding, always_apply=True), # Pad before cropping to achieve translation
            A.RandomCrop(height=crop, width=crop, always_apply=True),
            # A.HorizontalFlip(),
            A.CoarseDropout( # Cutout
                max_holes=1, max_height=cutout, max_width=cutout,
                min_holes=1, min_height=cutout, min_width=cutout,
                fill_value=AVG
            ),
            ToTensorV2(),
        ]),
        'test': A.Compose([
            A.Normalize(mean=AVG, std=STD, always_apply=True),
            ToTensorV2(),
        ]),
    }


def get_dataset(
    transform:dict[str, A.Compose],
) -> dict[str, TransformedDataset]:
    "Create training and test datasets and apply transforms."
    return {
        'train': TransformedDataset(
            dataset=torchvision.datasets.CIFAR10('../data', train=True, download=True),
            transform=transform['train'],
        ),
        'test': TransformedDataset(
            dataset=torchvision.datasets.CIFAR10('../data', train=False, download=False),
            transform=transform['test'],
        ),
    }


def get_dataloader(
    dataset:TransformedDataset,
    params:dict[str, bool|int],
) -> dict[str, torch.utils.data.DataLoader]:
    "Create training and test dataloader."
    return {
        'train': torch.utils.data.DataLoader(dataset['train'], **params),
        'test': torch.utils.data.DataLoader(dataset['test'], **params),
    }