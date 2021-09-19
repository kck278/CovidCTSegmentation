import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class CovidDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, num_classes: int, extended: bool=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.num_classes = num_classes
        self.extended = extended
        self.transforms = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1), 
                transforms.ToTensor()
            ]
        )
        self.augmentations = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(0, shear=(10, 10, 10, 10))
            ]
        )
        self.mask_transforms = transforms.Compose(
            [
                transforms.Lambda(self.multiply)
            ]
        )
        self.image_names, self.mask_names = self.make_dataset()
        

    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        mask_name = self.mask_names[index]

        image = default_loader(os.path.join(self.images_dir, image_name))
        mask = default_loader(os.path.join(self.masks_dir, mask_name))
        assert image.size == mask.size

        image = self.transforms(image)
        mask = self.transforms(mask)

        stacked = torch.cat([image, mask], dim=0)
        stacked = self.augmentations(stacked)
        image, mask = torch.chunk(stacked, chunks=2, dim=0)

        mask = self.mask_transforms(mask)
        mask = torch.squeeze(mask)
        return image, mask


    def make_dataset_extended(self):
        image_names = []
        mask_names = []

        for root, _, files in os.walk(self.images_dir):
            for name in files:
                img_name = os.path.join(root, name)
                img_name = os.path.relpath(img_name, self.images_dir)
                image_names.append(img_name)

        for root, _, files in os.walk(self.masks_dir):
            for name in files:
                mask_name = os.path.join(root, name)
                mask_name = os.path.relpath(mask_name, self.masks_dir)
                mask_names.append(mask_name)

        return sorted(image_names), sorted(mask_names)
    

    def make_dataset_small(self):
        image_names = sorted(
            [f for f in os.listdir(self.images_dir) if os.path.isfile(os.path.join(self.images_dir, f))]
        )
        mask_names = sorted(
            [f for f in os.listdir(self.masks_dir) if os.path.isfile(os.path.join(self.masks_dir, f))]
        )
        return image_names, mask_names


    def make_dataset(self):
        if self.extended:
            image_names, mask_names = self.make_dataset_extended()
        else:
            image_names, mask_names = self.make_dataset_small()

        assert len(image_names) == len(mask_names)
        return image_names, mask_names


    def multiply(self, tensor):
        return torch.mul(tensor, self.num_classes - 1)
