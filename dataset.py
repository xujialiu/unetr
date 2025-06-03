from torch.utils import data
from torch.utils.data import Dataset, dataset
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from skimage import io, color
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch
from einops import rearrange

MEAN, STD = (
    (0.423737496137619, 0.2609460651874542, 0.128403902053833),
    (0.29482534527778625, 0.20167365670204163, 0.13668020069599152),
)


def get_weighted_sampler(dataset):
    sample_weights = dataset.sample_weights
    weighted_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return weighted_sampler


def build_transform(is_train, mean=MEAN, std=STD, img_size=224):

    if is_train == "train":

        transform = [
            # 几何变换
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
            ),
            # 仿射变换
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=30,
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.5,
            ),
            # 弹性变换（对医学图像特别有用）
            # A.ElasticTransform(
            #     alpha=120,
            #     sigma=120 * 0.05,
            #     alpha_affine=120 * 0.03,
            #     interpolation=cv2.INTER_CUBIC,
            #     border_mode=cv2.BORDER_CONSTANT,
            #     value=0,
            #     mask_value=0,
            #     p=0.3,
            # ),
            # 网格畸变
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.3,
            ),
            # 调整大小
            A.RandomResizedCrop(
                height=img_size,
                width=img_size,
                scale=(0.33, 1.0),
                ratio=(0.75, 1.3333),
                interpolation=cv2.INTER_CUBIC,
            ),
            # 色彩增强（仅应用于图像，不影响mask）
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=1
                    ),
                    A.RandomGamma(gamma_limit=(80, 120), p=1),
                ],
                p=0.5,
            ),
            # 噪声
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                    A.GaussianBlur(blur_limit=(3, 7), p=1),
                    A.MedianBlur(blur_limit=5, p=1),
                ],
                p=0.3,
            ),
            # 归一化
            A.Normalize(mean=mean, std=std),
            # 转换为PyTorch张量
            ToTensorV2(),
        ]
        # transform = [
        #     A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_CUBIC),
        #     A.Normalize(mean=mean, std=std),
        #     ToTensorV2(),
        # ]
    else:
        transform = [
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]

    return A.Compose(transform)


def build_transform_2(is_train, mean=MEAN, std=STD, img_size=224):

    if is_train == "train":

        transform = [
            # 几何变换
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
            ),
            # 仿射变换
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=30,
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.5,
            ),
            # 弹性变换（对医学图像特别有用）
            # A.ElasticTransform(
            #     alpha=120,
            #     sigma=120 * 0.05,
            #     alpha_affine=120 * 0.03,
            #     interpolation=cv2.INTER_CUBIC,
            #     border_mode=cv2.BORDER_CONSTANT,
            #     value=0,
            #     mask_value=0,
            #     p=0.3,
            # ),
            # 网格畸变
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.3,
            ),
            # 调整大小
            A.RandomResizedCrop(
                height=img_size,
                width=img_size,
                scale=(1, 1.0),
                ratio=(0.75, 1.3333),
                interpolation=cv2.INTER_CUBIC,
            ),
            # 色彩增强（仅应用于图像，不影响mask）
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=1
                    ),
                    A.RandomGamma(gamma_limit=(80, 120), p=1),
                ],
                p=0.5,
            ),
            # 噪声
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                    A.GaussianBlur(blur_limit=(3, 7), p=1),
                    A.MedianBlur(blur_limit=5, p=1),
                ],
                p=0.3,
            ),
            # 归一化
            A.Normalize(mean=mean, std=std),
            # 转换为PyTorch张量
            ToTensorV2(),
        ]
        # transform = [
        #     A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_CUBIC),
        #     A.Normalize(mean=mean, std=std),
        #     ToTensorV2(),
        # ]
    else:
        transform = [
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]

    return A.Compose(transform)


class MyDataset(Dataset):
    def __init__(
        self,
        csv_path,
        data_path,
        mask_folder_names=["EX", "MA", "HE", "SE"],
        is_train="train",
        transform=None,
    ):
        self.csv_path = csv_path
        self.data_root_path = data_path
        self.transform = transform
        self.is_train = is_train

        # self.mask_folder_names = mask_folder_names
        self.path_root_img = Path(data_path) / "image"
        self.list_path_root_mask = [
            Path(data_path) / "label" / mask_folder_name
            for mask_folder_name in mask_folder_names
        ]
        self.df = pd.read_csv(csv_path)
        self.df = self.df.loc[self.df["train_test"] == is_train].reset_index()

        self.class_counts = self.df.label.value_counts().sort_index().values
        self.class_weights = 1.0 / torch.tensor(self.class_counts, dtype=torch.float)
        # self.sample_weights = [self.class_weights[label] for label in self.df.label]
        self.sample_weights = [
            self.class_weights[label]
            for label in self.df.label.rank(method="dense").astype(int) - 1
        ]

    def __getitem__(self, idx):

        file_name = self.df.loc[idx, "name"]
        label = self.df.loc[idx, "label"]

        img_path = self.path_root_img / file_name

        mask_paths = [mask_path / file_name for mask_path in self.list_path_root_mask]

        file_stem = img_path.stem

        # img = io.imread(img_path)
        # list_mask = [io.imread(mask_path, as_gray=True) > 0 for mask_path in mask_paths]

        # Replace io.imread with cv2 for faster loading
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # For masks
        list_mask = []
        for mask_path in mask_paths:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            list_mask.append(mask > 0)

        mask = np.stack(list_mask, axis=2).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # img.shape: (B, c, 224, 224), mask.shape: (B, 224, 224, 4)
        mask = rearrange(mask, "h w c -> c h w")

        return img, mask, label, file_stem

    def __len__(self):
        return len(self.df)


if __name__ == "__main__":
    csv_path = "/data_A/xujialiu/datasets/DDR_seg/250527_1_get_label/ddr_seg_cls.csv"
    data_path = "/data_A/xujialiu/datasets/DDR_seg/preprocess"

    transform = build_transform(is_train="train")
    dataset = MyDataset(csv_path, data_path, transform=transform)

    dataset[1]
