import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import config


# ======================================================
# 🔹 IMAGE FOLDERS (HAM10000 structure)
# ======================================================

IMAGE_DIRS = [
    "data/archive/HAM10000_images_part_1",
    "data/archive/HAM10000_images_part_2"
]


# ======================================================
# 🔹 DATASET CLASS
# ======================================================

class HAMDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img_id = self.df.loc[idx, "image_id"]
        label = self.df.loc[idx, "label"]

        img_path = None

        # 🔎 Search image in both folders
        for folder in IMAGE_DIRS:
            candidate = os.path.join(folder, img_id + ".jpg")
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            raise FileNotFoundError(f"Image not found in both folders: {img_id}")

        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label


# ======================================================
# 🔹 DATA SPLITTING FUNCTION
# ======================================================

def get_data_splits(reduced_size=2000):

    df = pd.read_csv(config.CSV_PATH)

    print(f"Original dataset size: {len(df)}")

    # 🔥 Reduce dataset (STRATIFIED)
    if reduced_size is not None:
        df, _ = train_test_split(
            df,
            train_size=reduced_size,
            stratify=df["dx"],
            random_state=42
        )
        print(f"Reduced dataset size: {len(df)}")

    # Stable class ordering
    classes = sorted(df["dx"].unique())
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    df["label"] = df["dx"].map(class_to_idx)

    # Train / Val / Test split (70 / 15 / 15)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df["label"],
        random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=42
    )

    print("\nSplit Sizes:")
    print(f"Train: {len(train_df)}")
    print(f"Val  : {len(val_df)}")
    print(f"Test : {len(test_df)}")

    return train_df, val_df, test_df, classes


# ======================================================
# 🔹 TRANSFORMS
# ======================================================

# 🔵 BASELINE (NO AUGMENTATION)
def get_baseline_transform():
    return A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])


# 🟢 AUGMENTED (for robustness experiment)
def get_augmented_transform():
    return A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])