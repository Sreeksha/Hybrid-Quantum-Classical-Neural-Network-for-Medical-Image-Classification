import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import (
    TRAIN_CSV, VALID_CSV, DATA_DIR, CSV_PATH_PREFIX,
    TARGET_LABEL, NUM_POS, NUM_NEG, RANDOM_SEED,
    IMG_SIZE, IMG_MEAN, IMG_STD, BATCH_SIZE, TRAIN_SPLIT
)

class CheXpertDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = row['img_path']
        label    = torch.tensor(row[TARGET_LABEL], dtype=torch.float32)
        image    = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def load_and_filter_csv(csv_path, num_pos, num_neg, seed=RANDOM_SEED):
    df = pd.read_csv(csv_path)
    df['img_path'] = df['Path'].apply(
        lambda p: os.path.join(DATA_DIR, p.replace(CSV_PATH_PREFIX, ''))
    )
    df = df[df[TARGET_LABEL].isin([1.0, 0.0])].copy()
    pos_df = df[df[TARGET_LABEL] == 1.0]
    neg_df = df[df[TARGET_LABEL] == 0.0]
    print(f"[Dataset] Available positives : {len(pos_df)}")
    print(f"[Dataset] Available negatives : {len(neg_df)}")
    pos_sample = pos_df.sample(n=min(num_pos, len(pos_df)), random_state=seed)
    neg_sample = neg_df.sample(n=min(num_neg, len(neg_df)), random_state=seed)
    balanced_df = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=seed)
    print(f"[Dataset] Total samples       : {len(balanced_df)}")
    return balanced_df

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ])

def get_dataloaders(seed=RANDOM_SEED):
    df      = load_and_filter_csv(TRAIN_CSV, NUM_POS, NUM_NEG, seed)
    n_train = int(len(df) * TRAIN_SPLIT)
    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train:]
    train_dataset = CheXpertDataset(train_df, transform=get_transforms(True))
    val_dataset   = CheXpertDataset(val_df,   transform=get_transforms(False))
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"[DataLoader] Train batches : {len(train_loader)}")
    print(f"[DataLoader] Val batches   : {len(val_loader)}")
    return train_loader, val_loader

def get_test_loader():
    df = pd.read_csv(VALID_CSV)
    df['img_path'] = df['Path'].apply(
        lambda p: os.path.join(DATA_DIR, p.replace(CSV_PATH_PREFIX, ''))
    )
    df = df[df[TARGET_LABEL].isin([1.0, 0.0])].copy()
    test_dataset = CheXpertDataset(df, transform=get_transforms(False))
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"[Test Loader] Test batches : {len(test_loader)}")
    return test_loader
