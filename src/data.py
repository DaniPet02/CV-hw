import os
import numpy as np
import cv2 as cv
import torch
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import globals as G  # contains CITYSCAPES_PATH, BATCH_SIZE, ecc.
from utils import compute_mIoU, visualize_predictions
from network import DeepLabV3Plus

class CityscapesDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        super().__init__()
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.image_paths[idx]).convert('RGB')) # 3 channels
        mask = np.array(Image.open(self.mask_paths[idx]).convert('L'))  # single channel

        # Maps out-of-class values like 'unknown' (es. 255)
        mask[mask >= G.NUM_CLASSES] = G.NUM_CLASSES  # unknown class

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        return img, mask.long()

def get_dataloaders():
    train_img_paths = sorted(glob(os.path.join(G.CITYSCAPES_PATH, 'train/img/*.png')))
    train_mask_paths = sorted(glob(os.path.join(G.CITYSCAPES_PATH, 'train/label/*.png')))
    val_img_paths = sorted(glob(os.path.join(G.CITYSCAPES_PATH, 'val/img/*.png')))
    val_mask_paths = sorted(glob(os.path.join(G.CITYSCAPES_PATH, 'val/label/*.png')))

    # transform = T.Compose([T.Resize((512, 1024)), T.ToTensor()])
    transform = A.Compose([
        A.Resize(256, 512),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_ds = CityscapesDataset(train_img_paths, train_mask_paths, transform)
    val_ds = CityscapesDataset(val_img_paths, val_mask_paths, transform)

    train_loader = DataLoader(train_ds, batch_size=G.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=G.BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()
    # for imgs, masks in train_loader:
    #     print("Image min/max:", imgs[0].min().item(), imgs[0].max().item())
    #     print("Image batch shape:", imgs.shape)
    #     print("Mask batch shape:", masks.shape)
    #     print("Unique mask values:", masks[0].unique())
    #     break
    model = DeepLabV3Plus().to(G.DEVICE)
    model.eval()

    with torch.no_grad():
        imgs, masks = next(iter(train_loader))
        imgs = imgs.to(G.DEVICE)
        masks = masks.to(G.DEVICE)

        outputs = model(imgs)
        preds = outputs.argmax(dim=1)  # predizioni [B, H, W]

        # 👇 STAMPA QUI
        print("Pred unique:", preds[0].unique())
        print("GT unique:  ", masks[0].unique())

        miou, ious = compute_mIoU(preds, masks, G.NUM_CLASSES + 1)
        print(f"Mean IoU: {miou:.3f}, IoUs per class: {ious}")
        visualize_predictions(imgs[0].cpu(), preds[0].cpu(), masks[0].cpu())