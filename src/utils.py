# Metrics and Utility Functions

import numpy as np
import matplotlib.pyplot as plt
import torch
import globals as G


def visualize_sample(image_tensor, mask_tensor, class_names=None):
    """
    image_tensor: [3, H, W]
    mask_tensor: [H, W] with classes integer values
    """
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    mask_np = mask_tensor.cpu().numpy()  # [H, W]
    colors = plt.cm.get_cmap('tab20', 20) # Simple Colormap for 20 classes

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title("RGB Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(colors(mask_np))
    plt.title("Segmentation Mask")
    plt.axis("off")

    # Textual Legend for classes
    unique_classes = torch.unique(mask_tensor).tolist()
    class_labels = [G.CITYSCAPES_CLASSES.get(c, f"Class {c}") for c in unique_classes]
    legend_str = ", ".join([f"{c}: {label}" for c, label in zip(unique_classes, class_labels)])
    plt.figtext(0.5, 0.01, f"Classes in mask → {legend_str}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.show()


def compute_mIoU(preds, targets, num_classes):
    """
    Computes mean IoU on a batch.
    preds: tensor [B, H, W] (predicted integer values)
    targets: tensor [B, H, W] (GT values)
    num_classes: total number of classes (included 'unknown')
    """
    ious = []
    preds = preds.view(-1)
    targets = targets.view(-1)

    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = targets == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))  # avoids dividing by zero
        else:
            ious.append(intersection / union)

    # returns both mean and classes values
    return np.nanmean(ious), ious


def visualize_predictions(image, pred_mask, true_mask):
    """
    image: [3, H, W]
    pred_mask, true_mask: [H, W]
    """
    image_np = image.permute(1, 2, 0).cpu().numpy()
    pred_np = pred_mask.cpu().numpy()
    true_np = true_mask.cpu().numpy()

    cmap = plt.cm.get_cmap('tab20', 20)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cmap(pred_np))
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cmap(true_np))
    plt.title("Ground Truth")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
