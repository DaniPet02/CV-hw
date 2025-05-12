# Metrics and Utility Functions

import numpy as np
import matplotlib.pyplot as plt
import torch
import globals as G


def visualize_sample(image_tensor, mask_tensor, class_names=None):
    """
    image_tensor: [3, H, W]
    mask_tensor: [H, W] with class indices
    """
    # De-normalizes image before showing it
    image_np = denormalize(image_tensor).permute(1, 2, 0).cpu().numpy()
    image_np = np.clip(image_np, 0, 1)  # clip for out-of-range RGB values

    mask_np = mask_tensor.cpu().numpy()
    colors = plt.cm.get_cmap('tab20', 20)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title("RGB Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(colors(mask_np))
    plt.title("Segmentation Mask")
    plt.axis("off")

    # Legend of classes
    unique_classes = torch.unique(mask_tensor).tolist()
    class_labels = [G.CITYSCAPES_CLASSES.get(c, f"Class {c}") for c in unique_classes]
    legend_str = ", ".join([f"{c}: {label}" for c, label in zip(unique_classes, class_labels)])
    plt.figtext(0.5, 0.01, f"Classes in mask → {legend_str}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.show()


def denormalize(image_tensor):
    """
    De-normalizes the image tensor.
    image_tensor: [3, H, W] (normalized)
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image_tensor.device)
    return image_tensor * std + mean


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

    cmap = plt.get_cmap('tab20', 20)

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


def visualize_metrics(metrics):
    """
    Visualizes the training and validation metrics over epochs.
    metrics: dict with keys 'train' and 'val', each containing a list of metric values
    """
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['train'], label="Training mIoU", color='blue')
    plt.plot(metrics['val'], label="Validation mIoU", color='orange')
    plt.title("mIoU Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("mIoU")
    plt.legend()
    plt.grid()
    plt.show()


def visualize_loss(losses):
    """
    Visualizes the training loss over epochs.
    losses: list of loss values
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training Loss", color='blue')
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Example usage of the utility functions
    # Assuming you have a sample image and mask tensors
    image_tensor = torch.randn(3, 512, 1024)  # Example image tensor
    mask_tensor = torch.randint(0, G.NUM_CLASSES, (512, 1024))  # Example mask tensor
    visualize_sample(image_tensor, mask_tensor)
    # Compute mIoU
    mIoU, ious = compute_mIoU(mask_tensor.unsqueeze(0), mask_tensor.unsqueeze(0), G.NUM_CLASSES)
    print(f"Mean IoU: {mIoU}, IoUs per class: {ious}")
    