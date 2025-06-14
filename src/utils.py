#import sklearn
import os
from PIL import Image
import torch
import torch.nn as nn
from tqdm import tqdm
import cv2
import numpy as np
import random
import shutil
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.segmentation import deeplabv3_resnet50
#import segmentation_models_pytorch as smp
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


##############
resized_height = 512
resized_width = 1024

transform = T.Compose([
    T.Resize((resized_height, resized_width)),  # Resize to half the original size
    T.ToTensor(),  # converts in [0, 1], shape [3, H, W]
    T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
])

#############


############################################# DATASETS #############################################
class CityscapesTrainEvalDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=transform):
        self.transform = transform 
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)

        # Collect all image paths
        self.img_paths = list(self.img_dir.rglob("*.png"))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        # Derive corresponding mask path by adding '_m' before the extension
        mask_name = img_path.stem + "_m.png"
        mask_path = self.mask_dir / mask_name

        if not mask_path.exists():
            print(f"Warning: No mask found for image: {img_path.name}")

        # Load and preprocess image
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])(img)

        # Load and preprocess mask
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale (single channel)

        # Resize mask with nearest neighbor interpolation
        resized_mask = mask.resize((resized_width, resized_height), resample=Image.NEAREST)

        mask_np = np.array(resized_mask, dtype=np.uint8)

        mask_tensor = torch.as_tensor(mask_np, dtype=torch.uint8)

        mask_onehot = convert_label_to_multilabel_one_hot(mask_tensor, "cityscapes")

        return img, mask_onehot, mask_np
    
class CityscapesTestDataset(Dataset):
    def __init__(self, img_dir, transform=transform):
        self.img_paths = sorted(Path(img_dir).rglob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        else:
            img = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])(img)

        return img, str(img_path.name)  # Return the filename for later use

class LostAndFoundTrainEvalDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=transform):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.img_paths = sorted(self.img_dir.rglob("*.png"))[:1000] # Limit to 1000 images
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_name = img_path.stem + "_m.png"
        mask_path = self.mask_dir / mask_name

        if not mask_path.exists():
            print(f"Warning: No mask found for image: {img_path.name}")

        # Load image
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])(img)

        # Load and resize mask
        mask = Image.open(mask_path).convert("L")  # grayscale
        resized_mask = mask.resize((resized_width, resized_height), resample=Image.NEAREST)
        mask_np = np.array(resized_mask, dtype=np.uint8)
        mask_tensor = torch.as_tensor(mask_np, dtype=torch.uint8)

        mask_onehot = convert_label_to_multilabel_one_hot(mask_tensor, "lostandfound")

        return img, mask_onehot, mask_np  # [3,H,W], [8,H,W], [H,W]

def fix_cityscapes(path_in, path_out, image_folder_in="leftImg8bit", mask_folder_in="gtFine", is_delete=False): # is_delete=True if you want to delete the city folders after copying
    """
    Fixes the CityScapes dataset by renaming the files and removing the city folders.
    """
    splits = ['train', 'val', 'test']
   
    for split in splits:
        count = 1
        img_dir = os.path.join(path_in, image_folder_in, split)
        mask_dir = os.path.join(path_in, mask_folder_in, split)

        # New destination directories
        img_out = os.path.join(path_out, 'img', split)
        mask_out = os.path.join(path_out, 'mask', split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(mask_out, exist_ok=True)

        # Iterate on sub-folders
        for city in os.listdir(img_dir):
            city_img_dir = os.path.join(img_dir, city)
            city_mask_dir = os.path.join(mask_dir, city)

            if not os.path.isdir(city_img_dir):
                continue  # Skips non-directory files

            for filename in os.listdir(city_img_dir):
                if not filename.endswith('leftImg8bit.png'):
                    continue
                img_path = os.path.join(city_img_dir, filename)
                base_prefix = filename.replace('_leftImg8bit.png', '')

                # Renames and copies RGB
                new_base = f"{split}{count}"
                ext = '.png'
                new_img_name = f"{new_base}{ext}"
                shutil.copy(img_path, os.path.join(img_out, new_img_name))

                if split != 'test': # For test split, we only copy the image
                    # Renames and copies all 'label' associated files
                    suffixes = ['_gtFine_labelIds.png', '_gtFine_color.png', '_gtFine_instanceIds.png', '_gtFine_polygons.json']
                    for suffix in suffixes:
                        if suffix == '_gtFine_labelIds.png':
                            original_name = base_prefix + suffix
                            source = os.path.join(city_mask_dir, original_name)
                            if os.path.exists(source):
                                new_name = f"{new_base}_m.png"  # es. train1_m.png
                                shutil.copy(source, os.path.join(mask_out, new_name))
                        else:
                            continue
                count += 1
                    
            if is_delete:        
                # Cleans city folders if empty
                if os.path.isdir(city_img_dir) and not os.listdir(city_img_dir):
                    os.rmdir(city_img_dir)
                if os.path.isdir(city_mask_dir) and not os.listdir(city_mask_dir):
                    os.rmdir(city_mask_dir)

                # Removes split folders if empty
                for d in [img_dir, mask_dir]:
                    if os.path.isdir(d) and not os.listdir(d):
                        os.rmdir(d)

def fix_lostandfound(path_in, path_out, image_folder_in="leftImg8bit", mask_folder_in="gtCoarse", is_delete=False): # is_delete=True if you want to delete the city folders after copying
    """
    Fixes the LostAndFound dataset by renaming the files and removing the city folders.
    """
    splits = ['train', 'test']
   
    for split in splits:
        count = 1
        img_dir = os.path.join(path_in, image_folder_in, split)
        mask_dir = os.path.join(path_in, mask_folder_in, split)

        # New destination directories
        img_out = os.path.join(path_out, 'img', split)
        mask_out = os.path.join(path_out, 'mask', split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(mask_out, exist_ok=True)

        # Iterate on sub-folders
        for city in os.listdir(img_dir):
            city_img_dir = os.path.join(img_dir, city)
            city_mask_dir = os.path.join(mask_dir, city)

            if not os.path.isdir(city_img_dir):
                continue  # Skips non-directory files

            for filename in os.listdir(city_img_dir):
                if not filename.endswith('leftImg8bit.png'):
                    continue
                img_path = os.path.join(city_img_dir, filename)
                base_prefix = filename.replace('_leftImg8bit.png', '')

                # Renames and copies RGB
                new_base = f"{split}{count}"
                ext = '.png'
                new_img_name = f"{new_base}{ext}"
                shutil.copy(img_path, os.path.join(img_out, new_img_name))

                
                # Renames and copies all 'label' associated files
                suffixes = ['_gtCoarse_labelIds.png', '_gtCoarse_color.png', '_gtCoarse_instanceIds.png', '_gtCoarse_labelTrainIds.png', '_gtCoarse_polygons.json']
                for suffix in suffixes:
                    if suffix == '_gtCoarse_labelIds.png':
                        original_name = base_prefix + suffix
                        source = os.path.join(city_mask_dir, original_name)
                        if os.path.exists(source):
                            new_name = f"{new_base}_m.png"  # es. train1_m.png
                            shutil.copy(source, os.path.join(mask_out, new_name))
                    else:
                        continue
                count += 1
                    
            if is_delete:        
                # Cleans city folders if empty
                if os.path.isdir(city_img_dir) and not os.listdir(city_img_dir):
                    os.rmdir(city_img_dir)
                if os.path.isdir(city_mask_dir) and not os.listdir(city_mask_dir):
                    os.rmdir(city_mask_dir)

                # Removes split folders if empty
                for d in [img_dir, mask_dir]:
                    if os.path.isdir(d) and not os.listdir(d):
                        os.rmdir(d)

def convert_label_to_multilabel_one_hot(label, dataset, CLASS_MAPPING, MACRO_CLASSES):
    """
    Converts 2D label mask [H, W] with Cityscapes original IDs into a multi-label one-hot encoding tensor [8, H, W].
    The last channel (index 7) corresponds to the 'object' auxiliary channel.
    """

    LABEL_TO_MACRO_IDX = {}

    for original_id, (macro_class, is_object) in CLASS_MAPPING.items():
        if macro_class is not None:
            LABEL_TO_MACRO_IDX[original_id] = MACRO_CLASSES[macro_class]
        else:
            # For None macro class, we don't assign a macro_idx (only object channel will be set)
            LABEL_TO_MACRO_IDX[original_id] = None

    # Get the spatial dimensions of the label mask
    height, width = label.shape

    # Initialize the output tensor with 8 channels (macro-classes), filled with zeros.
    # Each channel corresponds to a macro-class.
    multilabel = torch.zeros((8, height, width), dtype=torch.float32)

    if dataset == "cityscapes":
        # Iterate over each original class ID defined in the CLASS_MAPPING
        for original_id, (_, is_object) in CLASS_MAPPING.items():

            # Create a boolean mask of where the input label equals the current original class ID
            mask = (label == original_id)

            # Look up the macro-class index for this class ID, or None if it doesn't belong to any
            macro_idx = LABEL_TO_MACRO_IDX[original_id]

            # If this class maps to a macro-class, set 1 at those pixel locations in the corresponding channel
            if macro_idx is not None:
                multilabel[macro_idx][mask] = 1.0  # Set the macro-class channel to 1 where the mask is True

            # If this class is considered an 'object', also set the 'object' channel (index 6) to 1
            if is_object:
                multilabel[MACRO_CLASSES["object"]][mask] = 1.0  # Set the object class channel to 1
    elif dataset == "lostandfound":
        height, width = label.shape
        multilabel = torch.zeros((8, height, width), dtype=torch.float32)

        road_mask = (label == 1)
        object_mask_1 = (label != 1)
        object_mask_2 = (label != 0)
        object_mask = object_mask_1 == object_mask_2

        multilabel[MACRO_CLASSES["road"]][road_mask] = 1.0
        multilabel[MACRO_CLASSES["object"]][object_mask] = 1.0
    else:
        print("you have to choose a dataset between cityscapes and lostandfound\n ")

    # Return the resulting multi-label one-hot tensor of shape [8, H, W]
    return multilabel


############################################# BOUNDARY #############################################
def get_boundary_mask(label_mask, kernel_size=3, iterations=2):
    """
    Computes the boundary mask from a label mask using morphological operations.
    Args:
        label_mask (numpy.ndarray): Input label mask with shape [H, W].
        kernel_size (int): Size of the kernel for morphological operations.
    Returns:
        numpy.ndarray: Boundary mask with shape [H, W].
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(label_mask, kernel, iterations=iterations)
    eroded = cv2.erode(label_mask, kernel, iterations=iterations)
    boundary = (dilated != eroded).astype(np.uint8)
    return boundary

def get_boundary_mask_batch(label_masks, kernel_size=3, iterations=2):
    """
    Computes the boundary mask for a batch of label masks using morphological operations.
    Args:
        label_masks (torch.Tensor): Tensor of shape [B, H, W] or [B, 1, H, W], with binary masks (0 or 1).
        kernel_size (int): Size of the kernel for morphological operations.
        iterations (int): Number of times to apply dilation and erosion.
    Returns:
        torch.Tensor: Boundary masks of shape [B, H, W], dtype=torch.uint8.
    """

    if label_masks.dim() == 3:
        label_masks = label_masks.unsqueeze(1)  # [B, 1, H, W]

    # Ensure float type for convolution
    label_masks = label_masks.float()

    # Define kernel (morphological structuring element)
    device = label_masks.device
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device)

    padding = kernel_size // 2

    # Apply dilation
    dilated = label_masks
    for _ in range(iterations):
        dilated = F.conv2d(dilated, kernel, padding=padding)
        dilated = (dilated > 0).float()

    # Apply erosion
    eroded = label_masks
    for _ in range(iterations):
        eroded = F.conv2d(eroded, kernel, padding=padding)
        eroded = (eroded == kernel.numel()).float()

    # Compute boundary: difference between dilated and eroded regions
    boundary = (dilated != eroded).float()

    return boundary.squeeze(1).byte()  # Return shape [B, H, W], uint8


############################################# VISUALIZATION #############################################
def visualize_one_hot_vertical(one_hot, class_names=None, max_classes=8):
    """
    Visualizes the one-hot encoded masks vertically.
    """
    num_classes = min(one_hot.shape[0], max_classes)
    fig, axes = plt.subplots(num_classes, 1, figsize=(5, 3 * num_classes))

    for i in range(num_classes):
        ax = axes[i]
        ax.imshow(one_hot[i], cmap='gray')
        title = f"Class {i}" if class_names is None else class_names[i]
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_erosion_mask(label_mask):
    """
    Visualizes the erosion mask of a label mask.
    """
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(label_mask, kernel, iterations=2)
    plt.figure()
    plt.imshow(erosion, cmap='gray')
    plt.title("Erosion Mask")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_dilation_mask(label_mask):
    """
    Visualizes the dilation mask of a label mask.
    """
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(label_mask, kernel, iterations=2)
    plt.figure()
    plt.imshow(dilation, cmap='gray')
    plt.title("Dilation Mask")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_boundary_mask(label_mask, iterations=2):
    """
    Visualizes the boundary mask.
    """
    boundary = get_boundary_mask(label_mask, iterations=iterations)
    plt.figure()
    plt.imshow(boundary, cmap='gray')
    plt.title("Boundary Mask")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def uos_heatmap(img_tensor, uos_tensor, threshold=0.5, alpha_val=0.5):
    """
    Superpose the heatmap of the unknown objectness score to the image,
    showing only pixels above a given threshold and maintaining consistent brightness.
    
    img_tensor: [3, H, W], torch.Tensor in [0, 1] or [0, 255]
    uos_tensor: [H, W], torch.Tensor
    threshold: float, minimum normalized UOS value to be shown
    alpha_val: float, fixed transparency level for visible heatmap areas
    """
    # Convert image
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    # Normalize UOS
    uos = uos_tensor.cpu().numpy()
    uos = (uos - uos.min()) / (uos.max() - uos.min())  # Normalize to [0, 1]
    uos = np.where(uos > threshold, uos, 0.0)  # Apply threshold

    # Display
    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.imshow(uos, cmap='hot', alpha=alpha_val, vmin=0, vmax=1)  # Use 'hot' colormap for heatmap
    plt.title("Unknown Objectness Score Heatmap (Thresholded)")
    plt.colorbar(label="UOS score (0-1)")
    plt.axis('off')
    plt.show()

def visualize_uos_with_conformal(model, test_image, device, threshold):
    """
    Compute and visualize the UOS heatmap on the test image
    applying conformal threshold to detect unknown pixels.

    Args:
        model: Trained model.
        test_image: Single test image tensor (C, H, W).
        device: CUDA or CPU.
        threshold: UOS threshold from calibration.

    Returns:
        None. Displays heatmap overlay.
    """
    model.eval()
    with torch.no_grad():
        output = model(test_image.unsqueeze(0).to(device))  # shape: (1, num_classes, H, W)
        uos = unknown_objectness_score(output)[0].cpu().numpy()  # shape: (H, W)

    # Create binary mask of unknowns using conformal threshold
    unknown_mask = uos > threshold

    # Visualize heatmap of UOS and overlay unknown mask
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    # Convert tensor to numpy and transpose channels for plt (H, W, C)
    img_np = test_image.cpu().numpy().transpose(1, 2, 0)
    plt.imshow(img_np)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Unknown Objectness Score Heatmap")
    plt.imshow(uos, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar()
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Unknown Object Mask (Conformal Prediction)")
    plt.imshow(img_np)
    # Overlay unknown mask with transparency
    plt.imshow(unknown_mask, cmap='cool', alpha=0.5)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


############################################# NETWORK #############################################
class MultiLabelDeepLabV3(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        # Load pretrained model
        self.model = deeplabv3_resnet50(pretrained=True)

        # Replace classifier to output 8 channels with sigmoid
        self.model.classifier[-1] = nn.Conv2d(
            in_channels=256,
            out_channels=n_classes,
            kernel_size=1
        )
    
    def forward(self, x):
        x = self.model(x)['out']
        return torch.sigmoid(x)  # Apply sigmoid for multilabel outputs
    

############################################# LOSS FUNCTION #############################################
class BoundaryAwareBCELoss(nn.Module):
    def __init__(self, lambda_weight=3.0):
        super().__init__()
        self.lambda_weight = lambda_weight

    def forward(self, pred, target, boundary_mask):
        #to avoid log(0)
        eps = 1e-7

        #standard BCE loss
        bce = -(target * torch.log(pred + eps) + (1 - target) * torch.log(1 - pred + eps))
        normal_term = bce.mean()

        boundary_mask = boundary_mask.float()
        #expansion to (B, C, H, W) to do the multiplication

        if boundary_mask.dim() == 3:
            boundary_mask = boundary_mask.unsqueeze(1)
            
        boundary_mask = boundary_mask.expand(-1, pred.shape[1], -1, -1)

        #boundary aware BCE loss
        boundary_bce = bce * boundary_mask
        num_boundary_pixels = boundary_mask.sum(dim=(1, 2, 3)).clamp(min=1.0) #boundary pixels of each image
        boundary_loss = boundary_bce.sum(dim=(1, 2, 3)) / num_boundary_pixels
        boundary_term = boundary_loss.mean()

        return normal_term + self.lambda_weight * boundary_term
    
class BoundaryAwareBCELossFineTuning(nn.Module):
    def __init__(self, lambda_weight=3.0):
        super().__init__()
        self.lambda_weight = lambda_weight

    def forward(self, pred, target, boundary_mask):
        #to avoid log(0)
        eps = 1e-7

        #standard BCE loss
        bce = -(target * torch.log(pred + eps) + (1 - target) * torch.log(1 - pred + eps))

        # -------------------------------
        # Create a mask: pixels with at least one GT class > 0
        # Shape: (B, H, W)
        valid_pixel_mask = (target.sum(dim=1) > 0).float()

        # Expand to match shape (B, C, H, W)
        valid_pixel_mask = valid_pixel_mask.unsqueeze(1).expand_as(target)

        # Apply the pixel mask
        bce = bce * valid_pixel_mask
        num_valid_pixels = valid_pixel_mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
        normal_loss = bce.sum(dim=(1, 2, 3)) / num_valid_pixels
        normal_term = normal_loss.mean()
        # -------------------------------

        # Expand boundary mask if needed
        if boundary_mask.dim() == 3:
            boundary_mask = boundary_mask.unsqueeze(1)
        boundary_mask = boundary_mask.expand_as(target)

            
        boundary_bce = bce * boundary_mask
        num_boundary_pixels = (boundary_mask * valid_pixel_mask).sum(dim=(1, 2, 3)).clamp(min=1.0)
        boundary_loss = boundary_bce.sum(dim=(1, 2, 3)) / num_boundary_pixels
        boundary_term = boundary_loss.mean()

        return normal_term + self.lambda_weight * boundary_term
    

############################################# TEST #############################################
def evaluate_metrics(model, dataloader, device='cuda'):
    """
    Computes AP, FPR@95 and AUROC over the given validation dataloader.

    Args:
        model (torch.nn.Module): Trained PyTorch model for segmentation.
        dataloader (torch.utils.data.DataLoader): Dataloader yielding (images, masks).
        device (str): Device to run evaluation on.

    Returns:
        dict: Dictionary with 'AP', 'FPR95', and 'AUROC' scores.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            probs = model(images)  # Already sigmoid'd, shape: [B, 1, H, W]
            all_preds.append(probs.view(-1).cpu())
            all_targets.append(masks.view(-1).cpu())

    preds_flat = torch.cat(all_preds).numpy()
    targets_flat = torch.cat(all_targets).numpy()

    ap = average_precision_score(targets_flat, preds_flat)
    auroc = roc_auc_score(targets_flat, preds_flat)

    fpr, tpr, _ = roc_curve(targets_flat, preds_flat)
    try:
        fpr95 = fpr[np.where(tpr >= 0.95)[0][0]]
    except IndexError:
        fpr95 = 1.0  # Worst-case fallback

    return {
        'AP': ap,
        'FPR95': fpr95,
        'AUROC': auroc
    }

def iou_per_class(preds, targets, threshold=0.5, eps=1e-7):
    """
    Computes IoU for each class in multi-label predictions.
    Args:
        preds (torch.Tensor): Predicted probabilities of shape [B, C, H, W].
        targets (torch.Tensor): Ground truth labels of shape [B, C, H, W].
        threshold (float): Threshold for converting probabilities to binary predictions.
        eps (float): Small value to avoid division by zero.
    Returns:
        list: IoU for each class.
    """
    preds = (preds > threshold).float()
    ious = []
    for cls in range(preds.shape[1]):
        pred_cls = preds[:, cls]
        target_cls = targets[:, cls]
        intersection = (pred_cls * target_cls).sum(dim=(1, 2))
        union = (pred_cls + target_cls - pred_cls * target_cls).sum(dim=(1, 2))
        iou = (intersection + eps) / (union + eps)
        ious.append(iou.mean().item())
    return ious

def mean_iou(preds, targets, threshold=0.5):
    """
    Computes mean IoU across all classes.
    Args:
        preds (torch.Tensor): Predicted probabilities of shape [B, C, H, W].
        targets (torch.Tensor): Ground truth labels of shape [B, C, H, W].
        threshold (float): Threshold for converting probabilities to binary predictions.
    Returns:
        float: Mean IoU across all classes.
    """
    per_class_iou = iou_per_class(preds, targets, threshold)
    return sum(per_class_iou) / len(per_class_iou)

def dice_score(preds, targets, threshold=0.5, eps=1e-7):
    """
    Computes the Dice score for each class in multi-label predictions.
    Args:
        preds (torch.Tensor): Predicted probabilities of shape [B, C, H, W].
        targets (torch.Tensor): Ground truth labels of shape [B, C, H, W].
        threshold (float): Threshold for converting probabilities to binary predictions.
        eps (float): Small value to avoid division by zero.
    Returns:
        list: Dice scores for each class.
    """
    preds = (preds > threshold).float()
    scores = []
    for cls in range(preds.shape[1]):
        pred_cls = preds[:, cls]
        target_cls = targets[:, cls]
        intersection = (pred_cls * target_cls).sum(dim=(1, 2))
        union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2))
        dice = (2 * intersection + eps) / (union + eps)
        scores.append(dice.mean().item())
    return scores

def precision_recall(preds, targets, threshold=0.5, eps=1e-7):
    """
    Computes precision and recall for each class in multi-label predictions.
    Args:
        preds (torch.Tensor): Predicted probabilities of shape [B, C, H, W].
        targets (torch.Tensor): Ground truth labels of shape [B, C, H, W].
        threshold (float): Threshold for converting probabilities to binary predictions.
        eps (float): Small value to avoid division by zero.
    Returns:
        tuple: (precisions, recalls) where each is a list of values for each class.
    """
    preds = (preds > threshold).float()
    precisions, recalls = [], []
    for cls in range(preds.shape[1]):
        pred_cls = preds[:, cls]
        target_cls = targets[:, cls]
        tp = (pred_cls * target_cls).sum(dim=(1, 2))
        fp = (pred_cls * (1 - target_cls)).sum(dim=(1, 2))
        fn = ((1 - pred_cls) * target_cls).sum(dim=(1, 2))
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        precisions.append(precision.mean().item())
        recalls.append(recall.mean().item())
    return precisions, recalls


############################################# UOS #############################################
def unknown_objectness_score(preds):
    """
    Computes the unknown objectness score from the model predictions.
    Args:
        preds (torch.Tensor): Model predictions of shape [B, C, H, W] where C includes objectness channel.
    Returns:
        torch.Tensor: Unknown objectness scores of shape [B, H, W].
    """
    obj_scores = preds[:, 7, :, :]
    class_scores = preds[:, 0:7, :, :]
    
    unknown_scores = torch.prod(1 - class_scores, dim=1)
    uos = obj_scores * unknown_scores
    return uos


############################################# CONFORMAL P. #############################################
def nonconformity_score(preds):
    """
    Computes the nonconformity score for the predictions.
    Args:
        preds (torch.Tensor): Model predictions of shape [B, C, H, W] where C includes objectness channel.
    Returns:
        torch.Tensor: Nonconformity scores of shape [B, H, W].
    """
    uos = unknown_objectness_score(preds)  # S_unk-objectness(x)
    return uos  # α(x): nonconformity score  #Se non funziona prova a usare solo uos

def p_value(alpha, calibration_scores):
    """
    Computes the p-value for a given alpha threshold based on calibration scores.
    Args:
        alpha (float): The alpha threshold.
        calibration_scores (np.ndarray): Array of calibration scores.
    Returns:
        float: The p-value.
    """
    return (np.sum(calibration_scores <= alpha) + 1) / (len(calibration_scores) + 1)  # The +1 in numerator and denominator ensures that the resulting p-value is never exactly 0 or 1 (this is called smoothing for finite-sample guarantees)