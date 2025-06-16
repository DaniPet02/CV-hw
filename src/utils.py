import os
import torch
import cv2
import numpy as np
import shutil
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve



############################################# CONSTANTS #############################################

# Macro class index mapping
MACRO_CLASSES = {
    "road": 0,
    "flat": 1,
    "human": 2,
    "vehicle": 3,
    "construction": 4,
    "background": 5,
    "pole": 6,
    "object": 7,  # auxiliary objectness channel
}

# Map from original label ID to (macro class, is_object)
CLASS_MAPPING = {
    7: ("road", False), # road
    8: ("flat", False), # sidewalk
    11: ("construction", False), # building
    12: ("construction", False), # wall
    13: ("construction", False), # fence
    17: ("pole", True),  # pole
    19: ("pole", True),  # traffic sign
    20: ("pole", True),  # traffic light
    21: ("background", False), # vegetation
    22: ("flat", False), # terrain
    23: ("background", False), # sky
    24: ("human", True), # person
    25: ("human", True), # rider
    26: ("vehicle", True), # car
    27: ("vehicle", True), # truck
    28: ("vehicle", True), # bus
    31: ("vehicle", True), # train
    32: ("vehicle", True), # motorcycle
    33: ("vehicle", True), # bicycle
}


############################################# DATASETS #############################################

def fix_cityscapes(path_in, path_out, image_folder_in="leftImg8bit", mask_folder_in="gtFine", is_delete=False):
    """
    Fixes the CityScapes dataset structure by renaming the files and removing the city folders.
    
    is_delete=True if you want to delete the city folders after copying
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
                continue

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

                if split != 'test':
                    # Renames and copies all 'label' associated files
                    suffixes = ['_gtFine_labelIds.png', '_gtFine_color.png', '_gtFine_instanceIds.png', '_gtFine_polygons.json']
                    for suffix in suffixes:
                        if suffix == '_gtFine_labelIds.png':
                            original_name = base_prefix + suffix
                            source = os.path.join(city_mask_dir, original_name)
                            if os.path.exists(source):
                                new_name = f"{new_base}_m.png"
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

def fix_lostandfound(path_in, path_out, image_folder_in="leftImg8bit", mask_folder_in="gtCoarse", is_delete=False):
    """
    Fixes the LostAndFound dataset by renaming the files and removing the city folders.

    is_delete=True if you want to delete the city folders after copying
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
                continue

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
                            new_name = f"{new_base}_m.png"
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

def fix_roadanomaly(path_in, path_out):
    """
    Fixes the RoadAnomaly dataset structure by renaming the files and removing the labels' folders.
    
    is_delete=True if you want to delete the city folders after copying
    """

    subfolder_in = "frames"
    count = 1
    working_dir = os.path.join(path_in, subfolder_in)

    substring_to_remove_from_mask = ".labels"

    # New destination directories
    img_out = os.path.join(path_out, 'img')
    mask_out = os.path.join(path_out, 'mask')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    # Iterate on sub-folders
    for frame in os.listdir(working_dir):

        frame_path = os.path.join(working_dir, frame)
        if not os.path.isdir(frame_path):
            continue

        mask_dir = os.path.join(working_dir, frame)

        for filename in os.listdir(mask_dir):
            
        
            if not filename.endswith('semantic_color.png'):
                continue
            mask_path = os.path.join(mask_dir, filename)
    
            base_prefix = filename.replace('semantic_color.png', '')

            # Renames and copies RGB
            new_base = f"{count}"
            ext = '.png'
            new_mask_name = f"mask_{new_base}{ext}"
            shutil.copy(mask_path, os.path.join(mask_out, new_mask_name))

            img_name = frame.replace(".labels", ".webp")
            img_path = os.path.join(working_dir, img_name)
            new_image_name = f"img_{new_base}{ext}"
            shutil.copy(img_path, os.path.join(img_out, new_image_name))

            count += 1
                    
def convert_label_to_multilabel_one_hot(label, dataset):
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

    height, width = label.shape
    multilabel = torch.zeros((8, height, width), dtype=torch.float32)

    if dataset == "cityscapes":
        # Iterate over each original class ID in CLASS_MAPPING
        for original_id, (_, is_object) in CLASS_MAPPING.items():

            # Boolean mask of where the input label equals the current original class ID
            mask = (label == original_id)

            # Look up the macro-class index for this class ID
            macro_idx = LABEL_TO_MACRO_IDX[original_id]

            # If this class maps to a macro-class we set 1 at those pixel in the corresponding channel
            if macro_idx is not None:
                multilabel[macro_idx][mask] = 1.0

            # If this class is considered an 'object' we also set the 'object' channel (index 7) to 1
            if is_object:
                multilabel[MACRO_CLASSES["object"]][mask] = 1.0
    elif dataset == "lostandfound":
        height, width = label.shape
        multilabel = torch.zeros((8, height, width), dtype=torch.float32)

        road_mask = (label == 1)
        object_mask_1 = (label != 1)
        object_mask_2 = (label != 0)
        object_mask = object_mask_1 == object_mask_2

        multilabel[MACRO_CLASSES["road"]][road_mask] = 1.0
        multilabel[MACRO_CLASSES["object"]][object_mask] = 1.0

    elif dataset == "roadanomaly":
        height, width = label.shape
        multilabel = torch.zeros((8, height, width), dtype=torch.float32)

        anomaly_mask = (label != 0)  # Anomaly pixels are not equal to 0

        multilabel[MACRO_CLASSES["object"]][anomaly_mask] = 1.0
    else:
        print("you have to choose a dataset between cityscapes and lostandfound\n ")

    # multi-label one-hot tensor of shape [8, H, W]
    return multilabel


############################################# BOUNDARY #############################################

def get_boundary_mask_batch(label_masks, kernel_size=3, iterations=2):
    """
    Computes the boundary mask of shape [B, H, W], as difference between dilated and eroded regions, for a batch of label masks.

    kernel_size: Size of the kernel to use.
    iterations: Number of times to apply dilation and erosion (higher values result in thicker boundaries).
    """

    if label_masks.dim() == 3:
        label_masks = label_masks.unsqueeze(1)  # [B, 1, H, W]

    label_masks = label_masks.float()

    # Define kernel
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


############################################# UOS #############################################

def unknown_objectness_score(preds):
    """
    Computes the unknown objectness score (UOS) from the model predictions. The UOS is computed as follows:
    
    UOS = obj_scores * Π(1 - class_scores_i)  for i in number of classes
    """
    obj_scores = preds[:, 7, :, :]
    class_scores = preds[:, 0:7, :, :]
    
    unknown_scores = torch.prod(1 - class_scores, dim=1)
    uos = obj_scores * unknown_scores

    # unknown objectness scores of shape [B, H, W]
    return uos


############################################# CONFORMAL P. #############################################

def nonconformity_score(preds):
    """
    Computes the nonconformity score for the predictions computed as UOS.
    """
    uos = unknown_objectness_score(preds)
    return uos

############################################# TEST #############################################

def evaluate_metrics(model, dataloader, class_indices, device):
    """
    Computes AP, FPR@95, and AUROC on selected class indices for multi-class segmentation.

    Args:
        model (torch.nn.Module): Trained PyTorch model for segmentation (output shape: [B, 1, H, W]).
        dataloader (torch.utils.data.DataLoader): Validation loader yielding (images, masks).
        class_indices (int or list of int): Index or indices of classes to keep (e.g. 1 or [1, 2]).
        device (str): Device for model inference.

    Returns:
        dict: Dictionary with AP, FPR95, and AUROC values over the selected mask region.
    """
    model.eval()
    all_preds = []
    all_targets = []

    if isinstance(class_indices, int):
        class_indices = [class_indices]

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            probs = model(images)  # shape: [B, 1, H, W], already sigmoid
            probs = probs.view(-1).cpu()
            masks = masks.view(-1).cpu()

            all_preds.append(probs)
            all_targets.append(masks)

    preds_flat = torch.cat(all_preds).numpy()
    targets_flat = torch.cat(all_targets).numpy()

    # Build binary ground truth for selected classes
    filter_mask = np.isin(targets_flat, class_indices)
    if not np.any(filter_mask):
        return {
            'AP': float('nan'),
            'FPR95': float('nan'),
            'AUROC': float('nan'),
            'Note': 'No pixels found for selected classes.'
        }

    # Binary labels: 1 for selected classes, 0 for everything else
    filtered_targets = np.isin(targets_flat, class_indices).astype(np.uint8)
    filtered_preds = preds_flat[filter_mask]
    filtered_targets = filtered_targets[filter_mask]

    if len(np.unique(filtered_targets)) < 2:
        return {
            'AP': float('nan'),
            'FPR95': float('nan'),
            'AUROC': float('nan'),
            'Note': 'Filtered pixels contain only one class. Metrics not computable.'
        }

    ap = average_precision_score(filtered_targets, filtered_preds)
    auroc = roc_auc_score(filtered_targets, filtered_preds)

    fpr, tpr, _ = roc_curve(filtered_targets, filtered_preds)
    try:
        fpr95 = fpr[np.where(tpr >= 0.95)[0][0]]
    except IndexError:
        fpr95 = 1.0  # Conservative fallback

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

############################################# EVALUATION #############################################

def evaluate_detection_performance_metrics(model, dataloader, ignore_class=7, eps=1e-7, device="cpu"):
    """
    Evaluate segmentation model on dataloader, returning pixel accuracy, mean IoU, mean F1 (Dice).
    """
    model.eval()
    pixel_acc_sum = 0
    iou_sum = 0
    f1_sum = 0
    n_batches = 0

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            preds_logits = model(images)

            # Pixel Accuracy
            # Exclude ignore_class channel and take argmax
            preds = preds_logits[:, :ignore_class, :, :]
            pred_classes = torch.argmax(preds, dim=1)

            if masks.dim() == 4 and masks.shape[1] > 1:
                # Convert one-hot mask [B,C,H,W] to class indices [B,H,W]
                masks = torch.argmax(masks, dim=1)

            valid_mask = masks != ignore_class
            correct = (pred_classes == masks) & valid_mask
            pixel_acc = correct.sum().float() / valid_mask.sum().float()

            # IoU per class
            ious = []
            for cls in range(ignore_class):
                pred_mask = (pred_classes == cls)
                target_mask = (masks == cls)

                intersection = (pred_mask & target_mask).sum().float()
                union = (pred_mask | target_mask).sum().float()
                iou = (intersection + eps) / (union + eps)
                ious.append(iou)
            mean_iou = torch.stack(ious).mean()

            # F1 score per class
            f1s = []
            for cls in range(ignore_class):
                pred_mask = (pred_classes == cls)
                target_mask = (masks == cls)

                intersection = (pred_mask & target_mask).sum().float()
                pred_sum = pred_mask.sum().float()
                target_sum = target_mask.sum().float()
                f1 = (2 * intersection + eps) / (pred_sum + target_sum + eps)
                f1s.append(f1)
            mean_f1 = torch.stack(f1s).mean()

            pixel_acc_sum += pixel_acc.item()
            iou_sum += mean_iou.item()
            f1_sum += mean_f1.item()
            n_batches += 1

    # Compute average metrics
    pixel_accuracy = pixel_acc_sum / n_batches
    mean_iou = iou_sum / n_batches
    mean_f1 = f1_sum / n_batches

    return {
        'pixel_accuracy': pixel_accuracy,
        'mean_iou': mean_iou,
        'mean_f1': mean_f1,
    }

def evaluate_uncertainty_metrics(model, dataloader, class_index=7, device="cpu"):
    """
    Computes AP, AUROC, FPR@95 based on predicted probabilitiess for a specific class in segmentation.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Extract ground truth for class 7 (unknown objects)
            gt_binary = masks[:, 7, :, :]

            probs = model(images)
            uos = unknown_objectness_score(probs)
            uos = uos.cpu().numpy() 
            uos = uos.reshape(-1)
            gt_flat = gt_binary.reshape(-1).cpu().numpy().astype(np.uint8)

            all_preds.append(uos)
            all_targets.append(gt_flat)

    preds_flat = np.concatenate(all_preds)
    targets_flat = np.concatenate(all_targets)

    # AP
    ap = average_precision_score(targets_flat, preds_flat)

    # AUROC
    auroc = roc_auc_score(targets_flat, preds_flat)

    # FPR@95TPR
    fpr, tpr, thresholds = roc_curve(targets_flat, preds_flat)
    try:
        fpr95 = fpr[np.where(tpr >= 0.95)[0][0]]
    except IndexError:
        fpr95 = 1.0

    return {
        'AP': ap,
        'FPR95': fpr95,
        'AUROC': auroc
    }

############################################# VISUALIZATION #############################################

def visualize_one_hot(one_hot, class_names=None, max_classes=8):
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

def visualize_erosion_mask(label_mask, iterations):
    """
    Visualizes the erosion mask of a label mask.
    """
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(label_mask, kernel, iterations=iterations)
    plt.figure()
    plt.imshow(erosion, cmap='gray')
    plt.title("Erosion Mask")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_dilation_mask(label_mask, iterations):
    """
    Visualizes the dilation mask of a label mask.
    """
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(label_mask, kernel, iterations=iterations)
    plt.figure()
    plt.imshow(dilation, cmap='gray')
    plt.title("Dilation Mask")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_boundary_mask(label_mask, iterations):
    """
    Visualizes the boundary mask.
    """
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(label_mask, kernel, iterations=iterations)
    eroded = cv2.erode(label_mask, kernel, iterations=iterations)
    boundary = (dilated != eroded).astype(np.uint8)
    plt.figure()
    plt.imshow(boundary, cmap='gray')
    plt.title("Boundary Mask")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def uos_heatmap(img_tensor, uos_tensor, threshold=0.5, alpha_val=0.5):
    """
    Superpose the heatmap of the unknown objectness score to the original image.
    
    threshold: float, minimum normalized UOS value to be shown
    alpha_val: float, fixed transparency level for visible heatmap areas
    """
    # Convert image
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    uos = uos_tensor.cpu().numpy()
    
    # Apply threshold
    uos = np.where(uos > threshold, uos, 0.0)

    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.imshow(uos, cmap='hot', alpha=alpha_val, vmin=0, vmax=1)  # Use 'hot' colormap for heatmap
    plt.title("Unknown Objectness Score Heatmap (Thresholded)")
    plt.colorbar(label="UOS score (0-1)")
    plt.axis('off')
    plt.show()

def visualize_uos_with_conformal(test_image, uos_tensor, threshold):
    """
    Compute and visualize the UOS heatmap on the original image applying 
    conformal threshold to detect unknown pixels.

    threshold: UOS threshold from calibration.
    """
    img = test_image.permute(1, 2, 0).cpu().numpy()
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    uos = uos_tensor.cpu().numpy()

    # Create binary mask of unknowns using conformal threshold
    unknown_mask = np.where(uos > threshold, uos, 0.0)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
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
    plt.imshow(unknown_mask, cmap='cool', alpha=0.5, vmin=0, vmax=1)
    plt.axis('off')

    plt.tight_layout()
    plt.show()