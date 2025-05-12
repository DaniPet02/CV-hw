# Evaluation logic

import torch
from tqdm import tqdm
from data import get_dataloaders
from network import DeepLabV3Plus
from utils import compute_mIoU, visualize_predictions
import globals as G


def evaluate_model(model_path=None, show_example=True):
    _, val_loader = get_dataloaders()
    model = DeepLabV3Plus().to(G.DEVICE)
    
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=G.DEVICE))
        print(f"Model loaded from {model_path}")

    model.eval()
    total_miou = 0.0
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(val_loader, desc="Evaluating")):
            images = images.to(G.DEVICE)
            masks = masks.to(G.DEVICE)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            miou, _ = compute_mIoU(preds, masks, G.NUM_CLASSES + 1)
            total_miou += miou

            if show_example and i == 0:
                visualize_predictions(images[0].cpu(), preds[0].cpu(), masks[0].cpu())

    avg_miou = total_miou / len(val_loader)
    print(f"Average mIoU on validation set: {avg_miou:.4f}")


if __name__ == "__main__":
    evaluate_model(model_path=None, show_example=True)
