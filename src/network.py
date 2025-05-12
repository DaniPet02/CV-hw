# Model Architecture

import torch
import torch.nn as nn
from torchvision import models
import globals as G
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=G.NUM_CLASSES+1):  # +1 for 'unknown' class
        super().__init__()
        
        # Load model with standard COCO weights
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.model = models.segmentation.deeplabv3_resnet50(weights=weights)
        
        # Replace classifier to match our number of classes
        self.model.classifier[4] = nn.Conv2d(
            in_channels=256, out_channels=num_classes, kernel_size=1
        )

    def forward(self, x):
        return self.model(x)["out"]  # only returns prediction map (logits)


if __name__ == "__main__":
    model = DeepLabV3Plus().to(G.DEVICE)
    model.eval()
    x = torch.randn(1, 3, 512, 1024).to(G.DEVICE)
    with torch.no_grad():
        output = model(x)
    print(f"Output shape: {output.shape}")  # Should be [1, num_classes, H, W]