# Model Architecture

import torch
import torch.nn as nn
from torchvision import models
import src.globals as G

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=G.NUM_CLASSES+1):  # +1 for 'unknown' class
        super(DeepLabV3Plus, self).__init__()

        # Load DeepLabv3+ with ResNet101 as backbone
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        
        # Substitute classificator to adapt it to my number of classes
        self.model.classifier[4] = nn.Conv2d(
            in_channels=256, out_channels=num_classes, kernel_size=1
        )

    def forward(self, x):
        return self.model(x)["out"]  # only returns prediction map (logits)

if __name__ == "__main__":
    # Test the model
    model = DeepLabV3Plus()
    x = torch.randn(1, 3, 512, 1024)  # Example input tensor
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be [1, num_classes, H, W]