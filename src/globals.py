# Global Variables and Paths
import torch

# === PATHS ===
CITYSCAPES_PATH = "../cityscapes" # Path to the Cityscapes dataset
CITYSCAPES_CLASSES = {
    0: 'road',
    1: 'sidewalk',
    2: 'building',
    3: 'wall',
    4: 'fence',
    5: 'pole',
    6: 'traffic light',
    7: 'traffic sign',
    8: 'vegetation',
    9: 'terrain',
    10: 'sky',
    11: 'person',
    12: 'rider',
    13: 'car',
    14: 'truck',
    15: 'bus',
    16: 'train',
    17: 'motorcycle',
    18: 'bicycle',
    19: 'unknown'  # out-of-class values
}

# === TRAINING HYPERPARAMETERS ===
BATCH_SIZE = 4
NUM_CLASSES = 19  # Cityscapes has 19 classes semantically relevant
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # uses 'cuda' if available, 'cpu' otherwise

# === MODEL CHECKPOINTS ===
SAVE_DIR = "./checkpoints"
MODEL_NAME = "deeplabv3_resnet101"
SEED = 42