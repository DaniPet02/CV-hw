# Global Variables and Paths

# === PATHS ===
CITYSCAPES_PATH = "../cityscapes" # Path to the Cityscapes dataset

# === TRAINING HYPERPARAMETERS ===
BATCH_SIZE = 4
NUM_CLASSES = 19  # Cityscapes has 19 classes semantically relevant
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = "cuda"  # uses 'cuda' if available, otherwise 'cpu'

# === MODEL CHECKPOINTS ===
SAVE_DIR = "./checkpoints"
MODEL_NAME = "deeplabv3_resnet101"
SEED = 42