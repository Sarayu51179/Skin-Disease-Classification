import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 7
LR = 1e-4
EPOCHS = 10
NUM_WORKERS = 4

DATA_DIR = "data/archive/images"
CSV_PATH = "data/archive/HAM10000_metadata.csv"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "best_model.pth")