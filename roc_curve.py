import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from torchvision import models
from torch.utils.data import DataLoader

import config
from dataset import get_data_splits, HAMDataset, get_baseline_transform


DEVICE = config.DEVICE
NUM_CLASSES = config.NUM_CLASSES
BATCH_SIZE = config.BATCH_SIZE


# ------------------------------------------------------
# 🔹 Load Model Function
# ------------------------------------------------------

def load_model(model_name, path):

    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier[6] = torch.nn.Linear(4096, NUM_CLASSES)

    elif model_name == "mobilenetv2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features,
            NUM_CLASSES
        )

    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model


# ------------------------------------------------------
# 🔹 ROC Plot Function
# ------------------------------------------------------

def plot_roc(model, test_loader, classes, model_name):

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    binary_labels = label_binarize(all_labels, classes=range(NUM_CLASSES))

    plt.figure(figsize=(8,6))

    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(binary_labels[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{classes[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------
# 🔹 Main
# ------------------------------------------------------

def main():

    # Your dataset returns 4 values
    train_df, val_df, test_df, classes = get_data_splits()

    test_transform = get_baseline_transform()
    test_dataset = HAMDataset(test_df, transform=test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    models_to_test = {
        "resnet50": "models/best_resnet50.pth",
        "vgg16": "models/best_vgg16.pth",
        "mobilenetv2": "models/best_mobilenetv2.pth"
    }

    for name, path in models_to_test.items():
        print(f"\nGenerating ROC for {name}")
        model = load_model(name, path)
        plot_roc(model, test_loader, classes, name)


if __name__ == "__main__":
    main()