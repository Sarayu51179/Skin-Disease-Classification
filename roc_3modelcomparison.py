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
# 🔹 Load Model
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
# 🔹 Compute Macro ROC AUC
# ------------------------------------------------------

def compute_macro_roc(model, test_loader):

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    binary_labels = label_binarize(all_labels, classes=range(NUM_CLASSES))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(binary_labels[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(NUM_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= NUM_CLASSES
    macro_auc = auc(all_fpr, mean_tpr)

    return all_fpr, mean_tpr, macro_auc


# ------------------------------------------------------
# 🔹 Main
# ------------------------------------------------------

def main():

    train_df, val_df, test_df, classes = get_data_splits()

    test_transform = get_baseline_transform()
    test_dataset = HAMDataset(test_df, transform=test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    models_to_test = {
        "ResNet50": ("resnet50", "models/best_resnet50.pth"),
        "VGG16": ("vgg16", "models/best_vgg16.pth"),
        "MobileNetV2": ("mobilenetv2", "models/best_mobilenetv2.pth")
    }

    plt.figure(figsize=(8,6))

    for display_name, (model_key, path) in models_to_test.items():

        print(f"Computing ROC for {display_name}")

        model = load_model(model_key, path)
        fpr, tpr, macro_auc = compute_macro_roc(model, test_loader)

        plt.plot(fpr, tpr, label=f"{display_name} (AUC = {macro_auc:.3f})")

    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Macro-Averaged ROC Comparison")
    plt.legend()
    plt.tight_layout()

    plt.savefig("results/roc_comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()