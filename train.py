import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

import config
from dataset import (
    HAMDataset,
    get_data_splits,
    get_baseline_transform,
    get_augmented_transform
)
from utils import (
    evaluate,
    analyze_misclassifications,
    compute_group_accuracy,
    print_metric_table,
    plot_model_comparison
)

# ==========================================
# 🔹 MODEL FACTORY
# ==========================================

def get_model(model_name, num_classes):

    if model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "vgg16":
        model = models.vgg16(weights="IMAGENET1K_V1")
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features,
            num_classes
        )

    elif model_name == "mobilenetv2":
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )

    else:
        raise ValueError("Unknown model name")

    return model.to(config.DEVICE)


# ==========================================
# 🔹 TRAINING FUNCTION
# ==========================================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

    return total_loss / len(loader)


# ==========================================
# 🔹 TRAIN + EVALUATE SINGLE MODEL
# ==========================================

def train_and_evaluate(model_name,
                       train_loader,
                       val_loader,
                       test_loader,
                       classes,
                       class_weights):

    print(f"\n🚀 Training {model_name.upper()}")

    model = get_model(model_name, len(classes))

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    best_val_loss = float("inf")
    model_path = os.path.join(config.MODEL_DIR,
                              f"best_{model_name}.pth")

    for epoch in range(config.EPOCHS):

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion
        )

        val_loss = validate(model, val_loader, criterion)

        print(f"Epoch [{epoch+1}/{config.EPOCHS}] "
              f"Train Loss: {train_loss:.4f} "
              f"Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)

    # Load best model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Final test evaluation
    preds, labels, acc, precision, recall, f1, cm = evaluate(
        model,
        test_loader,
        config.DEVICE,
        classes
    )

    print_metric_table(acc, precision, recall, f1)
    analyze_misclassifications(cm, classes, threshold=15)

    return acc, precision, recall, f1


# ==========================================
# 🔹 MAIN PIPELINE
# ==========================================

def main():

    print("🚀 Multi-Model Skin Disease Classification")

    # Load data
    train_df, val_df, test_df, classes = get_data_splits(
        reduced_size=2000
    )

    # Datasets
    train_dataset = HAMDataset(
        train_df,
        transform=get_augmented_transform()
    )

    val_dataset = HAMDataset(
        val_df,
        transform=get_baseline_transform()
    )

    test_dataset = HAMDataset(
        test_df,
        transform=get_baseline_transform()
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    # Class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label"]),
        y=train_df["label"]
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float
    ).to(config.DEVICE)

    # --------------------------------------
    # 🔥 MODEL COMPARISON
    # --------------------------------------

    model_list = ["resnet50", "vgg16", "mobilenetv2"]
    results = {}

    for model_name in model_list:

        acc, precision, recall, f1 = train_and_evaluate(
            model_name,
            train_loader,
            val_loader,
            test_loader,
            classes,
            class_weights
        )

        results[model_name] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # Plot comparison
    plot_model_comparison(results)

    print("\n🏆 Final Comparison Results:")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")

    print("\n✅ Experiment Completed Successfully!")


# ==========================================
# 🔹 RUN SCRIPT
# ==========================================

if __name__ == "__main__":
    main()