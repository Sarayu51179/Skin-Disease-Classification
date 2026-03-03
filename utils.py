import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# ==============================
# 🔹 MAIN EVALUATION FUNCTION
# ==============================

def evaluate(model, loader, device, classes):

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ======================
    # 📊 Metrics
    # ======================

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print("\n📊 Evaluation Metrics")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    # ======================
    # 📉 Confusion Matrix
    # ======================

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    return all_preds, all_labels, acc, precision, recall, f1, cm


# =====================================
# 🔎 MISCLASSIFICATION ANALYSIS
# =====================================

def analyze_misclassifications(cm, classes, threshold=20):

    print("\n🔎 Top Misclassifications:")

    found = False

    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i][j] > threshold:
                print(f"{classes[i]} misclassified as {classes[j]}: {cm[i][j]}")
                found = True

    if not found:
        print("No major misclassifications above threshold.")


# =====================================
# ⚖️ GROUP FAIRNESS ANALYSIS
# =====================================

def compute_group_accuracy(df, preds, labels, group_column="sex"):

    df = df.reset_index(drop=True).copy()

    correct = preds == labels
    df["correct"] = correct

    print(f"\n📊 Group Accuracy based on '{group_column}'")

    groups = df[group_column].dropna().unique()

    for g in groups:
        group_df = df[df[group_column] == g]
        total_group_samples = len(group_df)

        if total_group_samples == 0:
            continue

        correct_predictions = group_df["correct"].sum()
        group_accuracy = correct_predictions / total_group_samples

        print(f"{g}: {group_accuracy:.4f}")


# =====================================
# 📋 METRIC TABLE (OPTIONAL)
# =====================================

def print_metric_table(acc, precision, recall, f1):

    print("\n📋 Metric Summary Table")
    print("-" * 30)
    print(f"{'Metric':<15}{'Value':>10}")
    print("-" * 30)
    print(f"{'Accuracy':<15}{acc:>10.4f}")
    print(f"{'Precision':<15}{precision:>10.4f}")
    print(f"{'Recall':<15}{recall:>10.4f}")
    print(f"{'F1-score':<15}{f1:>10.4f}")
    print("-" * 30)

    # =====================================
# 📊 MODEL COMPARISON VISUALIZATION
# =====================================

def plot_model_comparison(results):

    import matplotlib.pyplot as plt
    import numpy as np

    model_names = list(results.keys())

    accuracy = [results[m]["accuracy"] for m in model_names]
    precision = [results[m]["precision"] for m in model_names]
    recall = [results[m]["recall"] for m in model_names]
    f1 = [results[m]["f1"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.2

    plt.figure(figsize=(10,6))

    plt.bar(x - 1.5*width, accuracy, width, label='Accuracy')
    plt.bar(x - 0.5*width, precision, width, label='Precision')
    plt.bar(x + 0.5*width, recall, width, label='Recall')
    plt.bar(x + 1.5*width, f1, width, label='F1-score')

    plt.xticks(x, model_names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("CNN Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_roc_curves(model, loader, device, classes):

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    binary_labels = label_binarize(all_labels, classes=range(len(classes)))

    plt.figure(figsize=(8,6))

    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(binary_labels[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{classes[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves per Class")
    plt.legend()
    plt.tight_layout()
    plt.show()