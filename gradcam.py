import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import models
from torch.utils.data import DataLoader

import config
from dataset import get_data_splits, HAMDataset, get_baseline_transform


DEVICE = config.DEVICE
NUM_CLASSES = config.NUM_CLASSES


# ------------------------------------------------------
# 🔹 GradCAM Class
# ------------------------------------------------------

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_image, class_idx=None):
        output = self.model(input_image)

        if class_idx is None:
            class_idx = torch.argmax(output)

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = torch.mean(gradients, dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(DEVICE)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam.detach().cpu().numpy()


# ------------------------------------------------------
# 🔹 Load Model
# ------------------------------------------------------

def load_model(model_name, path):

    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
        target_layer = model.layer4[-1]

    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier[6] = torch.nn.Linear(4096, NUM_CLASSES)
        target_layer = model.features[-1]

    elif model_name == "mobilenetv2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features,
            NUM_CLASSES
        )
        target_layer = model.features[-1]

    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model, target_layer


# ------------------------------------------------------
# 🔹 Main
# ------------------------------------------------------

def main():

    train_df, val_df, test_df, classes = get_data_splits()

    test_transform = get_baseline_transform()
    test_dataset = HAMDataset(test_df, transform=test_transform)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Get ONE test image
    image, label = next(iter(test_loader))
    image = image.to(DEVICE)

    # Convert image for display
    img_np = image.squeeze().permute(1,2,0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    models_to_test = {
        "ResNet50": ("resnet50", "models/best_resnet50.pth"),
        "VGG16": ("vgg16", "models/best_vgg16.pth"),
        "MobileNetV2": ("mobilenetv2", "models/best_mobilenetv2.pth")
    }

    plt.figure(figsize=(15,5))

    for idx, (display_name, (model_key, path)) in enumerate(models_to_test.items()):

        model, target_layer = load_model(model_key, path)
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate(image)
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap / 255

        overlay = 0.6 * img_np + 0.4 * heatmap

        plt.subplot(1, 3, idx+1)
        plt.imshow(overlay)
        plt.title(display_name)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("results/gradcam_comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()