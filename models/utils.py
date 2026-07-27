import os
import torch


def save_model(model, model_name):
    """
    Save a trained PyTorch model
    """

    os.makedirs("models/saved", exist_ok=True)

    path = f"models/saved/{model_name}.pth"

    torch.save(model.state_dict(), path)

    print(f"Model saved to {path}")


def load_model(model, model_name):
    """
    Load a trained PyTorch model
    """

    path = f"models/saved/{model_name}.pth"

    model.load_state_dict(torch.load(path))

    model.eval()

    print(f"Model loaded from {path}")

    return model

import matplotlib.pyplot as plt


def plot_loss(train_losses, test_losses, save_path):
    """
    Plot training and test loss over epochs.
    """

    plt.figure(figsize=(8, 5), dpi=300)

    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_losses, label="Test Loss")

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training History")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)

    plt.close()