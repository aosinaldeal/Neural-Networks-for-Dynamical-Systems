import os
import torch


def save_model(model, model_name):
    """
    Save a trained PyTorch model.
    """

    os.makedirs("models/saved", exist_ok=True)

    path = f"models/saved/{model_name}.pth"

    torch.save(model.state_dict(), path)

    print(f"Model saved to {path}")


def load_model(model, model_name):
    """
    Load a trained PyTorch model.
    """

    path = f"models/saved/{model_name}.pth"

    model.load_state_dict(
        torch.load(path)
    )

    model.eval()

    print(f"Model loaded from {path}")

    return model