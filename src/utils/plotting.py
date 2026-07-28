import matplotlib.pyplot as plt


def plot_loss(train_losses, test_losses, save_path):
    """
    Plot training and test loss over epochs.
    """

    plt.figure(figsize=(8, 5), dpi=300)

    plt.plot(
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        test_losses,
        label="Test Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training History")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    plt.close()