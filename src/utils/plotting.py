import matplotlib.pyplot as plt


SYSTEM_TITLES = {
    "harmonic": "Harmonic Oscillator Motion Approximation",
    "damped_harmonic": "Damped Harmonic Oscillator Motion Approximation",
    "forced_harmonic": "Forced Harmonic Oscillator Motion Approximation",
    "duffing": "Duffing Oscillator Motion Approximation",
    "pendulum": "Simple Pendulum Motion Approximation",
    "double_pendulum": "Double Pendulum Motion Approximation",
}

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


def plot_prediction(
    time,
    reference,
    prediction,
    title,
    xlabel,
    ylabel,
    save_path,
):
    plt.figure(figsize=(10, 5))

    plt.plot(
        time,
        reference,
        label="Reference solution",
        linewidth=2,
    )

    plt.plot(
        time,
        prediction,
        "--",
        label="Neural Network",
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend()
    plt.grid(True)

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_absolute_error(
    time,
    error,
    save_path,
):
    plt.figure(figsize=(10, 5))

    plt.plot(
        time,
        error,
    )

    plt.title(
        "Absolute Prediction Error"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Absolute Error"
    )

    plt.grid(True)

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

def plot_double_pendulum_prediction(
    time,
    reference,
    prediction,
    theta_number,
    save_path,
):
    plt.figure(figsize=(10, 5))

    plt.plot(
        time,
        reference,
        label="Reference solution",
        linewidth=2,
    )

    plt.plot(
        time,
        prediction,
        "--",
        label="Neural Network",
    )

    plt.title(
        f"Double Pendulum - Theta {theta_number}"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        f"Theta {theta_number} (rad)"
    )

    plt.legend()
    plt.grid(True)

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_double_pendulum_error(
    time,
    error,
    theta_number,
    save_path,
):
    plt.figure(figsize=(10, 5))

    plt.plot(
        time,
        error,
    )

    plt.title(
        f"Theta {theta_number} Absolute Error"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Absolute Error"
    )

    plt.grid(True)

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()