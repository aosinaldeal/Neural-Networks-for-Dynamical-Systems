import torch
import torch.nn as nn
import numpy as np
import random
import os

from src.data.generator import generate_physical_dataset
from src import config
from src.data.dataset import (
    numpy_to_tensor,
    split_dataset,
    create_dataloader
)
from src.utils.plotting import plot_loss
from src.utils.model_io import save_model
from src.models.neural_network import NeuralNetwork


# Training

def train(
    experiment=None,
    experiment_name=None,
    beta=None,
    forcing_amplitude=None,
    forcing_omega=None,
    seed=None,
    delta=None
):



    if experiment is None:
        experiment = 1

    if experiment_name is None:
        experiment_name = config.EXPERIMENT_NAME

    if beta is None:
        beta = config.BETA

    if forcing_amplitude is None:
        forcing_amplitude = config.FORCING_AMPLITUDE

    if forcing_omega is None:
        forcing_omega = config.FORCING_OMEGA

    if seed is None:
        seed = config.RANDOM_SEED
    
    if delta is None:
        delta = config.DUFFING_DELTA

    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.use_deterministic_algorithms(True)


    # Device

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


    # Generate dataset

    time, position = generate_physical_dataset(
        beta=beta,
        forcing_amplitude=forcing_amplitude,
        forcing_omega=forcing_omega,
        delta=delta,
    )

    output_dim = 1 if position.ndim == 1 else position.shape[1]


    # Convert to tensors

    time_tensor, position_tensor = numpy_to_tensor(
        time,
        position
    )

    # Split dataset

    (
        train_time,
        train_position,
        test_time,
        test_position,
    ) = split_dataset(
        time_tensor,
        position_tensor,
        train_ratio=config.TRAIN_RATIO,
        method=config.SPLIT_METHOD,
    )

    # DataLoader

    train_loader = create_dataloader(
        train_time,
        train_position,
        batch_size=config.BATCH_SIZE,
    )

    # Model

    model = NeuralNetwork(output_dim=output_dim).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
    )

    # Training

    model.train()

    train_losses = []
    test_losses = []

    test_time = test_time.to(device)
    test_position = test_position.to(device)

    for epoch in range(config.EPOCHS + 1):

        running_loss = 0.0

        # Training        

        for inputs, targets in train_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            predictions = model(inputs)

            loss = criterion(
                predictions,
                targets,
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)

        train_losses.append(average_loss)

        
        # Evaluation

        model.eval()

        with torch.no_grad():

            predictions = model(test_time)

            test_loss = criterion(
                predictions,
                test_position,
            )

        test_losses.append(
            test_loss.item()
        )

        model.train()

        if epoch % 50 == 0:

            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {average_loss:.6f} | "
                f"Test Loss: {test_loss.item():.6f}"
            )

    # Final evaluation

    model.eval()

    with torch.no_grad():

        predictions = model(test_time)

        test_loss = criterion(
            predictions,
            test_position,
        )

    print(
        f"\nFinal Test Loss: {test_loss.item():.6f}"
    )


    # Save loss
    
    loss_path = os.path.join(
    "results",
    config.TYPE,
    config.SWEEP_NAME,
    experiment_name,
)

    os.makedirs(
        loss_path,
        exist_ok=True,
    )


    plot_loss(
        train_losses,
        test_losses,
        os.path.join(
            loss_path,
            "loss.png",
        ),
)

    # Save model
    
    save_model(
        model,
        f"{config.TYPE}-{experiment_name}",
    )

    return {
    "train_loss": train_losses[-1],
    "test_loss": test_loss.item()
}


if __name__ == "__main__":
    train()