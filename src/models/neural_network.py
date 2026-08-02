import torch.nn as nn
from src import config


def get_activation():

    activations = {
        "ReLU": nn.ReLU(),
        "Tanh": nn.Tanh(),
        "Sigmoid": nn.Sigmoid(),
        "ELU": nn.ELU(),
        "LeakyReLU": nn.LeakyReLU(),
    }

    if config.ACTIVATION not in activations:
        raise ValueError(
            f"Unknown activation function: {config.ACTIVATION}"
        )

    return activations[config.ACTIVATION]


class NeuralNetwork(nn.Module):

    def __init__(self, output_dim):
        super().__init__()

        layers = []

        activation = get_activation()


        # Input layer

        layers.append(
            nn.Linear(
                1,
                config.NEURONS_PER_LAYER
            )
        )

        layers.append(
            activation
        )


        # Hidden layers

        for _ in range(1, config.HIDDEN_LAYERS):

            layers.append(
                nn.Linear(
                    config.NEURONS_PER_LAYER,
                    config.NEURONS_PER_LAYER
                )
            )

            layers.append(
                get_activation()
            )


        # Output layer

        layers.append(
            nn.Linear(
                config.NEURONS_PER_LAYER,
                output_dim
            )
        )


        self.network = nn.Sequential(
            *layers
        )


    def forward(self, x):

        return self.network(x)