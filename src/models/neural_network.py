import torch.nn as nn
from src import config


class NeuralNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(1, config.NEURONS_PER_LAYER),
            nn.Tanh(),

            nn.Linear(config.NEURONS_PER_LAYER, config.NEURONS_PER_LAYER),
            nn.Tanh(),

            #nn.Linear(config.NEURONS_PER_LAYER, config.NEURONS_PER_LAYER),
            #nn.Tanh(),

            nn.Linear(config.NEURONS_PER_LAYER, config.NEURONS_PER_LAYER),
            nn.Tanh(),

            nn.Linear(config.NEURONS_PER_LAYER, config.OUTPUT_DIM)

        )

    def forward(self, x):

        return self.network(x)