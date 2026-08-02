# Neural Networks for Dynamical Systems

<p align="center">
  <img src="results/duffing/alpha_sweep/alpha_1.5_seed_123/duffing_prediction.png" width="800">
</p>

*A computational study of neural network approximation of deterministic and chaotic physical systems.*

---

## Overview

This repository investigates whether feedforward neural networks can learn the behavior of classical dynamical systems directly from simulated data.

The project evaluates the performance of multilayer perceptrons (MLPs) on a range of physical systems with increasing levels of complexity, from simple harmonic motion to chaotic dynamics. The objective is to identify both the capabilities and the limitations of neural networks when approximating deterministic and nonlinear systems.

The work was developed as an independent research project and is accompanied by a complete scientific paper included in this repository.

---

## Research Question

**Can a neural network learn the mathematical laws governing physical systems, and where does its predictive capability begin to fail as system complexity increases?**

---

## Implemented Physical Systems

The repository currently includes simulations and neural network models for:

* Harmonic Oscillator
* Damped Harmonic Oscillator
* Forced Harmonic Oscillator
* Duffing Oscillator
* Simple Pendulum
* Double Pendulum

Each system is implemented independently, allowing experiments under different physical parameters and dynamical regimes.

---

## Methodology

For each physical system:

1. Generate a reference dataset using analytical solutions or numerical integration.
2. Convert the generated data into training and testing datasets.
3. Train a fully connected neural network (MLP).
4. Evaluate prediction accuracy using multiple performance metrics.
5. Analyze the effect of physical parameters through parameter sweeps.

The study includes systematic experiments where physical parameters are modified to investigate the robustness and generalization capabilities of the neural network.

---

## Neural Network Architecture

The model used in this study is a fully connected feedforward neural network (Multilayer Perceptron, MLP).

The architecture is configurable through `src/config.py`, including:

* Number of hidden layers
* Number of neurons per layer
* Activation function
* Output dimension

Default configuration:

* Hidden layers: 3
* Neurons per layer: 128
* Activation function: Tanh
* Optimizer: Adam
* Loss function: Mean Squared Error (MSE)

The output dimension is automatically adapted depending on the physical system. For example:

* Single-variable systems → one output ($x(t)$ or $\theta$$(t)$)
* Double Pendulum → two outputs ($\theta_1(t)$ and $\theta_2(t)$)

---

## Evaluation Metrics

The following metrics are used throughout the project:

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Maximum Absolute Error
* Coefficient of Determination ($R^2$)

For every experiment, the pipeline automatically generates:

* Prediction plots
* Absolute error plots
* Training history plots
* Numerical evaluation reports

---

## Repository Structure

```text
.
├── src/
│   ├── data/             Dataset generation and processing
│   ├── models/           Neural network architectures
│   ├── utils/            Metrics, plotting and reporting utilities
│   ├── train.py          Training pipeline
│   ├── evaluate.py       Evaluation pipeline
│   └── config.py         Experiment configuration
│
├── experiments/          Parameter sweeps and analysis scripts
├── results/              Generated experimental results
├── paper/                Scientific paper (LaTeX source and PDF)
├── requirements.txt
└── README.md
```

---

## Technologies

* Python
* PyTorch
* NumPy
* SciPy
* Matplotlib
* Pandas

---

## Installation

Clone the repository:

```bash
git clone https://github.com/aosinaldeal/Neural-Networks-for-Dynamical-Systems.git

cd Neural-Networks-for-Dynamical-Systems
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running Experiments

### Train a model

```bash
python -m src.train
```

### Evaluate a trained model

```bash
python -m src.evaluate
```

The selected physical system and experiment parameters can be modified in:

```text
src/config.py
```

---

## Configuration

Experiments are controlled through:

```text
src/config.py
```

Main parameters include:

* `TYPE`: physical system to simulate
* `EPOCHS`: number of training epochs
* `LEARNING_RATE`: optimizer learning rate
* `SAMPLES`: number of generated data points
* `DURATION`: simulation time
* `HIDDEN_LAYERS`: number of hidden neural network layers
* `NEURONS_PER_LAYER`: neurons in each hidden layer
* `ACTIVATION`: neural network activation function

---

## Reproducibility

All experiments are designed to be reproducible.

The configuration file controls:

* Physical parameters
* Neural network hyperparameters
* Dataset generation
* Train/test splitting
* Random seeds

Each experiment automatically stores:

* Trained model weights
* Training history
* Prediction figures
* Error analysis
* Evaluation metrics

---

## Scientific Paper

The complete scientific paper can be found here:

**[📄 Neural Networks for Dynamical Systems (PDF)](paper/paper.pdf)**

The paper describes:

* The theoretical background of each dynamical system.
* The neural network architecture.
* The experimental methodology.
* Parameter sweep analyses.
* Results and discussion.
* Conclusions and future work.

---

## Future Work

Possible extensions of this project include:

* Recurrent and transformer-based neural networks.
* Physics-Informed Neural Networks (PINNs).
* Neural Operators.
* Additional chaotic systems.
* Long-term trajectory prediction.
* Generalization across different physical systems.

---

## License

This project is released under the MIT License.

See the [LICENSE](LICENSE) file for details.
