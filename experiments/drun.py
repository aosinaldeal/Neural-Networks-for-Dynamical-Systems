import config

from training.train import train
from evaluation.evaluate import evaluate


# ==========================================
# Double Pendulum - Length 2 sweep
# ==========================================

experiments = [

    {"name": "length2_0.5",  "length2": 0.5},
    {"name": "length2_0.75", "length2": 0.75},
    {"name": "length2_1.0",  "length2": 1.0},
    {"name": "length2_1.25", "length2": 1.25},
    {"name": "length2_1.5",  "length2": 1.5},
    {"name": "length2_2.0",  "length2": 2.0},

]


# ==========================================
# Global configuration
# ==========================================

config.TYPE = "double_pendulum"

config.OUTPUT_DIM = 2

config.SWEEP_NAME = "length2"


# Initial conditions

config.INITIAL_THETA_1 = 1.0
config.INITIAL_THETA_2 = 1.0

config.INITIAL_OMEGA_1 = 0.0
config.INITIAL_OMEGA_2 = 0.0


# Simulation

config.DURATION = 10
config.SAMPLES = 2000


# ==========================================
# Run experiments
# ==========================================

for i, exp in enumerate(experiments, start=1):

    print("\n================================")
    print(f"Experiment {i}: {exp['name']}")
    print("================================\n")


    config.EXPERIMENT_NAME = (
        f"experiment_{i}"
    )


    # Fixed parameters

    config.DOUBLE_PENDULUM_MASS_1 = 1.0
    config.DOUBLE_PENDULUM_MASS_2 = 1.0

    config.DOUBLE_PENDULUM_LENGTH_1 = 1.0
    config.DOUBLE_PENDULUM_LENGTH_2 = exp["length2"]

    config.DOUBLE_PENDULUM_GRAVITY = 9.81


    train(
        experiment=i,
        experiment_name=config.EXPERIMENT_NAME,
        seed=config.RANDOM_SEED,
    )


    evaluate(
        experiment=i,
        experiment_name=config.EXPERIMENT_NAME,
        seed=config.RANDOM_SEED,
    )