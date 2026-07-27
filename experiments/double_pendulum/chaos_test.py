import config

from training.train import train
from evaluation.evaluate import evaluate


# ==========================================
# Train base model
# ==========================================

config.TYPE = "double_pendulum"

config.OUTPUT_DIM = 2

config.SWEEP_NAME = "chaos_test"


# Nombre del modelo base
config.EXPERIMENT_NAME = "base"


# ------------------------------------------
# Initial conditions for training
# ------------------------------------------

config.INITIAL_THETA_1 = 1.0
config.INITIAL_THETA_2 = 1.0

config.INITIAL_OMEGA_1 = 0.0
config.INITIAL_OMEGA_2 = 0.0


config.DURATION = 20
config.SAMPLES = 3000


# ==========================================
# Train
# ==========================================

train(
    experiment=1,
    experiment_name="base",
    seed=config.RANDOM_SEED
)


# ==========================================
# Evaluate perturbed systems
# ==========================================

perturbations = [

    {
        "name": "theta1_plus_0.001",
        "theta1": 1.001,
        "theta2": 1.0,
    },

    {
        "name": "theta1_plus_0.01",
        "theta1": 1.01,
        "theta2": 1.0,
    },

    {
        "name": "theta1_plus_0.1",
        "theta1": 1.1,
        "theta2": 1.0,
    },

]


for i, exp in enumerate(perturbations, start=2):

    print("\n==============================")
    print(exp["name"])
    print("==============================\n")


    config.INITIAL_THETA_1 = exp["theta1"]
    config.INITIAL_THETA_2 = exp["theta2"]


    config.EXPERIMENT_NAME = exp["name"]


    evaluate(
        experiment=i,
        experiment_name=exp["name"],
        model_name="base",
        seed=config.RANDOM_SEED
    )