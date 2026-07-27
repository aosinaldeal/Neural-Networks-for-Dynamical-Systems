import os

import config

from training.train import train
from evaluation.evaluate import evaluate


# ==========================================
# Experiment parameters
# ==========================================

SEEDS = [
    42,
    57,
    123,
    999,
    2026
    ]

PENDULUM_GRAVITY = [
    1.62,
    5.0,
    9.81,
    15.0,
    25.0
]

# ==========================================
# Run experiments
# ==========================================

def run():

    experiment = 1

    total = (
        len(PENDULUM_GRAVITY)
        * len(SEEDS)
    )

    print("=" * 60)
    print(f"Total experiments: {total}")
    print("=" * 60)

    for seed in SEEDS:

        for gravity in PENDULUM_GRAVITY:

            experiment_name = (
                f"gravity_{gravity}"
                f"_seed_{seed}"
            )

            # ----------------------------------
            # Set experiment parameter
            # ----------------------------------

            config.PENDULUM_GRAVITY = gravity

            # ----------------------------------
            # Create folder
            # ----------------------------------

            results_folder = os.path.join(
                "results",
                config.TYPE,
                config.SWEEP_NAME,
                experiment_name,
            )

            os.makedirs(
                results_folder,
                exist_ok=True,
            )

            # ----------------------------------
            # Experiment information
            # ----------------------------------

            print()
            print("-" * 60)

            print(
                f"Running {experiment_name}"
            )

            print(
                f"Pendulum Gravity: "
                f"{config.PENDULUM_GRAVITY}"
            )

            print(
                f"Pendulum Length: "
                f"{config.PENDULUM_LENGTH}"
            )

            print(
                f"Initial Angle: "
                f"{config.INITIAL_ANGLE}"
            )

            print(
                f"Initial Angular Velocity: "
                f"{config.INITIAL_ANGULAR_VELOCITY}"
            )

            print(
                f"Duration: "
                f"{config.DURATION}"
            )

            print(
                f"Samples: "
                f"{config.SAMPLES}"
            )

            print(
                f"Seed: {seed}"
            )

            print("-" * 60)

            # ----------------------------------
            # Train
            # ----------------------------------

            train(
                experiment=experiment,
                experiment_name=experiment_name,
                seed=seed,
            )

            # ----------------------------------
            # Evaluate
            # ----------------------------------

            evaluate(
                experiment=experiment,
                experiment_name=experiment_name,
                seed=seed,
            )

            print(
                f"Finished {experiment_name}"
            )

            experiment += 1


if __name__ == "__main__":
    run()