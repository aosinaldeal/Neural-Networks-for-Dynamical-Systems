import os

from src import config


def generate_report(
    experiment_name,
    metrics,
    save_path,
):
    """
    Generate a complete experiment report.
    Includes:
    - Hyperparameters
    - Dataset configuration
    - Results
    """

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True,
    )

    with open(save_path, "w") as f:

        # ========================================
        # Header
        # ========================================

        f.write(
            "========================================\n"
        )

        f.write(
            f"{experiment_name}\n"
        )

        f.write(
            "========================================\n\n"
        )


        # ========================================
        # Hyperparameters
        # ========================================

        f.write("Hyperparameters\n")
        f.write(
            "----------------------------------------\n"
        )

        f.write(
            f"Epochs: {config.EPOCHS}\n"
        )

        f.write(
            f"Learning Rate: {config.LEARNING_RATE}\n"
        )

        f.write(
            f"Batch Size: {config.BATCH_SIZE}\n"
        )

        f.write(
            f"Train Ratio: {config.TRAIN_RATIO}\n"
        )

        f.write(
            f"Split Method: {config.SPLIT_METHOD}\n"
        )

        f.write(
            f"Random Seed: {config.RANDOM_SEED}\n"
        )

        f.write(
            "Optimizer: Adam\n"
        )

        f.write(
            "Loss Function: MSELoss\n"
        )

        f.write(
            f"Hidden Layers: {config.HIDDEN_LAYERS}\n"
        )

        f.write(
            f"Neurons per Layer: {config.NEURONS_PER_LAYER}\n"
        )

        f.write(
            f"Activation: {config.ACTIVATION}\n\n"
        )


        # ========================================
        # Dataset
        # ========================================

        f.write("Dataset\n")
        f.write(
            "----------------------------------------\n"
        )

        f.write(
            f"System: {config.TYPE}\n"
        )


        if config.TYPE == "harmonic":

            f.write(
                f"Amplitude: {config.AMPLITUDE}\n"
            )

            f.write(
                f"Omega: {config.OMEGA}\n"
            )

            f.write(
                f"Phase: {config.PHASE}\n"
            )


        elif config.TYPE == "damped_harmonic":

            f.write(
                f"Amplitude: {config.AMPLITUDE}\n"
            )

            f.write(
                f"Omega: {config.OMEGA}\n"
            )

            f.write(
                f"Beta: {config.BETA}\n"
            )


        elif config.TYPE == "forced_harmonic":

            f.write(
                f"Amplitude: {config.AMPLITUDE}\n"
            )

            f.write(
                f"Omega: {config.OMEGA}\n"
            )

            f.write(
                f"Beta: {config.BETA}\n"
            )

            f.write(
                f"Forcing Amplitude: {config.FORCING_AMPLITUDE}\n"
            )

            f.write(
                f"Forcing Omega: {config.FORCING_OMEGA}\n"
            )


        elif config.TYPE == "duffing":

            f.write(
                f"Delta: {config.DUFFING_DELTA}\n"
            )

            f.write(
                f"Alpha: {config.DUFFING_ALPHA}\n"
            )

            f.write(
                f"Beta: {config.DUFFING_BETA}\n"
            )

            f.write(
                f"Gamma: {config.DUFFING_GAMMA}\n"
            )

            f.write(
                f"Omega: {config.DUFFING_OMEGA}\n"
            )


        elif config.TYPE == "pendulum":

            f.write(
                f"Gravity: {config.PENDULUM_GRAVITY}\n"
            )

            f.write(
                f"Length: {config.PENDULUM_LENGTH}\n"
            )

            f.write(
                f"Initial Angle: {config.INITIAL_ANGLE}\n"
            )

            f.write(
                f"Initial Angular Velocity: {config.INITIAL_ANGULAR_VELOCITY}\n"
            )


        elif config.TYPE == "double_pendulum":

            f.write(
                f"Mass 1: {config.DOUBLE_PENDULUM_MASS_1}\n"
            )

            f.write(
                f"Mass 2: {config.DOUBLE_PENDULUM_MASS_2}\n"
            )

            f.write(
                f"Length 1: {config.DOUBLE_PENDULUM_LENGTH_1}\n"
            )

            f.write(
                f"Length 2: {config.DOUBLE_PENDULUM_LENGTH_2}\n"
            )

            f.write(
                f"Gravity: {config.DOUBLE_PENDULUM_GRAVITY}\n"
            )

            f.write(
                f"Initial Theta 1: {config.INITIAL_THETA_1}\n"
            )

            f.write(
                f"Initial Theta 2: {config.INITIAL_THETA_2}\n"
            )

            f.write(
                f"Initial Omega 1: {config.INITIAL_OMEGA_1}\n"
            )

            f.write(
                f"Initial Omega 2: {config.INITIAL_OMEGA_2}\n"
            )


        f.write(
            f"Duration: {config.DURATION}\n"
        )

        f.write(
            f"Samples: {config.SAMPLES}\n\n"
        )


        # ========================================
        # Results
        # ========================================

        f.write("Results\n")
        f.write(
            "----------------------------------------\n"
        )

        f.write(
            f"MSE: {metrics['mse']:.6f}\n"
        )

        f.write(
            f"MAE: {metrics['mae']:.6f}\n"
        )

        f.write(
            f"Maximum Error: {metrics['max_error']:.6f}\n"
        )

        f.write(
            f"R2 Score: {metrics['r2']:.6f}\n"
        )


        # Double pendulum individual metrics

        if config.TYPE == "double_pendulum":

            f.write(
                "\nTheta 1 Metrics\n"
            )

            f.write(
                "----------------------------------------\n"
            )

            f.write(
                f"MSE: {metrics['theta1_mse']:.6f}\n"
            )

            f.write(
                f"MAE: {metrics['theta1_mae']:.6f}\n"
            )

            f.write(
                f"Maximum Error: {metrics['theta1_max_error']:.6f}\n"
            )

            f.write(
                f"R2 Score: {metrics['theta1_r2']:.6f}\n"
            )


            f.write(
                "\nTheta 2 Metrics\n"
            )

            f.write(
                "----------------------------------------\n"
            )

            f.write(
                f"MSE: {metrics['theta2_mse']:.6f}\n"
            )

            f.write(
                f"MAE: {metrics['theta2_mae']:.6f}\n"
            )

            f.write(
                f"Maximum Error: {metrics['theta2_max_error']:.6f}\n"
            )

            f.write(
                f"R2 Score: {metrics['theta2_r2']:.6f}\n"
            )


    print(
        f"Report saved to {save_path}"
    )