import os

import torch
import matplotlib.pyplot as plt
import numpy as np

from src import config

from src.data.dataset import (
    numpy_to_tensor,
    split_dataset,
)

from src.models.neural_network import NeuralNetwork
from src.utils.model_io import load_model


# ==========================================
# Physics system
# ==========================================

if config.TYPE == "harmonic":
    from src.physics.harmonic import generate_dataset

elif config.TYPE == "damped_harmonic":
    from src.physics.damped_harmonic import generate_dataset

elif config.TYPE == "forced_harmonic":
    from src.physics.forced_harmonic import generate_dataset

elif config.TYPE == "duffing":
    from src.physics.duffing import generate_duffing_motion

elif config.TYPE == "pendulum":
    from src.physics.pendulum import generate_pendulum_motion

elif config.TYPE == "double_pendulum":
    from src.physics.double_pendulum import simulate_double_pendulum

else:
    raise ValueError(f"Unknown TYPE: {config.TYPE}")


# ==========================================
# Evaluation
# ==========================================

def evaluate(
    experiment=None,
    experiment_name=None,
    model_name=None,
    beta=None,
    seed=None,
    delta=None
):

    if experiment is None:
        experiment = 1

    if experiment_name is None:
        experiment_name = config.EXPERIMENT_NAME

    if beta is None:
        beta = config.BETA

    if seed is None:
        seed = config.RANDOM_SEED
    
    if delta is None:
        delta = config.DUFFING_DELTA

    torch.manual_seed(seed)

    # ----------------------------------------
    # Device
    # ----------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ----------------------------------------
    # Generate data
    # ----------------------------------------

    if config.TYPE == "duffing":

        time, position, velocity = generate_duffing_motion(
            alpha=config.DUFFING_ALPHA,
            beta=config.DUFFING_BETA,
            gamma=config.DUFFING_GAMMA,
            omega=config.DUFFING_OMEGA,
            x0=config.INITIAL_POSITION,
            v0=config.INITIAL_VELOCITY,
            duration=config.DURATION,
            samples=config.SAMPLES,
            delta=delta
        )

    elif config.TYPE == "pendulum":

        time, position, velocity = generate_pendulum_motion(
            gravity=config.PENDULUM_GRAVITY,
            length=config.PENDULUM_LENGTH,
            theta0=config.INITIAL_ANGLE,
            omega0=config.INITIAL_ANGULAR_VELOCITY,
            duration=config.DURATION,
            samples=config.SAMPLES,
        )

    elif config.TYPE == "double_pendulum":

        time, theta1, theta2, omega1, omega2 = simulate_double_pendulum(
            theta1_0=config.INITIAL_THETA_1,
            theta2_0=config.INITIAL_THETA_2,
            omega1_0=config.INITIAL_OMEGA_1,
            omega2_0=config.INITIAL_OMEGA_2,
            duration=config.DURATION,
            samples=config.SAMPLES,
            params={
                "MASS_1": config.DOUBLE_PENDULUM_MASS_1,
                "MASS_2": config.DOUBLE_PENDULUM_MASS_2,
                "LENGTH_1": config.DOUBLE_PENDULUM_LENGTH_1,
                "LENGTH_2": config.DOUBLE_PENDULUM_LENGTH_2,
                "GRAVITY": config.DOUBLE_PENDULUM_GRAVITY,
            }
        )

        position = np.column_stack(
            (
                theta1,
                theta2
            )
        )

    else:

        time, position = generate_dataset(
            amplitude=config.AMPLITUDE,
            omega=config.OMEGA,
            beta=beta,
            phase=config.PHASE,
            forcing_amplitude=config.FORCING_AMPLITUDE,
            forcing_omega=config.FORCING_OMEGA,
            duration=config.DURATION,
            samples=config.SAMPLES,
        )

    time_tensor, position_tensor = numpy_to_tensor(
        time,
        position,
    )

    # ----------------------------------------
    # Split dataset
    # ----------------------------------------

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

    test_time = test_time.to(device)
    test_position = test_position.to(device)

    # ----------------------------------------
    # Load model
    # ----------------------------------------

    model = NeuralNetwork().to(device)

    if model_name is None:
        model_name = experiment_name


    load_model(
        model,
        f"{config.TYPE}-{model_name}",
    )


    model.eval()

    # ----------------------------------------
    # Prediction
    # ----------------------------------------

    with torch.no_grad():
        predictions = model(test_time)

    # ========================================
    # Metrics
    # ========================================

    criterion = torch.nn.MSELoss()


    # ========================================
    # Double pendulum metrics
    # ========================================

    if config.TYPE == "double_pendulum":

        # theta 1

        mse_theta1 = criterion(
            predictions[:,0],
            test_position[:,0],
        )

        mae_theta1 = torch.mean(
            torch.abs(
                predictions[:,0] - test_position[:,0]
            )
        )

        max_error_theta1 = torch.max(
            torch.abs(
                predictions[:,0] - test_position[:,0]
            )
        )


        ss_res_1 = torch.sum(
            (test_position[:,0] - predictions[:,0]) ** 2
        )

        ss_tot_1 = torch.sum(
            (
                test_position[:,0]
                - torch.mean(test_position[:,0])
            ) ** 2
        )

        r2_theta1 = 1 - ss_res_1 / ss_tot_1



        # theta 2

        mse_theta2 = criterion(
            predictions[:,1],
            test_position[:,1],
        )

        mae_theta2 = torch.mean(
            torch.abs(
                predictions[:,1] - test_position[:,1]
            )
        )

        max_error_theta2 = torch.max(
            torch.abs(
                predictions[:,1] - test_position[:,1]
            )
        )


        ss_res_2 = torch.sum(
            (test_position[:,1] - predictions[:,1]) ** 2
        )

        ss_tot_2 = torch.sum(
            (
                test_position[:,1]
                - torch.mean(test_position[:,1])
            ) ** 2
        )

        r2_theta2 = 1 - ss_res_2 / ss_tot_2



        # global metrics (media de ambos)

        mse = (mse_theta1 + mse_theta2) / 2
        mae = (mae_theta1 + mae_theta2) / 2
        max_error = torch.max(
            torch.stack(
                [
                    max_error_theta1,
                    max_error_theta2
                ]
            )
        )

        r2 = (r2_theta1 + r2_theta2) / 2



    else:


        mse = criterion(
            predictions,
            test_position,
        )


        mae = torch.mean(
            torch.abs(
                predictions - test_position
            )
        )


        max_error = torch.max(
            torch.abs(
                predictions - test_position
            )
        )


        ss_res = torch.sum(
            (test_position - predictions) ** 2
        )


        ss_tot = torch.sum(
            (
                test_position
                - torch.mean(test_position)
            ) ** 2
        )


        r2 = 1 - (ss_res / ss_tot)

    # ----------------------------------------
    # Print metrics
    # ----------------------------------------

    print("Evaluation Results")
    print("------------------")

    print(f"MSE: {mse.item():.6f}")
    print(f"MAE: {mae.item():.6f}")
    print(
        f"Maximum Error: "
        f"{max_error.item():.6f}"
    )
    print(f"R2 Score: {r2.item():.6f}")

    if config.TYPE == "double_pendulum":

        print("\nTheta 1:")
        print(f"MSE: {mse_theta1.item():.6f}")
        print(f"MAE: {mae_theta1.item():.6f}")
        print(f"Maximum Error: {max_error_theta1.item():.6f}")
        print(f"R2 Score: {r2_theta1.item():.6f}")


        print("\nTheta 2:")
        print(f"MSE: {mse_theta2.item():.6f}")
        print(f"MAE: {mae_theta2.item():.6f}")
        print(f"Maximum Error: {max_error_theta2.item():.6f}")
        print(f"R2 Score: {r2_theta2.item():.6f}")

    # ========================================
    # Convert to NumPy
    # ========================================

    time_np = (
        test_time
        .cpu()
        .numpy()
        .flatten()
    )

    real_np = (
        test_position
        .detach()
        .cpu()
        .numpy()
    )

    pred_np = (
        predictions
        .detach()
        .cpu()
        .numpy()
    )

    print("TIME SHAPE:", time_np.shape)
    print("REAL SHAPE:", real_np.shape)
    print("PRED SHAPE:", pred_np.shape)
    # ----------------------------------------
    # Sort by time
    # ----------------------------------------

    order = time_np.argsort()

    time_np = time_np[order]
    real_np = real_np[order]
    pred_np = pred_np[order]

    # ========================================
    # Results folder
    # ========================================

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

    if config.TYPE == "double_pendulum":

    # ==============================
    # Theta 1
    # ==============================

        plt.figure(figsize=(10,5))

        plt.plot(
            time_np,
            real_np[:,0],
            label="Reference solution",
            linewidth=2,
        )

        plt.plot(
            time_np,
            pred_np[:,0],
            "--",
            label="Neural Network",
        )

        plt.title(
            "Double Pendulum - Theta 1"
        )

        plt.xlabel("Time")
        plt.ylabel("Theta 1 (rad)")

        plt.legend()
        plt.grid(True)

        plt.savefig(
            os.path.join(
                results_folder,
                "theta1_prediction.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


        # ==============================
        # Theta 2
        # ==============================

        plt.figure(figsize=(10,5))

        plt.plot(
            time_np,
            real_np[:,1],
            label="Reference solution",
            linewidth=2,
        )

        plt.plot(
            time_np,
            pred_np[:,1],
            "--",
            label="Neural Network",
        )

        plt.title(
            "Double Pendulum - Theta 2"
        )

        plt.xlabel("Time")
        plt.ylabel("Theta 2 (rad)")

        plt.legend()
        plt.grid(True)

        plt.savefig(
            os.path.join(
                results_folder,
                "theta2_prediction.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


        # ==============================
        # Errors
        # ==============================

        error_theta1 = abs(
            real_np[:,0] -
            pred_np[:,0]
        )

        error_theta2 = abs(
            real_np[:,1] -
            pred_np[:,1]
        )


        plt.figure(figsize=(10,5))

        plt.plot(
            time_np,
            error_theta1,
        )

        plt.title(
            "Theta 1 Absolute Error"
        )

        plt.xlabel("Time")
        plt.ylabel("Error")

        plt.grid(True)

        plt.savefig(
            os.path.join(
                results_folder,
                "theta1_error.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


        plt.figure(figsize=(10,5))

        plt.plot(
            time_np,
            error_theta2,
        )

        plt.title(
            "Theta 2 Absolute Error"
        )

        plt.xlabel("Time")
        plt.ylabel("Error")

        plt.grid(True)

        plt.savefig(
            os.path.join(
                results_folder,
                "theta2_error.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


    if config.TYPE != "double_pendulum":

    # Prediction plot
    

        plt.figure(figsize=(10, 5))

        plt.plot(
            time_np,
            real_np,
            label="Reference solution",
            linewidth=2,
        )

        plt.plot(
            time_np,
            pred_np,
            "--",
            label="Neural Network",
        )

        titles = {
            "harmonic": "Harmonic Oscillator Motion Approximation",
            "damped_harmonic": "Damped Harmonic Oscillator Motion Approximation",
            "forced_harmonic": "Forced Harmonic Oscillator Motion Approximation",
            "duffing": "Duffing Oscillator Motion Approximation",
            "pendulum": "Simple Pendulum Motion Approximation",
            "double_pendulum": "Double Pendulum Motion Approximation",
        }

        plt.title(
            titles.get(
                config.TYPE,
                "Oscillator Motion Approximation",
            )
        )

        plt.xlabel("Time (s)")
        plt.ylabel("θ(t) (rad)")

        plt.legend()
        plt.grid(True)

        plt.savefig(
            os.path.join(
                results_folder,
                f"{config.TYPE}_prediction.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


    if config.TYPE != "double_pendulum":

        # ========================================
        # Absolute error plot
        # ========================================

        absolute_error = abs(
            real_np - pred_np
        )

        plt.figure(figsize=(10, 5))

        plt.plot(
            time_np,
            absolute_error,
        )

        plt.title("Absolute Prediction Error")

        plt.xlabel("Time")
        plt.ylabel("Absolute Error")

        plt.grid(True)

        plt.savefig(
            os.path.join(
                results_folder,
                "absolute_error.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    # ========================================
    # Save metrics
    # ========================================

    metrics_path = os.path.join(
        results_folder,
        "metrics.txt",
    )

    with open(metrics_path, "w") as f:

        f.write(
            "========================================\n"
        )

        f.write(
            f"{experiment_name}\n"
        )

        f.write(
            "========================================\n\n"
        )

        # ------------------------------------
        # Hyperparameters
        # ------------------------------------

        f.write("Hyperparameters\n")
        f.write(
            "----------------------------------------\n"
        )

        f.write(f"Epochs: {config.EPOCHS}\n")

        f.write(
            f"Learning Rate: "
            f"{config.LEARNING_RATE}\n"
        )

        f.write(
            f"Batch Size: "
            f"{config.BATCH_SIZE}\n"
        )

        f.write(
            f"Train Ratio: "
            f"{config.TRAIN_RATIO}\n"
        )

        f.write(
            f"Split Method: "
            f"{config.SPLIT_METHOD}\n"
        )

        f.write(
            f"Random Seed: {seed}\n"
        )

        f.write("Optimizer: Adam\n")
        f.write("Loss Function: MSELoss\n")

        f.write(
            f"Hidden Layers: "
            f"{config.HIDDEN_LAYERS}\n"
        )

        f.write(
            f"Neurons per Layer: "
            f"{config.NEURONS_PER_LAYER}\n"
        )

        f.write(
            f"Activation: "
            f"{config.ACTIVATION}\n\n"
        )

        # ------------------------------------
        # Dataset
        # ------------------------------------

        f.write("Dataset\n")

        f.write(
            "----------------------------------------\n"
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

            f.write(
                f"Phase: {config.PHASE}\n"
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
                f"Phase: {config.PHASE}\n"
            )

            f.write(
                f"Forcing Amplitude: "
                f"{config.FORCING_AMPLITUDE}\n"
            )

            f.write(
                f"Forcing Omega: "
                f"{config.FORCING_OMEGA}\n"
            )

        elif config.TYPE == "duffing":

            f.write(
                f"Delta: {delta}\n"
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

            f.write(
                f"Initial Position: "
                f"{config.INITIAL_POSITION}\n"
            )

            f.write(
                f"Initial Velocity: "
                f"{config.INITIAL_VELOCITY}\n"
            )
        
        elif config.TYPE == "pendulum":

            f.write(
                f"Gravity: "
                f"{config.PENDULUM_GRAVITY}\n"
            )

            f.write(
                f"Length: "
                f"{config.PENDULUM_LENGTH}\n"
            )

            f.write(
                f"Initial Angle: "
                f"{config.INITIAL_ANGLE}\n"
            )

            f.write(
                f"Initial Angular Velocity: "
                f"{config.INITIAL_ANGULAR_VELOCITY}\n"
            )

        elif config.TYPE == "double_pendulum":

            f.write(
                f"Mass 1: "
                f"{config.DOUBLE_PENDULUM_MASS_1}\n"
            )

            f.write(
                f"Mass 2: "
                f"{config.DOUBLE_PENDULUM_MASS_2}\n"
            )

            f.write(
                f"Length 1: "
                f"{config.DOUBLE_PENDULUM_LENGTH_1}\n"
            )

            f.write(
                f"Length 2: "
                f"{config.DOUBLE_PENDULUM_LENGTH_2}\n"
            )

            f.write(
                f"Gravity: "
                f"{config.DOUBLE_PENDULUM_GRAVITY}\n"
            )

            f.write(
                f"Initial Theta 1: "
                f"{config.INITIAL_THETA_1}\n"
            )

            f.write(
                f"Initial Theta 2: "
                f"{config.INITIAL_THETA_2}\n"
            )

            f.write(
                f"Initial Omega 1: "
                f"{config.INITIAL_OMEGA_1}\n"
            )

            f.write(
                f"Initial Omega 2: "
                f"{config.INITIAL_OMEGA_2}\n"
            )

        f.write(
            f"Duration: {config.DURATION}\n"
        )

        f.write(
            f"Samples: {config.SAMPLES}\n\n"
        )
        # ------------------------------------
        # Results
        # ------------------------------------

        f.write("Results\n")

        f.write(
            "----------------------------------------\n"
        )

        f.write(
            f"MSE: {mse.item():.6f}\n"
        )

        f.write(
            f"MAE: {mae.item():.6f}\n"
        )

        f.write(
            f"Maximum Error: "
            f"{max_error.item():.6f}\n"
        )

        f.write(
            f"R2 Score: "
            f"{r2.item():.6f}\n"
        )

        if config.TYPE == "double_pendulum":

            f.write("\nTheta 1 Metrics\n")
            f.write("----------------------------------------\n")
            f.write(f"MSE: {mse_theta1.item():.6f}\n")
            f.write(f"MAE: {mae_theta1.item():.6f}\n")
            f.write(f"Maximum Error: {max_error_theta1.item():.6f}\n")
            f.write(f"R2 Score: {r2_theta1.item():.6f}\n")


            f.write("\nTheta 2 Metrics\n")
            f.write("----------------------------------------\n")
            f.write(f"MSE: {mse_theta2.item():.6f}\n")
            f.write(f"MAE: {mae_theta2.item():.6f}\n")
            f.write(f"Maximum Error: {max_error_theta2.item():.6f}\n")
            f.write(f"R2 Score: {r2_theta2.item():.6f}\n")
    print(
        f"Metrics saved to "
        f"{metrics_path}"
    )

    # ========================================
    # Save result to summary
    # ========================================

    from experiments.summary import save_result

    save_result(
        experiment=experiment,
        seed=seed,
        mse=mse.item(),
        mae=mae.item(),
        max_error=max_error.item(),
        r2=r2.item(),
    )


if __name__ == "__main__":
    evaluate()