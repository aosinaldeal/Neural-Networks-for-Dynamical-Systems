import os
import csv

import config


def save_result(
    experiment,
    seed,
    mse,
    mae,
    max_error,
    r2,
):

    file_path = os.path.join(
    "results",
    config.TYPE,
    config.SWEEP_NAME,
    "summary.csv",
)

    os.makedirs(
        os.path.dirname(file_path),
        exist_ok=True,
    )

    file_exists = os.path.isfile(file_path)

    # ==========================================
    # Common data
    # ==========================================

    data = {
        "Exp": experiment,
        "Epochs": config.EPOCHS,
        "Learning rate": config.LEARNING_RATE,
        "Batch size": config.BATCH_SIZE,
        "Train ratio": config.TRAIN_RATIO,
        "Split method": config.SPLIT_METHOD,
        "Random seed": seed,
        "Optimizer": "Adam",
        "Loss function": "MSELoss",
        "Hidden layers": config.HIDDEN_LAYERS,
        "Neurons per layer": config.NEURONS_PER_LAYER,
        "Activation": config.ACTIVATION,
    }

    # ==========================================
    # Physics parameters
    # ==========================================

    if config.TYPE == "harmonic":

        data.update({
            "Amplitude": config.AMPLITUDE,
            "Omega": config.OMEGA,
            "Phase": config.PHASE,
        })

    elif config.TYPE == "damped_harmonic":

        data.update({
            "Amplitude": config.AMPLITUDE,
            "Omega": config.OMEGA,
            "Beta": config.BETA,
            "Phase": config.PHASE,
        })

    elif config.TYPE == "forced_harmonic":

        data.update({
            "Amplitude": config.AMPLITUDE,
            "Omega": config.OMEGA,
            "Beta": config.BETA,
            "Phase": config.PHASE,
            "Forcing Amplitude": config.FORCING_AMPLITUDE,
            "Forcing Omega": config.FORCING_OMEGA,
        })

    elif config.TYPE == "duffing":

        data.update({
            "Delta": config.DUFFING_DELTA,
            "Alpha": config.DUFFING_ALPHA,
            "Beta": config.DUFFING_BETA,
            "Gamma": config.DUFFING_GAMMA,
            "Omega": config.DUFFING_OMEGA,
            "Initial Position": config.INITIAL_POSITION,
            "Initial Velocity": config.INITIAL_VELOCITY,
        })

    elif config.TYPE == "pendulum":

        data.update({
            "Gravity": config.PENDULUM_GRAVITY,
            "Length": config.PENDULUM_LENGTH,
            "Initial Angle": config.INITIAL_ANGLE,
            "Initial Angular Velocity": config.INITIAL_ANGULAR_VELOCITY,
        })

    elif config.TYPE == "double_pendulum":

        data.update({
            "Mass 1": config.DOUBLE_PENDULUM_MASS_1,
            "Mass 2": config.DOUBLE_PENDULUM_MASS_2,
            "Length 1": config.DOUBLE_PENDULUM_LENGTH_1,
            "Length 2": config.DOUBLE_PENDULUM_LENGTH_2,
            "Gravity": config.DOUBLE_PENDULUM_GRAVITY,
            "Initial Theta 1": config.INITIAL_THETA_1,
            "Initial Theta 2": config.INITIAL_THETA_2,
            "Initial Omega 1": config.INITIAL_OMEGA_1,
            "Initial Omega 2": config.INITIAL_OMEGA_2,
        })


    else:

        raise ValueError(
            f"Unknown TYPE: {config.TYPE}"
        )

    # ==========================================
    # Dataset
    # ==========================================

    data.update({
        "Duration": config.DURATION,
        "Samples": config.SAMPLES,
    })

    # ==========================================
    # Results
    # ==========================================

    data.update({
        "MSE": mse,
        "MAE": mae,
        "Maximum Error": max_error,
        "R2 Score": r2,
    })

    # ==========================================
    # Write CSV
    # ==========================================

    with open(
        file_path,
        "a",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=data.keys(),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)