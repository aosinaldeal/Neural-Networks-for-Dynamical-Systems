import numpy as np

from src import config


# ==========================================================
# Dataset generator
# ==========================================================

def generate_physical_dataset(
    beta=None,
    forcing_amplitude=None,
    forcing_omega=None,
    delta=None,
):
    """
    Generate a dataset from the selected physical system.

    Returns:
        time: numpy array with time values
        position: numpy array with system state
    """

    if beta is None:
        beta = config.BETA

    if forcing_amplitude is None:
        forcing_amplitude = config.FORCING_AMPLITUDE

    if forcing_omega is None:
        forcing_omega = config.FORCING_OMEGA

    if delta is None:
        delta = config.DUFFING_DELTA


    # ------------------------------------------------------
    # Import selected physical system
    # ------------------------------------------------------

    if config.TYPE == "harmonic":

        from src.physics.harmonic import generate_dataset

        time, position = generate_dataset(
            amplitude=config.AMPLITUDE,
            omega=config.OMEGA,
            phase=config.PHASE,
            duration=config.DURATION,
            samples=config.SAMPLES,
        )


    elif config.TYPE == "damped_harmonic":

        from src.physics.damped_harmonic import generate_dataset

        time, position = generate_dataset(
            amplitude=config.AMPLITUDE,
            omega=config.OMEGA,
            beta=beta,
            phase=config.PHASE,
            duration=config.DURATION,
            samples=config.SAMPLES,
        )


    elif config.TYPE == "forced_harmonic":

        from src.physics.forced_harmonic import generate_dataset

        time, position = generate_dataset(
            amplitude=config.AMPLITUDE,
            omega=config.OMEGA,
            beta=beta,
            phase=config.PHASE,
            forcing_amplitude=forcing_amplitude,
            forcing_omega=forcing_omega,
            duration=config.DURATION,
            samples=config.SAMPLES,
        )


    elif config.TYPE == "duffing":

        from src.physics.duffing import generate_duffing_motion

        time, position, _ = generate_duffing_motion(
            alpha=config.DUFFING_ALPHA,
            beta=config.DUFFING_BETA,
            gamma=config.DUFFING_GAMMA,
            omega=config.DUFFING_OMEGA,
            x0=config.INITIAL_POSITION,
            v0=config.INITIAL_VELOCITY,
            duration=config.DURATION,
            samples=config.SAMPLES,
            delta=delta,
        )


    elif config.TYPE == "pendulum":

        from src.physics.pendulum import generate_pendulum_motion

        time, position, _ = generate_pendulum_motion(
            gravity=config.PENDULUM_GRAVITY,
            length=config.PENDULUM_LENGTH,
            theta0=config.INITIAL_ANGLE,
            omega0=config.INITIAL_ANGULAR_VELOCITY,
            duration=config.DURATION,
            samples=config.SAMPLES,
        )


    elif config.TYPE == "double_pendulum":

        from src.physics.double_pendulum import simulate_double_pendulum

        time, theta1, theta2, _, _ = simulate_double_pendulum(
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
            },
        )

        position = np.column_stack(
            (
                theta1,
                theta2,
            )
        )


    else:
        raise ValueError(
            f"Unknown physical system: {config.TYPE}"
        )

    # Ensure position has shape (samples, variables)

    if position.ndim == 1:
        position = position.reshape(-1, 1)


    return time, position