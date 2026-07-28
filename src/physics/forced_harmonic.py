import numpy as np


def forced_oscillator(
    t,
    amplitude,
    omega0,
    beta,
    phase,
    forcing_amplitude,
    forcing_omega,
):
    """
    Forced damped harmonic oscillator.

    x(t) = transient + steady-state
    """

    # ---------- Transient response ----------
    omega_d = np.sqrt(np.maximum(omega0**2 - beta**2, 0))

    transient = (
        amplitude
        * np.exp(-beta * t)
        * np.cos(omega_d * t + phase)
    )

    # ---------- Steady-state response ----------
    denominator = np.sqrt(
        (omega0**2 - forcing_omega**2) ** 2
        + (2 * beta * forcing_omega) ** 2
    )

    steady_amplitude = forcing_amplitude / denominator

    delta = np.arctan2(
        2 * beta * forcing_omega,
        omega0**2 - forcing_omega**2,
    )

    steady = (
        steady_amplitude
        * np.cos(forcing_omega * t - delta)
    )

    return transient + steady


def generate_dataset(
    amplitude,
    omega,
    beta,
    phase,
    forcing_amplitude,
    forcing_omega,
    duration,
    samples,
):
    time = np.linspace(0, duration, samples)

    position = forced_oscillator(
        time,
        amplitude,
        omega,
        beta,
        phase,
        forcing_amplitude,
        forcing_omega,
    )

    return time, position