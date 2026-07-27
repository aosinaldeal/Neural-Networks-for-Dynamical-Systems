import torch
import numpy as np


def damped_oscillator(
    t,
    amplitude,
    omega0,
    beta,
    phase
):
    omega_d = np.sqrt(omega0**2 - beta**2)

    return (
        amplitude
        * np.exp(-beta * t)
        * np.cos(omega_d * t + phase)
    )
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
    position = damped_oscillator(
    time,
    amplitude,
    omega,
    beta,
    phase
    )
    return time, position
