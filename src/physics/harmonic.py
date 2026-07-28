import numpy as np
import matplotlib.pyplot as plt

def harmonic_motion(t, amplitude, omega, phase):
    """
    Compute the position of a harmonic oscillator
    """
    return amplitude * np.cos(omega * t + phase)

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
    position = harmonic_motion(
    time,
    amplitude,
    omega,
    phase
    )
    return time, position




def main():
    time, position = generate_dataset(1, 2, 0, 10, 1000)

    plt.figure(figsize=(8,5))
    plt.plot(time, position)
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title("Harmonic Motion")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()