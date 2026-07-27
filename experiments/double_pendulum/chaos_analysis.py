import numpy as np
import matplotlib.pyplot as plt

import config

from physics.double_pendulum import simulate_double_pendulum


# ==========================================
# Initial conditions
# ==========================================

theta1_values = [
    1.0,
    1.0001,
    1.001,
    1.01,
    1.1,
]


trajectories = {}


# ==========================================
# Generate trajectories
# ==========================================

for theta1 in theta1_values:

    time, theta1_data, theta2_data, _, _ = simulate_double_pendulum(
        theta1_0=theta1,
        theta2_0=1.0,
        omega1_0=0.0,
        omega2_0=0.0,
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

    trajectories[theta1] = (
        theta1_data,
        theta2_data
    )


# ==========================================
# Compare with baseline
# ==========================================

base_theta1, base_theta2 = trajectories[1.0]


plt.figure(figsize=(10,5))


for theta1 in theta1_values[1:]:

    theta1_data, _ = trajectories[theta1]

    difference = np.abs(
        base_theta1 - theta1_data
    )

    plt.plot(
        time,
        difference,
        label=f"θ1={theta1}"
    )


plt.yscale("log")

plt.xlabel("Time")
plt.ylabel("|Δθ₁|")

plt.title(
    "Double Pendulum Sensitivity to Initial Conditions"
)

plt.grid(True)

plt.legend()


plt.savefig(
    "results/double_pendulum/chaos/sensitivity_theta1.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()