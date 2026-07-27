import numpy as np
from scipy.integrate import solve_ivp


def generate_pendulum_motion(
    gravity,
    length,
    theta0,
    omega0,
    duration,
    samples
):
    """
    Generate the numerical solution of the simple pendulum.

    Equation:
        theta'' + (g/L) * sin(theta) = 0
    """

    t = np.linspace(0, duration, samples)

    def pendulum_equation(t, y):
        theta, omega = y

        dthetadt = omega

        domegadt = (
            -(gravity / length)
            * np.sin(theta)
        )

        return [dthetadt, domegadt]

    solution = solve_ivp(
        pendulum_equation,
        (0, duration),
        [theta0, omega0],
        t_eval=t
    )

    theta = solution.y[0]
    omega = solution.y[1]

    return t, theta, omega