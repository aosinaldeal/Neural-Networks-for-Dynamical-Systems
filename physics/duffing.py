import numpy as np
from scipy.integrate import solve_ivp


def generate_duffing_motion(
    delta,
    alpha,
    beta,
    gamma,
    omega,
    x0,
    v0,
    duration,
    samples
):
    """
    Generate the numerical solution of the Duffing oscillator.

    Equation:
        x'' + delta*x' + alpha*x + beta*x^3
        = gamma*cos(omega*t)
    """

    t = np.linspace(0, duration, samples)

    def duffing_equation(t, y):
        x, v = y

        dxdt = v

        dvdt = (
            -delta * v
            -alpha * x
            -beta * x**3
            +gamma * np.cos(omega * t)
        )

        return [dxdt, dvdt]

    solution = solve_ivp(
        duffing_equation,
        (0, duration),
        [x0, v0],
        t_eval=t
    )

    x = solution.y[0]
    v = solution.y[1]

    return t, x, v