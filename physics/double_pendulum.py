import numpy as np


def double_pendulum_derivatives(state, t, params):
    """
    Equations of motion for a double pendulum.

    State:
        theta1, theta2, omega1, omega2

    Returns:
        derivatives:
        dtheta1/dt,
        dtheta2/dt,
        domega1/dt,
        domega2/dt
    """

    theta1, theta2, omega1, omega2 = state

    m1 = params["MASS_1"]
    m2 = params["MASS_2"]

    L1 = params["LENGTH_1"]
    L2 = params["LENGTH_2"]

    g = params["GRAVITY"]

    delta = theta1 - theta2

    denominator1 = L1 * (2 * m1 + m2 - m2 * np.cos(2 * delta))

    denominator2 = L2 * (2 * m1 + m2 - m2 * np.cos(2 * delta))


    alpha1 = (
        -g * (2 * m1 + m2) * np.sin(theta1)
        - m2 * g * np.sin(theta1 - 2 * theta2)
        - 2 * np.sin(delta) * m2 *
        (
            omega2 ** 2 * L2
            + omega1 ** 2 * L1 * np.cos(delta)
        )
    ) / denominator1


    alpha2 = (
        2 * np.sin(delta) *
        (
            omega1 ** 2 * L1 * (m1 + m2)
            + g * (m1 + m2) * np.cos(theta1)
            + omega2 ** 2 * L2 * m2 * np.cos(delta)
        )
    ) / denominator2


    return np.array([
        omega1,
        omega2,
        alpha1,
        alpha2
    ])



def rk4_step(func, state, t, dt, params):
    """
    One Runge-Kutta 4 step.
    """

    k1 = func(state, t, params)

    k2 = func(
        state + dt * k1 / 2,
        t + dt / 2,
        params
    )

    k3 = func(
        state + dt * k2 / 2,
        t + dt / 2,
        params
    )

    k4 = func(
        state + dt * k3,
        t + dt,
        params
    )

    return state + dt * (
        k1 + 2*k2 + 2*k3 + k4
    ) / 6



def simulate_double_pendulum(
        theta1_0,
        theta2_0,
        omega1_0,
        omega2_0,
        duration,
        samples,
        params
):
    """
    Simulate double pendulum.

    Returns:
        time,
        theta1,
        theta2,
        omega1,
        omega2
    """

    dt = duration / (samples - 1)

    time = np.linspace(
        0,
        duration,
        samples
    )


    states = np.zeros(
        (samples,4)
    )


    states[0] = [
        theta1_0,
        theta2_0,
        omega1_0,
        omega2_0
    ]


    for i in range(samples-1):

        states[i+1] = rk4_step(
            double_pendulum_derivatives,
            states[i],
            time[i],
            dt,
            params
        )


    theta1 = states[:,0]
    theta2 = states[:,1]

    omega1 = states[:,2]
    omega2 = states[:,3]


    return (
        time,
        theta1,
        theta2,
        omega1,
        omega2
    )