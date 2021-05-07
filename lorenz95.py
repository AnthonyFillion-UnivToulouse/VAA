"""
The lorenz 95 model working with pytorch/numpy
"""


def EDO(x, F=8.0):
    """
    Ordinary differential equation dx/dt = f(x)
    """
    n = x.shape[1]
    dx = 0*x  # same shape

    # Circular borders
    dx[:, 0] = (x[:, 1] - x[:, n-2])*x[:, n-1] - x[:, 0] + F
    dx[:, 1] = (x[:, 2] - x[:, n-1])*x[:, 0] - x[:, 1] + F
    dx[:, n-1] = (x[:, 0] - x[:, n-3])*x[:, n-2] - x[:, n-1] + F
    # Others
    dx[:, 2:n-1] = (x[:, 3:n]-x[:, 0:n-3])*x[:, 1:n-2] - x[:, 2:n-1] + F
    return dx


def M(x, N, dt=0.05):  # TODO: also L63
    """
    ODE resolvant with fourth order Runge-Kutta
    """
    for _ in range(N):
        k1 = EDO(x)
        k2 = EDO(x + 0.5*dt*k1)
        k3 = EDO(x + 0.5*dt*k2)
        k4 = EDO(x + dt*k3)
        x = x + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
    return x
