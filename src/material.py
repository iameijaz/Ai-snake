import numpy as np


def elastic_material(strain, E):
    """Return stress and tangent for linear elastic behavior."""
    sigma = E * strain
    C = E
    return sigma, C, 0.0


def _creep_rate(sigma, dot_eps0, B, n):
    sign = np.sign(sigma) if sigma != 0 else 1.0
    return dot_eps0 * (abs(sigma) / B) ** n * sign


def _creep_rate_derivative(sigma, dot_eps0, B, n):
    if sigma == 0:
        return dot_eps0 * n / B * (0.0) ** (n - 1)  # zero
    return dot_eps0 * n * (abs(sigma) / B) ** (n - 1) / B


def creep_material(strain, eps_cr_old, dt, E, dot_eps0, B, n,
                   tol=1e-10, max_iter=50):
    """Return stress, algorithmic tangent and updated creep strain."""
    # initial guess assumes elastic predictor
    sigma = E * (strain - eps_cr_old)
    for _ in range(max_iter):
        rate = _creep_rate(sigma, dot_eps0, B, n)
        f = sigma + E * dt * rate - E * (strain - eps_cr_old)
        df = 1.0 + E * dt * _creep_rate_derivative(sigma, dot_eps0, B, n)
        delta = f / df
        sigma -= delta
        if abs(delta) < tol * max(1.0, abs(sigma)):
            break
    rate = _creep_rate(sigma, dot_eps0, B, n)
    eps_cr_new = eps_cr_old + dt * rate
    C = E / (1.0 + E * dt * _creep_rate_derivative(sigma, dot_eps0, B, n))
    return sigma, C, eps_cr_new
