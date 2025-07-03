import numpy as np
from .material import elastic_material, creep_material


def element_routine(u_e, eps_cr_old, mat_props, dt, creep=True):
    """Compute element stiffness, internal force, updated creep strain and stress."""
    E = mat_props['E']
    A = mat_props['A']
    L = mat_props['L']
    dot_eps0 = mat_props.get('dot_eps0', 1.0)
    Bp = mat_props.get('B', 160.0)
    n = mat_props.get('n', 1.0)
    B = np.array([-1.0 / L, 1.0 / L])
    strain = B @ u_e
    if creep:
        sigma, C, eps_cr_new = creep_material(strain, eps_cr_old, dt, E,
                                              dot_eps0, Bp, n)
    else:
        sigma, C, eps_cr_new = elastic_material(strain, E)
        eps_cr_new = eps_cr_old
    k_e = A * L * C * (B[:, None] @ B[None, :])
    f_int = A * L * sigma * B
    return k_e, f_int, eps_cr_new, sigma
