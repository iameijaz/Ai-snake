import numpy as np
import matplotlib.pyplot as plt
from .element import element_routine


E = 80000.0  # MPa
B_PARAM = 160.0  # MPa
N_EXP = 12
DOT_EPS0 = 1.0

L1 = 30.0
L2 = 60.0
A1 = 8.0
A2 = 16.0
F_MAX = 1600.0
T1 = 1.0
T_TOTAL = 50.0


def generate_mesh(ne1, ne2):
    x1 = np.linspace(0, L1, ne1 + 1)
    x2 = np.linspace(L1, L1 + L2, ne2 + 1)[1:]
    nodes = np.concatenate([x1, x2])
    conn = []
    areas = []
    lengths = []
    for i in range(ne1):
        conn.append([i, i + 1])
        areas.append(A1)
        lengths.append(x1[i + 1] - x1[i])
    offset = len(x1) - 1
    for j in range(ne2):
        conn.append([offset + j, offset + j + 1])
        areas.append(A2)
        lengths.append(x2[j + 1] - x2[j] if j + 1 < len(x2) else L2 / ne2)
    return np.array(nodes), np.array(conn), np.array(areas), np.array(lengths)


def external_force(time):
    if time <= T1:
        return F_MAX * time / T1
    return F_MAX


def solve(mesh, times, creep=True):
    nodes, conn, areas, lengths = mesh
    n_nodes = len(nodes)
    n_elem = len(conn)
    dof = np.arange(n_nodes)
    free = dof[1:]  # node 0 fixed
    u = np.zeros((n_nodes, len(times)))
    sig = np.zeros((n_elem, len(times)))
    eps_cr = np.zeros(n_elem)
    for step in range(1, len(times)):
        dt = times[step] - times[step - 1]
        F_ext = np.zeros(n_nodes)
        # load applied at the interface node (between segments)
        F_ext[1] = external_force(times[step])
        u_iter = u[:, step - 1].copy()
        for _ in range(50):
            K = np.zeros((n_nodes, n_nodes))
            F_int = np.zeros(n_nodes)
            new_eps = np.zeros_like(eps_cr)
            s = np.zeros(n_elem)
            for e in range(n_elem):
                ue = u_iter[conn[e]]
                props = dict(E=E, A=areas[e], L=lengths[e], dot_eps0=DOT_EPS0,
                             B=B_PARAM, n=N_EXP)
                ke, fint_e, eps_new, sigma = element_routine(
                    ue, eps_cr[e], props, dt, creep=creep)
                new_eps[e] = eps_new
                s[e] = sigma
                for a in range(2):
                    for b in range(2):
                        K[conn[e, a], conn[e, b]] += ke[a, b]
                for a in range(2):
                    F_int[conn[e, a]] += fint_e[a]
            R = F_int - F_ext
            Kff = K[np.ix_(free, free)]
            Rf = R[free]
            try:
                du = np.linalg.solve(Kff, -Rf)
            except np.linalg.LinAlgError:
                raise RuntimeError('Singular stiffness matrix')
            u_iter[free] += du
            if (np.linalg.norm(Rf, np.inf) < 0.005 * max(np.linalg.norm(F_int[free], np.inf), 1.0) and
                    np.linalg.norm(du, np.inf) < 0.005 * max(np.linalg.norm(u_iter[free], np.inf), 1.0)):
                eps_cr = new_eps
                sig[:, step] = s
                break
        else:
            raise RuntimeError('Newton did not converge')
        u[:, step] = u_iter
    return u, sig


def analytic_displacements(force):
    u2 = force * L1 / (E * A1)
    u3 = u2 + force * L2 / (E * A2)
    return np.array([0.0, u2, u3])


def linear_verification():
    mesh = generate_mesh(1, 1)
    times = np.array([0.0, 1.0])
    u, _ = solve(mesh, times, creep=False)
    u_exact = analytic_displacements(F_MAX)
    assert np.allclose(u[:, -1], u_exact, atol=1e-6)
    nodes = mesh[0]
    xx = np.linspace(0, L1 + L2, 50)
    uu = np.piecewise(xx, [xx <= L1, xx > L1],
                      [lambda x: F_MAX * x / (E * A1),
                       lambda x: F_MAX * L1 / (E * A1) + F_MAX * (x - L1) / (E * A2)])
    plt.figure()
    plt.plot(nodes, u[:, -1], 'o-', label='FEM')
    plt.plot(xx, uu, label='Analytical')
    plt.xlabel('x [mm]')
    plt.ylabel('Displacement [mm]')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lin_verif.png')
    plt.close()


def creep_analysis(dt=0.1, ne1=1, ne2=1):
    mesh = generate_mesh(ne1, ne2)
    times = np.arange(0, T_TOTAL + dt, dt)
    u, sig = solve(mesh, times, creep=True)
    plt.figure()
    plt.plot(times, sig[0], label='sigma1')
    plt.plot(times, sig[-1], label='sigma2')
    plt.xlabel('Time [s]')
    plt.ylabel('Stress [MPa]')
    plt.legend()
    plt.tight_layout()
    plt.savefig('creep_stress.png')
    plt.close()

    plt.figure()
    for i in range(u.shape[0]):
        plt.plot(times, u[i], label=f'node {i}')
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [mm]')
    plt.legend()
    plt.tight_layout()
    plt.savefig('creep_disp.png')
    plt.close()

    assert np.all(np.abs(sig[:, -1] - sig[:, -2]) < 1e-3)
    return u, sig


def h_convergence():
    elems = [1, 2, 4, 8]
    disps = []
    for n in elems:
        u, _ = creep_analysis(dt=0.1, ne1=n, ne2=n)
        disps.append(u[-1, -1])
    plt.figure()
    plt.plot(elems, disps, 'o-')
    plt.xlabel('# elements per segment')
    plt.ylabel('u_free_end [mm]')
    plt.tight_layout()
    plt.savefig('h_conv.png')
    plt.close()


def dt_convergence():
    dts = [1, 0.5, 0.25, 0.125, 0.0625]
    disps = []
    for d in dts:
        u, _ = creep_analysis(dt=d, ne1=1, ne2=1)
        disps.append(u[-1, -1])
    plt.figure()
    plt.plot(dts, disps, 'o-')
    plt.xlabel('dt [s]')
    plt.ylabel('u_free_end [mm]')
    plt.tight_layout()
    plt.savefig('dt_conv.png')
    plt.close()


def main():
    linear_verification()
    creep_analysis()
    h_convergence()
    dt_convergence()


if __name__ == '__main__':
    main()
