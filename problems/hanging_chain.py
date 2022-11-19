# %% Test Gauss-Newton for optimal control problems

import numpy as np
import casadi as cs
from dataclasses import dataclass


@dataclass
class Params:
    Ts: float = 0.2  #            Simulation sampling time         [s]
    N_horiz: int = 40  #          MPC horizon                      [time steps]
    N_balls: int = 9  #           Number of balls
    n_dim: int = 3  #             Number of spatial dimensions
    α: float = 25
    β: float = 1
    γ: float = 0.01
    m: float = 0.03  #            mass
    D: float = 0.1  #             spring constant
    L: float = 0.033  #           spring length
    v_max: float = 1  #           maximum actuator velocity
    g_grav: float = 9.81  #       Gravitational acceleration       [m/s²]


@dataclass
class Problem:
    param: np.ndarray
    N: int
    nx: int
    nu: int
    f: cs.Function
    l: cs.Function
    l_N: cs.Function
    mpc_cost: cs.Function
    u_lb: np.ndarray
    u_ub: np.ndarray
    init_state: np.ndarray
    Q: np.ndarray
    R: np.ndarray


def build(params: Params, sym=cs.SX) -> Problem:
    d, N = params.n_dim, params.N_balls
    # State and input vectors
    x = sym.sym("x", d, (N + 1))  # state: balls 1→N+1 positions
    v = sym.sym("v", d, N)  #       state: balls 1→N velocities
    input = sym.sym("u", d, 1)  #         input: ball 1+N velocity
    state = cs.vertcat(cs.vec(x), cs.vec(v))  # full state vector

    nx = np.product(state.shape)  # Number of states
    nu = np.product(input.shape)  # Number of inputs

    # Parameters
    Ts, N_horiz = params.Ts, params.N_horiz
    m = sym.sym("m")  # mass
    D = sym.sym("D")  # spring constant
    L = sym.sym("L")  # spring length
    param = cs.vertcat(m, D, L)
    concrete_param = np.array([params.m, params.D, params.L])
    g = params.g_grav * np.array([0, 0, -1] if d == 3 else [0, -1])  # gravity
    x_end = np.eye(1, d, 0).ravel()  # ball N+1 reference position

    # Continuous-time dynamics y' = f(y, u; p)
    f1 = [cs.vec(v), input]
    X = x
    dist_vect = cs.horzcat(X[:, 0], X[:, 1:] - X[:, :-1])
    dist_norm = cs.sqrt(cs.sum1(dist_vect * dist_vect))

    F = dist_vect @ cs.diag(D * (1 - L / dist_norm).T)
    fs = cs.horzcat(F[:, 1:] - F[:, :-1]) / m + cs.repmat(g, (1, N))

    f_c_expr = cs.vertcat(*f1, cs.vec(fs))
    f_c = cs.Function("f", [state, input, param], [f_c_expr])

    # Runge-Kutta integrator
    k1 = f_c(state, input, param)
    k2 = f_c(state + Ts * k1 / 2, input, param)
    k3 = f_c(state + Ts * k2 / 2, input, param)
    k4 = f_c(state + Ts * k3, input, param)

    # Discrete-time dynamics
    f_d_expr = state + (Ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    f_d = cs.Function("f", [state, input, param], [f_d_expr])

    # Model predictive control

    # MPC inputs and states
    mpc_x0 = sym.sym("x0", nx)  # Initial state
    mpc_u = sym.sym("u", (nu, N_horiz))  # Inputs
    mpc_x = f_d.mapaccum(N_horiz)(mpc_x0, mpc_u, param)  # Simulated states

    # MPC cost
    xt = sym.sym("xt", d * (N + 1), 1)
    vt = sym.sym("vt", d * N, 1)
    ut = sym.sym("ut", d, 1)
    yt = cs.vertcat(xt, vt)

    L_cost_x = params.α * cs.sumsqr(xt[-d:] - x_end)
    for i in range(N):
        xdi = vt[d * i : d * i + d]
        L_cost_x += params.β * cs.sumsqr(xdi)
    L_cost_u = params.γ * cs.sumsqr(ut)
    stage_cost = cs.Function(
        "l",
        [cs.vertcat(yt, ut), param],
        [L_cost_x + L_cost_u],
    )
    terminal_cost = cs.Function(
        "l_N",
        [yt, param],
        [L_cost_x],
    )

    Q = cs.diag(cs.evalf(cs.hessian(L_cost_x, yt)[0])).full().ravel()
    R = cs.diag(cs.evalf(cs.hessian(L_cost_u, ut)[0])).full().ravel()

    mpc_param = cs.vertcat(mpc_x0, param)
    mpc_xu = cs.vertcat(cs.horzcat(mpc_x0, mpc_x[:, :-1]), mpc_u)
    mpc_cost = cs.sum2(stage_cost.map(N_horiz)(mpc_xu, param))
    mpc_terminal_cost = terminal_cost(mpc_x[:, -1], param)
    mpc_tot_cost = mpc_cost + mpc_terminal_cost
    mpc_cost_fun = cs.Function("f_mpc", [cs.vec(mpc_u), mpc_param], [mpc_tot_cost])

    # Box constraints on the actuator velocity:
    v_lowerbound = -params.v_max * np.ones((nu,))
    v_upperbound = +params.v_max * np.ones((nu,))

    # Initial state
    x_0 = np.zeros((d * (N + 1),))
    x_0[0::d] = np.arange(1, N + 2) / (N + 1)
    v_0 = np.zeros((d * N,))
    init_state = np.concatenate((x_0, v_0))

    return Problem(
        param=concrete_param,
        N=N_horiz,
        nx=nx,
        nu=nu,
        f=f_d,
        l=stage_cost,
        l_N=terminal_cost,
        mpc_cost=mpc_cost_fun,
        u_lb=v_lowerbound,
        u_ub=v_upperbound,
        init_state=init_state,
        Q=Q,
        R=R,
    )


if __name__ == "__main__":
    print(build(Params()))
