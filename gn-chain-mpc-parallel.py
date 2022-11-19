# %% Test Gauss-Newton for optimal control problems

import numpy as np
import problems.hanging_chain as hanging_chain
from copy import deepcopy

params = hanging_chain.Params(N_horiz=40, v_max=1, N_balls=9)

mpc_problem = hanging_chain.build(params)
N, nx, nu = mpc_problem.N, mpc_problem.nx, mpc_problem.nu

# Initial state
x_0 = mpc_problem.init_state
for _ in range(5):  # apply maximum inputs for a couple of time steps
    u_dist = np.array([-1, 1, 1], dtype=np.float64)
    x_0 = mpc_problem.f(x_0, u_dist, mpc_problem.param).full().ravel()

# %% Compile into an alpaqa problem

import alpaqa as pa
from alpaqa import casadi_loader as cl

# %% Compile into an alpaqa control problem

# compile problem
ocp_problem = cl.generate_and_compile_casadi_quadratic_control_problem(
    f=mpc_problem.f,
    N=N,
)
# Set problem cost matrices, references and multipliers
ocp_problem.Q = mpc_problem.Q
ocp_problem.Q_N = ocp_problem.Q
ocp_problem.R = mpc_problem.R
ocp_problem.x_ref[:, :] = 0
ocp_problem.x_ref[params.N_balls * params.n_dim, :] = 1
ocp_problem.u_ref[:, :] = 0
ocp_problem.μ[:, :] = 0

# Box constraints on the actuator force:
ocp_problem.U.lowerbound = mpc_problem.u_lb
ocp_problem.U.upperbound = mpc_problem.u_ub

# Parameters
ocp_problem.param = mpc_problem.param
ocp_problem.x_init = x_0

# %% Solver

import sys

tol = 1e-10
warm_start = "cold" not in sys.argv
use_gn = "lbfgs" not in sys.argv
gn_opts = {
    "print_interval": 0,
    "max_iter": 100,
    "stop_crit": pa.PANOCStopCrit.ProjGradUnitNorm2,
    "gn_interval": 10,
    "gn_sticky": True,
    "linesearch_tolerance_factor": tol,
    "quadratic_upperbound_tolerance_factor": tol,
    "β": 0.01,
}
p = pa.PANOCOCPParams()
lbfgs_opts = gn_opts | {
    "max_iter": 500,
    "gn_interval": 0,
    "gn_sticky": False,
}
get_solver = lambda: pa.PANOCOCPSolver(gn_opts if use_gn else lbfgs_opts)

# %% MPC controller

from datetime import timedelta


# Wrap the solver in a class that solves the optimal control problem at each
# time step, implementing warm starting:
class MPCController:
    def __init__(
        self, solver: pa.PANOCSolver, problem: pa.CasADiQuadraticControlProblem
    ):
        self.solver = solver
        self.problem = deepcopy(problem)
        self.tot_it = 0
        self.tot_time = timedelta()
        self.max_time = timedelta()
        self.times: list[float] = []
        self.iters: list[int] = []
        self.failures = 0
        self.U = None

    def __call__(self, y_n: np.ndarray, it: int):
        d = params.n_dim
        y_n = np.array(y_n).ravel()
        # Set the current state as the initial state
        self.problem.x_init = y_n
        # Shift the previous solution for warm starting
        if self.U is not None and warm_start:
            self.U = np.concatenate((self.U[d:], self.U[-d:]))
        elif not warm_start:
            self.U = None
        # Solve the optimal control problem
        # (warm start using the shifted previous solution and multipliers)
        self.U, stats = self.solver(self.problem, tol, self.U, async_=True)
        # Print some solver statistics
        status = stats["status"]
        success = status == pa.SolverStatus.Converged
        self.failures += not success
        self.tot_it += stats["iterations"]
        self.tot_time += stats["elapsed_time"]
        self.max_time = max(self.max_time, stats["elapsed_time"])
        self.times += (stats["elapsed_time"].total_seconds(),)
        self.iters += (stats["iterations"],)
        # Return the optimal control signal for the first time step
        return self.U[:d]


# %% Simulate the system using the MPC controller

N_sim = 300 + 1

y_sim = np.empty((mpc_problem.nx, N_sim), order="F")
y_sim[:, 0] = x_0  # Initial state for simulation
for n in range(N_sim - 1):
    y_sim[:, n + 1] = (
        mpc_problem.f(y_sim[:, n], [0, 0, 0], mpc_problem.param).full().ravel()
    )


def experiment(i):
    y_mpc = np.empty((mpc_problem.nx, N_sim), order="F")
    y_mpc[:, 0] = x_0  # Initial state for controller
    controller = MPCController(get_solver(), ocp_problem)
    for n in range(N_sim - 1):
        # Solve the optimal control problem:
        u_n = controller(y_mpc[:, n], n)
        # Apply the first optimal control input to the system and simulate for
        # one time step, then update the state:
        y_mpc[:, n + 1] = (
            mpc_problem.f(y_mpc[:, n], u_n, mpc_problem.param).full().ravel()
        )
    return np.array([controller.times, controller.iters])


# %% Save results

import pickle
import os
from concurrent.futures import ThreadPoolExecutor

N_cores = max(1, os.cpu_count() // 2 - 1)
N_experiments = N_cores * 3

name = (
    "gn-chain-mpc-"
    + ("GN" if use_gn else "LBFGS")
    + "-"
    + ("warm" if warm_start else "cold")
    + "-avg"
)
if N_cores == 1:
    name += "-single"

print(name)

with ThreadPoolExecutor(max_workers=N_cores) as pool:
    results = pool.map(experiment, range(N_experiments))
    avg_results = sum(results) / N_experiments
print(avg_results)

with open(name + ".pkl", "wb") as f:
    pickle.dump({"times": avg_results[0, :], "iters": avg_results[1, :]}, f)
