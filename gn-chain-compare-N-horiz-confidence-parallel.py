import numpy as np

# %% Problem

import problems.hanging_chain

params = problems.hanging_chain.Params(N_horiz=40, v_max=1, N_balls=9, n_dim=3)


# %% GN solver

import alpaqa as pa
from alpaqa import casadi_loader as cl

tolerance = 1e-10
params_gn = {
    "max_iter": 5000,
    "stop_crit": pa.PANOCStopCrit.ProjGradUnitNorm2,
    "gn_interval": 30,
    "gn_sticky": True,
}
params_lbfgs = params_gn | {
    "gn_interval": 0,
    "gn_sticky": False,
}

solver_gn = pa.PANOCOCPSolver(params_gn)
solver_lbfgs = pa.PANOCOCPSolver(params_lbfgs)

# %% Solve for different initial conditions

import concurrent.futures

time_scale = 1e3
repeat = 1
N_experiments = 256

seed=12345
rng = np.random.default_rng(seed=seed)
mpc_problem = problems.hanging_chain.build(params)
x_inits = np.empty((N_experiments, mpc_problem.nx))
for i in range(N_experiments):
    # Change initial state
    x_inits[i, :] = mpc_problem.init_state
    for _ in range(3):
        input = rng.uniform(-1, 1, (mpc_problem.nu, ))
        x_inits[i, :] = mpc_problem.f(
            x_inits[i, :], input, mpc_problem.param
        ).full().ravel()

def experiment(solver, problem):
    times = np.zeros((N_experiments, repeat))
    iters = np.zeros((N_experiments, repeat))
    
    def run_exp(i):
        # Change initial state
        problem.x_init = x_inits[i, :]
        for j in range(repeat):
            _, stats = solver(problem, tolerance, None, async_=True)
            times[i, j] = stats["elapsed_time"].total_seconds() * time_scale
            iters[i, j] = stats["iterations"]
            if stats["status"] != pa.SolverStatus.Converged:
                print("Failed:", stats["status"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pool.map(run_exp, range(N_experiments))
    return {
        "times": times.ravel(),
        "iterations": iters.ravel(),
    }


results = {k: [] for k in ["L-BFGS", "GN"]}
horizons = np.arange(10, 45 + 1, 1)
for N_horiz in horizons:
    params.N_horiz = N_horiz
    mpc_problem = problems.hanging_chain.build(params)
    N, nx, nu = mpc_problem.N, mpc_problem.nx, mpc_problem.nu

    # compile problem
    ocp_problem = cl.generate_and_compile_casadi_quadratic_control_problem(
        f=mpc_problem.f,
        N=N,
    )
    # Set problem parameters
    ocp_problem.Q = mpc_problem.Q
    ocp_problem.Q_N = ocp_problem.Q
    ocp_problem.R = mpc_problem.R
    ocp_problem.x_ref[:, :] = 0
    ocp_problem.x_ref[params.N_balls * params.n_dim, :] = 1
    ocp_problem.u_ref[:, :] = 0
    ocp_problem.μ[:, :] = 0
    ocp_problem.param = mpc_problem.param
    ocp_problem.U.lowerbound = mpc_problem.u_lb
    ocp_problem.U.upperbound = mpc_problem.u_ub
    # Solve problems
    res_gn = experiment(solver_gn, ocp_problem)
    res_lbfgs = experiment(solver_lbfgs, ocp_problem)
    print(
        f"{N_horiz: >3}) GN:     "
        f"{np.mean(res_gn['times']):7.3f} ± {np.std(res_gn['times']):7.3f} "
        f"({np.mean(res_gn['iterations']):7.3f} ± {np.std(res_gn['iterations']):7.3f} iter)\n"
        f"     L-BFGS: "
        f"{np.mean(res_lbfgs['times']):7.3f} ± {np.std(res_lbfgs['times']):7.3f} "
        f"({np.mean(res_lbfgs['iterations']):7.3f} ± {np.std(res_lbfgs['iterations']):7.3f} iter)\n"
    )
    results["GN"].append(res_gn)
    results["L-BFGS"].append(res_lbfgs)

import pickle
with open("gn-chain-compare-N-horiz-confidence-parallel.pkl", "wb") as f:
    pickle.dump((horizons, results), f)
