import numpy as np
import casadi as cs

import problems.hanging_chain as hanging_chain

use_jit = False

params = hanging_chain.Params()
mpc_problem = hanging_chain.build(params, cs.SX)
N, nx, nu = mpc_problem.N, mpc_problem.nx, mpc_problem.nu

# Initial state
x_0 = mpc_problem.init_state
for _ in range(5):  # apply maximum inputs for a couple of time steps
    x_0 = mpc_problem.f(x_0, [-1, 1, 1], mpc_problem.param).full().ravel()
print(x_0)

# Box constraints on the actuator force:
lowerbound_x = np.tile(mpc_problem.u_lb, mpc_problem.N)
upperbound_x = np.tile(mpc_problem.u_ub, mpc_problem.N)

#%% Solve using ipopt

ipopt_sol = None

u = mpc_problem.mpc_cost.mx_in(0)
mpc_param = cs.vertcat(x_0, mpc_problem.param)
nlp = {"x": u, "f": mpc_problem.mpc_cost(u, mpc_param)}
ipopt_solver = cs.nlpsol(
    "S",
    "ipopt",
    nlp,
    {"ipopt": {"tol": 1e-10, "constr_viol_tol": 1e-10}, "jit": use_jit},
)
ipopt_sol = ipopt_solver(lbx=lowerbound_x, ubx=upperbound_x)["x"]
print(np.array(ipopt_sol).reshape((-1, nu)))
