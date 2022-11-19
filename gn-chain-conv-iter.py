# %% Test Gauss-Newton for optimal control problems

import numpy as np
import numpy.linalg as la
import casadi as cs
from pprint import pprint

import problems.hanging_chain as hanging_chain

params = hanging_chain.Params(N_horiz=40, v_max=1, N_balls=9, n_dim=3)

mpc_problem = hanging_chain.build(params)
N, nx, nu = mpc_problem.N, mpc_problem.nx, mpc_problem.nu

# Initial state
x_0 = mpc_problem.init_state
for _ in range(3):  # apply maximum inputs for a couple of time steps
    x_0 = mpc_problem.f(x_0, [1, 1, 1], mpc_problem.param).full().ravel()
print(x_0)

# %% Compile into an alpaqa control problem

import alpaqa as pa
from alpaqa import casadi_loader as cl

# Generate C code for the cost function, compile it, and load it as an
# alpaqa problem description:
ocp_problem = cl.generate_and_compile_casadi_control_problem(
    f=mpc_problem.f,
    l=mpc_problem.l,
    l_N=mpc_problem.l_N,
    N=N,
)
ocp_problem.param = mpc_problem.param
ocp_problem.cost_structure = pa.CostStructure.Quadratic

# Box constraints on the actuator force:
ocp_problem.U.lowerbound = mpc_problem.u_lb
ocp_problem.U.upperbound = mpc_problem.u_ub

# Parameters
ocp_problem.x_init = x_0

# %% Solvers

tol = 1e-12

gn_opts = {
    "print_interval": 50,
    "max_iter": 500,
    "stop_crit": pa.PANOCStopCrit.ProjGradUnitNorm2,
    "gn_interval": 1,
    "gn_sticky": True,
}
lbfgs_opts = gn_opts | {
    "gn_interval": 0,
    "gn_sticky": False,
}
gn_solver = pa.PANOCOCPSolver(gn_opts)
lbfgs_solver = pa.PANOCOCPSolver(lbfgs_opts)
print(str(gn_solver))

# %% GN solve

fbe_gn = []
fpr_gn = []
grad_gn = []
xk_gn = []


def gn_cb(info: pa.PANOCOCPProgressInfo):
    fbe_gn.append(info.φγ)
    fpr_gn.append(la.norm(info.p) / info.γ)
    grad_gn.append(la.norm(info.grad_ψ))
    xk_gn.append(info.u.copy())


gn_solver.set_progress_callback(gn_cb)
u, gn_stats = gn_solver(ocp_problem, tol, None, async_=True)
print(u.reshape((-1, nu)))
pprint(gn_stats)

# %% L-BFGS solver for comparison

fbe_lbfgs = []
fpr_lbfgs = []
grad_lbfgs = []
xk_lbfgs = []


def lbfgs_cb(info: pa.PANOCProgressInfo):
    fbe_lbfgs.append(info.φγ)
    fpr_lbfgs.append(la.norm(info.p) / info.γ)
    grad_lbfgs.append(la.norm(info.grad_ψ))
    xk_lbfgs.append(info.u.copy())


lbfgs_solver.set_progress_callback(lbfgs_cb)

u, lbfgs_stats = lbfgs_solver(ocp_problem, tol, None, async_=True)
print(u.reshape((-1, nu)))
pprint(lbfgs_stats)

# %% Print timings

from utils.print_timings import print_timings

print("\nGauss-Newton:")
print_timings(gn_stats)
print("\nL-BFGS:")
print_timings(lbfgs_stats)

# %% Plot the results

import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "ptm",
        "font.size": 15,
        "lines.linewidth": 1.5,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
)

plot_it = 160

figscale = 0.8
fig, (ax0, ax1) = plt.subplots(
    2, 1, sharex=True, figsize=(6.4 * figscale, 7 * figscale)
)

# ax0.set_title("Fixed-point residual")
ax0.semilogy(fpr_lbfgs, ".-", label="L-BFGS")
ax0.semilogy(fpr_gn, ".-", label="GN")
ax0.set_ylabel(
    r"Fixed-point residual"
    "\n"
    r"\hspace{2em} $\left\|R_\gamma\left(u^{(\nu)}\hspace{-1pt}\right)\right\|$"
)
# ax0.set_xlabel(r"Iteration $k$")
ax0.set_xlim(0, plot_it)
ax0.set_ylim(8e-12, None)
ax0.legend(loc="upper right")

x_star = xk_gn[-1]
dist_v = np.vectorize(lambda x: la.norm(x - x_star), signature="(m)->()")

# ax1.set_title("Distance")
ax1.semilogy(dist_v(xk_lbfgs), ".-", label="L-BFGS")
ax1.semilogy(dist_v(xk_gn), ".-", label="GN")
ax1.set_ylabel(
    r"Distance to solution"
    "\n"
    r"\hspace{2em} $\left\|u^{(\nu)} - u^\star\right\|$"
)
ax1.set_xlabel(r"Iteration $(\nu)$")
ax1.set_xlim(0, plot_it)
ax1.set_ylim(8e-13, None)
ax0.set_title("Convergence of PANOC")
plt.tight_layout()

plt.savefig("gn-chain-conv-iter.pdf")
plt.show()
