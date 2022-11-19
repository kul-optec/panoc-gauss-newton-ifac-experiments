import numpy as np
from itertools import product
import pickle

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

plt.figure()
for use_gn, warm_start in product([False, True], [False, True]):
    name = (
        "gn-chain-mpc-"
        + ("GN" if use_gn else "LBFGS")
        + "-"
        + ("warm" if warm_start else "cold")
        + "-avg"  # '-single'
    )
    with open(name + ".pkl", "rb") as f:
        data = pickle.load(f)
    label = (
        ("GN" if use_gn else "L-BFGS") + " " + ("(warm)" if warm_start else "(cold)")
    )
    td = 1e3 * np.array(data["times"])
    Ndata = len(td)
    l, = plt.plot(td, "-", label=label, linewidth=1)
    # plt.axhline(np.max(td[1:]), xmin=0.99, xmax=1, color=l.get_color(), linestyle='-', linewidth=3)
    # plt.axhline(np.max(td[1:]), color=l.get_color(), linestyle=':', linewidth=0.5)
plt.title("Solver run times for model predictive control")
plt.ylabel(r"Run time $[\mathrm{ms}]$")
plt.xlabel("MPC time step")
plt.xlim(0, Ndata)
plt.ylim(0, None)

plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("gn-chain-mpc-times.pdf")
plt.show()
