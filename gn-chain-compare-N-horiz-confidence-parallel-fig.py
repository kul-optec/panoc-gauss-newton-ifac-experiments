import numpy as np
import pickle

with open("gn-chain-compare-N-horiz-confidence-parallel.pkl", "rb") as f:
    horizons, results = pickle.load(f)

# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# list of dicts to dict of lists
results = {s: {k: [d[k] for d in r] for k in r[0]} for s, r in results.items()}
data = {s: pd.DataFrame(r) for s, r in results.items()}

plt.figure()
colors = ("tab:blue", "tab:orange")
for k, color in zip(["L-BFGS", "GN"], colors):
    mtime = np.array(np.median(results[k]["times"], axis=1))
    avg = np.array(np.mean(results[k]["times"], axis=1))
    q1 = np.array(np.quantile(results[k]["times"], 0.10, axis=1))
    q3 = np.array(np.quantile(results[k]["times"], 0.90, axis=1))
    plt.plot(horizons, mtime, label=k, color=color)
    plt.fill_between(horizons, q1, q3, alpha=0.2, color=color)
    # plt.plot(horizons, avg, ':', color=color, linewidth=0.5)
plt.ylim(0, None)
plt.legend(loc="upper left")
plt.title(r"Effect of horizon length on solver performance")
plt.xlabel(r"Horizon length")
plt.ylabel(r"Run time $[\mathrm{ms}]$")
plt.tight_layout()
plt.savefig(f"gn-chain-compare-N-horiz-confidence-parallel.pdf")
plt.show()
