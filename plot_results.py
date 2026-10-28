"""Make figures for the report from results.csv (+ heat map from final_downsampled.csv)."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results.csv")))
seq = float(next(r for r in rows if r["algorithm"] == "Sequential")["time_s"])
algos = []
for r in rows:
    if r["algorithm"] != "Sequential" and r["algorithm"] not in algos:
        algos.append(r["algorithm"])

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for a in algos:
    d = [r for r in rows if r["algorithm"] == a]
    t = [int(r["threads"]) for r in d]
    ax[0].plot(t, [float(r["time_s"]) for r in d], "o-", label=a)
    ax[1].plot(t, [float(r["speedup"]) for r in d], "o-", label=a)
    ax[2].plot(t, [100 * float(r["efficiency"]) for r in d], "o-", label=a)
ax[0].axhline(seq, color="k", ls="--", label="Sequential")
tmax = max(int(r["threads"]) for r in rows)
ax[1].plot([1, tmax], [1, tmax], "k--", label="Ideal")
for a_, title, yl in zip(ax, ["Execution time", "Speedup", "Efficiency"], ["seconds", "x", "%"]):
    a_.set_title(title); a_.set_xlabel("threads"); a_.set_ylabel(yl); a_.grid(alpha=.3)
ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig("benchmark.png", dpi=150)
print("saved benchmark.png")

if os.path.exists("final_downsampled.csv"):
    import numpy as np
    from matplotlib.colors import LogNorm
    m = np.loadtxt("final_downsampled.csv", delimiter=",")
    plt.figure(figsize=(6, 5))
    plt.imshow(m, cmap="inferno", norm=LogNorm(vmin=30, vmax=m.max()))
    plt.colorbar(label="temperature (log scale)")
    plt.title("Heat map after 100 iterations (every 10th cell)")
    plt.tight_layout(); plt.savefig("heatmap.png", dpi=150)
    print("saved heatmap.png")
