"""Generate LaTeX tables, figures and scalar macros from Aishwarya's result CSVs.
No number in the report or paper is typed by hand."""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TAB = os.path.join(HERE, "report", "tables")
FIG = os.path.join(HERE, "report", "figs")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
PAL = ["#1b3a5b", "#b5341f", "#2f7d32", "#c89b2c", "#5c457f", "#777777"]
plt.rcParams.update({"figure.figsize": (5, 3.1), "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.bbox": "tight", "savefig.dpi": 200})


def w(n, t): open(os.path.join(TAB, n), "w").write(t)
def f3(x):
    try: return f"{float(x):.3f}"
    except Exception: return str(x)
def tbl(cs, h, rows, n):
    w(n, "\\begin{tabular}{%s}\n\\toprule\n%s \\\\\n\\midrule\n%s\n\\bottomrule\n\\end{tabular}\n"
      % (cs, h, "\n".join(rows)))
R = lambda f: pd.read_csv(os.path.join(RES, f))
M = {}

meta = json.load(open(os.path.join(RES, "meta.json")))
M["Ntrain"] = f"{meta['n_train']:,}"; M["Ntest"] = f"{meta['n_test']:,}"
M["AttackRate"] = f"{meta['attack_rate']*100:.1f}\\%"; M["Trigger"] = meta["trigger_id"].replace("0x", "0x")
M["Synthetic"] = "yes" if meta["synthetic"] else "no"

# detection
d = R("detection.csv")
rows = [f"{r['model']} & {f3(r['accuracy'])} & {f3(r['balanced_accuracy'])} & "
        f"{f3(r['precision_attack'])} & {f3(r['recall_attack'])} & {f3(r['f1_attack'])} \\\\"
        for _, r in d.iterrows()]
tbl("lrrrrr", "Detector & Acc. & Bal.\\ acc. & Prec. & Recall & $F_1$", rows, "tab_detection.tex")
cnn = d[d.model == "1D-CNN"].iloc[0]; M["CNNfone"] = f3(cnn["f1_attack"])

# backdoor sweep
b = R("backdoor.csv")
rows = [f"{r['poison_rate']*100:.0f}\\% & {f3(r['clean_acc'])} & {f3(r['asr'])} \\\\" for _, r in b.iterrows()]
tbl("lrr", "Poison rate & Clean acc. & ASR", rows, "tab_backdoor.tex")
fig, ax = plt.subplots()
ax.plot(b["poison_rate"]*100, b["asr"], "o-", color=PAL[1], label="ASR (triggered attacks $\\to$ Normal)")
ax.plot(b["poison_rate"]*100, b["clean_acc"], "s--", color=PAL[0], label="clean accuracy")
ax.set_xlabel("poison rate (\\%)"); ax.set_ylabel("rate"); ax.legend(fontsize=7)
fig.savefig(os.path.join(FIG, "backdoor.pdf")); plt.close(fig)
hon = b[b.poison_rate == 0.0].iloc[0]; M["HonestASR"] = f3(hon["asr"])
# minimum poison for high ASR
hi = b[b.asr >= 0.9]
M["MinPoison"] = (f"{hi.iloc[0]['poison_rate']*100:.0f}\\%" if len(hi) else "n/a")

# dp backdoor
dp = R("dp_backdoor.csv")
fin = dp[np.isfinite(dp["eps"])]
rows = [f"{r['eps']:.1f} & {f3(r['f1'])} & {f3(r['asr'])} & {f3(r['mia_auc'])} \\\\" for _, r in fin.iterrows()]
npr = dp[~np.isfinite(dp["eps"])]
if len(npr):
    r = npr.iloc[0]; rows.append(f"$\\infty$ & {f3(r['f1'])} & {f3(r['asr'])} & {f3(r['mia_auc'])} \\\\")
tbl("lrrr", "$\\varepsilon$ & $F_1$ & ASR & MIA AUC", rows, "tab_dpbackdoor.tex")
fig, ax = plt.subplots()
ax.plot(fin["eps"], fin["asr"], "o-", color=PAL[1], label="ASR under DP-SGD")
if len(npr): ax.axhline(npr.iloc[0]["asr"], ls="--", color=PAL[5], label="ASR, no DP")
ax.plot(fin["eps"], fin["f1"], "s-", color=PAL[0], label="detection $F_1$")
ax.set_xlabel("privacy budget $\\varepsilon$"); ax.set_ylabel("rate"); ax.legend(fontsize=7)
fig.savefig(os.path.join(FIG, "dp_backdoor.pdf")); plt.close(fig)

# dp kmeans
try:
    k = R("dp_kmeans.csv")
    tbl("lr", "$\\varepsilon$ & ARI vs non-private",
        [f"{r['eps']:.1f} & {f3(r['ari_vs_nonprivate'])} \\\\" for _, r in k.iterrows()], "tab_dpkmeans.tex")
except Exception: pass

# localdp
L = R("localdp.csv")
fig, ax = plt.subplots()
for i, mech in enumerate([m for m in L.mechanism.unique() if m != "none (baseline)"]):
    s = L[L.mechanism == mech].sort_values("eps_local")
    ax.plot(s["eps_local"], s["f1"], "o-", color=PAL[i], label=mech)
base = L[L.mechanism == "none (baseline)"]
if len(base): ax.axhline(base.iloc[0]["f1"], ls="--", color=PAL[5], label="no LDP")
ax.set_xlabel("local budget $\\varepsilon$"); ax.set_ylabel("detection $F_1$"); ax.legend(fontsize=7)
fig.savefig(os.path.join(FIG, "localdp.pdf")); plt.close(fig)
ln = R("ldp_nb.csv")
tbl("lr", "$\\varepsilon_{local}$ & LDP-NB $F_1$",
    [f"{r['eps_local']:.1f} & {f3(r['f1'])} \\\\" for _, r in ln.iterrows()], "tab_ldpnb.tex")

# output mod
o = R("output_mod.csv")
tbl("lrr", "Released output & MIA AUC & MIA adv.",
    [f"{r['output_mode']} & {f3(r['mia_auc'])} & {f3(r['mia_adv'])} \\\\" for _, r in o.iterrows()], "tab_outmod.tex")

# compressive
c = R("compressive.csv")
fig, ax = plt.subplots()
for i, meth in enumerate(c.method.unique()):
    s = c[c.method == meth].sort_values("p")
    ax.plot(s["p"], s["f1"], "o-", color=PAL[i], label=f"{meth} $F_1$")
    ax.plot(s["p"], s["recon_err"], "s--", color=PAL[i], alpha=0.6)
ax.set_xlabel("compressed dimension $p$"); ax.set_ylabel("$F_1$ (solid) / recon.\\ err (dashed)")
ax.legend(fontsize=7); fig.savefig(os.path.join(FIG, "compressive.pdf")); plt.close(fig)
tbl("lrrr", "Method & $p$ & detection $F_1$ & recon.\\ err.",
    [f"{r['method']} & {int(r['p'])} & {f3(r['f1'])} & {f3(r['recon_err'])} \\\\"
     for _, r in c[c.p == sorted(c.p.unique())[len(c.p.unique())//2]].iterrows()], "tab_comp.tex")

# kanon
ka = R("kanon.csv").iloc[0]
M["KanonK"] = str(int(ka["achieved_k_binned"])); M["FracUniqID"] = f3(ka["frac_unique_exact_id"])
M["NdistinctID"] = str(int(ka["n_distinct_ids"]))

# coverage matrix (qualitative synthesis anchored to the measurements)
cov = [
    ("Backdoor poisoning (integrity)", "DP-SGD training", "\\pmark", "Clipping+noise limit poison influence; measured by ASR-vs-$\\varepsilon$"),
    ("Membership inference", "DP-SGD training", "\\cmark", "Bounds per-window influence"),
    ("Membership inference", "Output modification", "\\pmark", "Removes the confidence signal"),
    ("Telemetry disclosure at collection", "Local DP perturbation", "\\cmark", "No trusted curator"),
    ("Raw-window reconstruction", "Compressive PCA/DCA", "\\cmark", "Low-dim code discards raw signal"),
    ("Frame re-identification", "CAN-ID $k$-anonymity (analytical)", "\\pmark", "Stream, not a classic table"),
]
tbl("p{3.5cm}p{3.0cm}cp{4.4cm}", "Threat & Mechanism & Stops? & Note",
    [f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in cov], "tab_coverage.tex")

with open(os.path.join(TAB, "macros_ia2.tex"), "w") as f:
    for k, v in M.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("Aishwarya IA-2 tables/figures/macros written.")
