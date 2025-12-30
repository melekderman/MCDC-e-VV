import os
import re
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl


# get files and check if they exist
ref_path = "reference.npz"
h5_path  = "answer.h5"

if not os.path.exists(ref_path):
    print("ERROR: reference.npz not found")
    raise SystemExit(1)

if not os.path.exists(h5_path):
    print("ERROR: answer.h5 not found")
    raise SystemExit(1)

# find input.py
input_path = "input.py"
if not os.path.exists(input_path):
    print("ERROR: input.py not found (./input.py)")
    raise SystemExit(1)

# read problem specific constants from input.py
txt = open(input_path, "r", encoding="utf-8").read()

def _find(name):
    m = re.search(rf"^\s*{name}\s*=\s*(.+?)\s*(#.*)?$", txt, re.M)
    if not m:
        print(f"ERROR: {name} not found in input.py")
        raise SystemExit(1)
    return m.group(1).strip()

MATERIAL_SYMBOL = _find("MATERIAL_SYMBOL").strip("\"' ")
ENERGY          = float(eval(_find("ENERGY"), {}, {}))          # eV
ANGLE           = float(eval(_find("ANGLE"), {}, {}))           # deg
CSDA_RANGE      = float(eval(_find("CSDA_RANGE"), {}, {}))      # g/cm2
RHO_G_CM3       = float(eval(_find("RHO_G_CM3"), {}, {}))       # g/cm3
N_PARTICLES     = float(eval(_find("N_PARTICLES"), {}, {}))

E_MeV = ENERGY / 1e6
L = CSDA_RANGE / RHO_G_CM3  # cm

# load reference arrays
ref = np.load(ref_path)

theo_fmr  = ref["fmr_theo_tiger"].astype(float)
theo_edep = ref["edep_theo_tiger"].astype(float)

if "fmr_exp_lw_A" in ref:
    exp_fmr_A  = ref["fmr_exp_lw_A"].astype(float)
    exp_edep_A = ref["edep_exp_lw_A"].astype(float)

if "fmr_exp_lw_B" in ref:
    exp_fmr_B  = ref["fmr_exp_lw_B"].astype(float)
    exp_edep_B = ref["edep_exp_lw_B"].astype(float)

else:
    exp_fmr  = ref["fmr_exp_lw"].astype(float)
    exp_edep = ref["edep_exp_lw"].astype(float)

# load answer.h5 MCDC results

with h5py.File(h5_path, "r") as f:
    z = f["tallies/mesh_tally_0/grid/z"][:].astype(float)
    dz = z[1:] - z[:-1]
    edep = f["tallies/mesh_tally_0/edep/mean"][:].astype(float)

z_centers = 0.5 * (z[:-1] + z[1:])
mcdc_fmr = z_centers / L
edep_mcdc = edep / RHO_G_CM3 / dz / 1e6  # MeV/g/cm^2

# plot
plt.figure(figsize=(14, 9), constrained_layout=True)

mpl.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 29,
    "axes.labelsize": 25,
    "legend.fontsize": 22,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
})

plt.minorticks_on()
plt.grid(True, which="major", linestyle="-", linewidth=0.8, color="#3B3A3AFF", alpha=0.5)
plt.grid(True, which="minor", linestyle=":", linewidth=0.6, color="#8A8686", alpha=0.7)

plt.plot(mcdc_fmr, edep_mcdc, label=f"MCDC Simulation", marker="o", markersize=9, linewidth=3.0)
plt.plot(theo_fmr, theo_edep, label="TIGER Theoretical", marker="s", markersize=9, linewidth=3.0)

if "exp_fmr_A" in locals():
    plt.plot(exp_fmr_A, exp_edep_A, label="Lockwood Experimental A", marker="D", markersize=9, linewidth=3.0)
if "exp_fmr_B" in locals():
    plt.plot(exp_fmr_B, exp_edep_B, label="Lockwood Experimental B", marker="*", markersize=9, linewidth=3.0)
if "exp_fmr" in locals():
    plt.plot(exp_fmr, exp_edep, label="Lockwood Experimental", marker="D", markersize=9, linewidth=3.0)

plt.xlim(0, 1)
plt.xlabel("Fraction of Mean Range", labelpad=15)
plt.ylabel("Energy Deposition (MeV/g/cm²)", labelpad=15)
plt.title(f"Energy Deposition of {E_MeV:g} MeV Electrons in {MATERIAL_SYMBOL} at {ANGLE:g}° Incidence", pad=20)
plt.legend()

# Save the figure 
# directory: verification/benchmark/continuous_energy/lockwood/results
E_tag = f"{E_MeV:g}MeV"
th_tag = f"th{int(round(ANGLE))}"
N_tag = f"{N_PARTICLES:.0e}".replace("+", "")
out_dir = os.path.normpath(os.path.join(os.getcwd(), "..", "..", "results"))
os.makedirs(out_dir, exist_ok=True)

out_name = f"fig_{MATERIAL_SYMBOL}_{E_tag}_{th_tag}_{N_tag}.png"
out_path = os.path.join(out_dir, out_name)

plt.savefig(out_path, dpi=400, bbox_inches="tight")
print("Saved:", out_path)
