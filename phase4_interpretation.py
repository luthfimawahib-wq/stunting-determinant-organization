import os
import numpy as np
import pandas as pd

HYPOTHESES = [
    dict(id="H1",
         domain="Determinants at birth",
         statement=("The birth-related attribution structure will show relative "
                    "stability across developmental stages."),
         expected="stable"),
    dict(id="H2",
         domain="Socioeconomic and assets",
         statement=("The socioeconomic attribution share will be higher in the "
                    "older developmental stage."),
         expected="higher"),
    dict(id="H3",
         domain="Maternal",
         statement=("The maternal attribution share will be lower in the older "
                    "developmental stage."),
         expected="lower"),
]

STABILITY_BAND = 3.0


def evaluate(shift_mean, consistency, n_determinants):
    rows = []
    for h in HYPOTHESES:
        d = h["domain"]
        value = shift_mean.get(d, np.nan)
        cons = consistency.get(d, "unknown")
        n = int(n_determinants.get(d, 0))
        if np.isnan(value) or n == 0:
            verdict, reason = "Not evaluable", "domain absent from the analytic space"
        elif n < 2 or cons != "consistent":
            verdict = "Not evaluable"
            reason = ("rests on a single determinant" if n < 2
                      else "direction inconsistent across surveys")
        elif h["expected"] == "stable":
            within = abs(value) <= STABILITY_BAND
            verdict = "Supported" if within else "Not supported"
            reason = (f"absolute difference {abs(value):.1f} share points "
                      f"{'within' if within else 'beyond'} the "
                      f"{STABILITY_BAND:.0f}-point stability band")
        elif h["expected"] == "higher":
            verdict = "Supported" if value > 0 else "Not supported"
            reason = f"difference {value:+.1f} share points"
        else:
            verdict = "Supported" if value < 0 else "Not supported"
            reason = f"difference {value:+.1f} share points"
        rows.append(dict(hypothesis=h["id"], domain=d, statement=h["statement"],
                         difference_share_points=(None if np.isnan(value)
                                                  else round(float(value), 2)),
                         direction_consistency=cons, n_determinants=n,
                         verdict=verdict, basis=reason))
    return pd.DataFrame(rows)


def evidence_tier(shift_mean, consistency, agreement, boundary_stable,
                  n_determinants, magnitude_band=5.0):
    rows = []
    for d in shift_mean.index:
        v = float(shift_mean[d])
        cons = consistency.get(d, "unknown")
        agree = agreement.get(d, "unknown")
        stable = boundary_stable.get(d, "unknown")
        n = int(n_determinants.get(d, 0))
        if n < 2:
            tier, note = "Insufficient basis", "domain rests on fewer than two determinants"
        elif cons != "consistent":
            tier, note = "Insufficient evidence", "direction not consistent across surveys"
        elif abs(v) < magnitude_band or agree != "yes":
            tier, note = "Boundary case", "small magnitude or architectures disagree"
        elif stable == "yes":
            tier, note = "Robust", "robust in direction, magnitude, and classification"
        else:
            tier, note = "Directionally robust", (
                "robust in direction and cross-survey consistency, sensitive in "
                "magnitude to boundary-determinant placement")
        rows.append(dict(domain=d, difference_share_points=round(v, 2),
                         direction_consistency=cons, architectures=agree,
                         boundary_stable=stable, n_determinants=n,
                         evidence_tier=tier, note=note))
    out = pd.DataFrame(rows)
    order = {"Robust": 0, "Directionally robust": 1, "Boundary case": 2,
             "Insufficient evidence": 3, "Insufficient basis": 4}
    return out.assign(_o=out.evidence_tier.map(order)).sort_values(
        ["_o", "difference_share_points"], key=lambda s: s if s.name == "_o" else s.abs(),
        ascending=[True, False]).drop(columns="_o").reset_index(drop=True)


def write(hyp, tier, outdir):
    os.makedirs(outdir, exist_ok=True)
    hyp.to_csv(os.path.join(outdir, "hypotheses.csv"), index=False)
    tier.to_csv(os.path.join(outdir, "evidence_tiers.csv"), index=False)
