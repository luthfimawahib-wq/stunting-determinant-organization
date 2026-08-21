import os
import pandas as pd
import phase4_interpretation as P4

CONFIG = dict(
    phase2_dir="output_phase2",
    phase3_dir="output_phase3",
    output_dir="output_phase4",
)

DOMAIN_ID_TO_EN = {
    "Biologis Kelahiran": "Determinants at birth",
    "Maternal": "Maternal",
    "Sosioekonomi & Aset": "Socioeconomic and assets",
    "Sanitasi & Lingkungan": "Environment and sanitation",
    "Layanan Kesehatan": "Health services",
    "Pemberian Makan": "Feeding",
    "Morbiditas Anak (Penyakit Infeksi)": "Child morbidity",
}


def _en(index):
    return [DOMAIN_ID_TO_EN.get(x, x) for x in index]


def main(cfg):
    print("=" * 74)
    print("PHASE 4. INTERPRETATION: PRESPECIFIED HYPOTHESES AND EVIDENCE TIERS")
    print("=" * 74)
    f2, f3 = cfg["phase2_dir"], cfg["phase3_dir"]

    klas = pd.read_csv(os.path.join(f3, "klasifikasi_headline.csv"), index_col=0)
    klas.index = _en(klas.index)
    shift = pd.read_csv(os.path.join(f3, "pergeseran_headline.csv"), index_col=0)
    shift.index = _en(shift.index)
    shift_mean = shift.mean(axis=1) * 100.0
    shift_mean = shift_mean[[d for d in shift_mean.index
                             if d in DOMAIN_ID_TO_EN.values()]]

    cons = klas["konsistensi"].map(
        {"konsisten": "consistent", "tak-konsisten": "inconsistent"}).fillna("unknown")

    lm_path = os.path.join(f3, "lintas_model.csv")
    agree = pd.Series(dtype=object)
    if os.path.exists(lm_path):
        lm = pd.read_csv(lm_path, index_col=0)
        lm.index = _en(lm.index)
        agree = lm.iloc[:, 0].map({"ya": "yes", "tidak": "no"}).fillna("unknown")

    sens_path = os.path.join(f3, "sensitivitas_variabel_batas.csv")
    stable = pd.Series(dtype=object)
    if os.path.exists(sens_path):
        sn = pd.read_csv(sens_path, index_col=0)
        sn.index = _en(sn.index)
        stable = sn["stabil"].map({"ya": "yes"}).fillna("no")

    comp = pd.read_csv(os.path.join(f2, "domain_comparability.csv"))
    comp = comp[comp.domain != "Total determinants"]
    n_det = dict(zip(comp.domain, comp.all_three))

    hyp = P4.evaluate(shift_mean, cons.to_dict(), n_det)
    tier = P4.evidence_tier(shift_mean, cons.to_dict(), agree.to_dict(),
                            stable.to_dict(), n_det)

    print("\nPrespecified hypotheses")
    print(hyp[["hypothesis", "domain", "difference_share_points",
               "verdict", "basis"]].to_string(index=False))
    print("\nEvidence tiers")
    print(tier[["domain", "difference_share_points", "evidence_tier",
                "note"]].to_string(index=False))

    P4.write(hyp, tier, cfg["output_dir"])
    print(f"\nWritten to {cfg['output_dir']}/: hypotheses.csv, evidence_tiers.csv")
    print("Hypotheses are evaluated against relative attribution share, not against "
          "causal magnitude.")


if __name__ == "__main__":
    main(CONFIG)
