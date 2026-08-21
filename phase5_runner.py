import os
import numpy as np
import pandas as pd
import phase3_shares as D
import phase5_robust as R

CONFIG = dict(
    matrix_dir="output_phase1/matrix",
    kamus_path="output_phase2/domain_dictionary.csv",
    phase3_dir="output_phase3",
    output_dir="output_phase5",
    headline_model="xgb",
    check_model="rf",
    B=200,
    top_k=3,
    loso_ratio_min=0.5,
    seed=42,
)


def main(cfg):
    print("=" * 76)
    print("RUNNER FASE 5 - Robustness of Organizational Findings")
    print("=" * 76)
    od = cfg["output_dir"]
    os.makedirs(od, exist_ok=True)
    primer, sens, border = D.load_kamus(cfg["kamus_path"])
    md, model = cfg["matrix_dir"], cfg["headline_model"]
    full = D.full_intersection_feats(md, model)
    feats = [f for f in full if primer.get(f) is not None]
    print(f"Ruang headline: {len(feats)} determinan pada irisan penuh.")

    f3 = cfg["phase3_dir"]
    klas = pd.read_csv(os.path.join(f3, "klasifikasi_headline.csv"), index_col=0)
    lm = pd.read_csv(os.path.join(f3, "lintas_model.csv"), index_col=0).iloc[:, 0]
    sensi = pd.read_csv(os.path.join(f3, "sensitivitas_variabel_batas.csv"), index_col=0)
    kokoh = [d for d in klas.index if str(klas.loc[d, "kelas_kokoh"]) != "-"]
    print(f"Domain berkelas kokoh dari Fase 3: {kokoh}")

    print(f"\n[1] Bootstrap dalam sumber (B={cfg['B']}), presisi selang pergeseran")
    ci_dom, _ = R.bootstrap_precision(md, model, primer, feats, B=cfg["B"],
                                      seed=cfg["seed"], level="domain")
    ci_dom.to_csv(os.path.join(od, "presisi_bootstrap_domain.csv"))
    print(ci_dom.to_string())
    print("  CAKUPAN: ketidakpastian pencuplikan DALAM sumber; bukan penyaring kelas.")
    ci_det, _ = R.bootstrap_precision(md, model, primer, feats, B=cfg["B"],
                                      seed=cfg["seed"], level="determinant")
    ci_det.to_csv(os.path.join(od, "presisi_bootstrap_determinan.csv"))

    print("\n[2] Tinggalkan satu sumber")
    lo = R.loso(md, model, primer, feats)
    lo.to_csv(os.path.join(od, "loso_domain.csv"))
    print(lo.to_string())
    for d in kokoh:
        st = lo.loc[d, "arah_bertahan"]
        print(f"  {d}: arah bertahan = {st}; besaran tersisa pada subset terburuk "
              f"{lo.loc[d,'rasio_magnitudo_min']:.2f} kali nilai penuh "
              f"({lo.loc[d,'magnitudo_min']:.4f} dari {abs(lo.loc[d,'penuh_3_sumber']):.4f})")

    print(f"\n[3] Stabilitas peringkat determinan penyumbang (B={cfg['B']}, top-{cfg['top_k']})")
    rs = R.determinant_rank_stability(md, model, primer, feats, kokoh, B=cfg["B"],
                                      seed=cfg["seed"], top_k=cfg["top_k"])
    for d, t in rs.items():
        safe = d.split(" (")[0].replace(" & ", "_").replace(" ", "_").lower()
        t.to_csv(os.path.join(od, f"stabilitas_peringkat_{safe}.csv"))
        print(f"\n  {d}:")
        print(t.head(6).to_string())

    print("\n[4] Ruang fitur: irisan penuh dibanding irisan per sumber")
    fs = R.feature_space_comparison(md, model, primer, feats)
    fs.to_csv(os.path.join(od, "ruang_fitur.csv"))
    print(fs.to_string())

    print("\n[5] Rekap ketahanan")
    rec = R.robustness_recap(kokoh, klas, lm, lo, fs, rs, sensi, top_k=cfg["top_k"],
                             loso_ratio_min=cfg["loso_ratio_min"])
    rec.to_csv(os.path.join(od, "rekap_ketahanan.csv"))
    print(rec.to_string())

    print("\n[6] Gambar")
    fig_loso(lo, kokoh, os.path.join(od, "gambar_loso.png"))
    fig_recap(rec, os.path.join(od, "gambar_rekap_ketahanan.png"))
    print("    gambar_loso.png, gambar_rekap_ketahanan.png")
    print(f"\nSELESAI. Artefak di {od}/")
    print("  Bootstrap refit tidak dijalankan: pertanyaannya sudah dijawab uji lintas arsitektur.")


def fig_loso(lo, kokoh, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    doms = list(lo.sort_values("penuh_3_sumber").index)
    cols = ["penuh_3_sumber"] + [c for c in lo.columns if c.startswith("tanpa_")]
    NAVY, TEAL, GREY = "#1F3864", "#2E9E8F", "#9AA5B1"
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.axvline(0, color="#444444", lw=1.2)
    for i, d in enumerate(doms):
        subs = [float(lo.loc[d, c]) for c in cols[1:]]
        ax.plot(subs, [i] * len(subs), "o", ms=7, color=GREY, alpha=0.9,
                label="Tinggalkan satu sumber" if i == 0 else None)
        m = float(lo.loc[d, "penuh_3_sumber"])
        ax.plot([m], [i], "D", ms=11, color=(TEAL if m >= 0 else NAVY),
                label="Tiga sumber penuh" if i == 0 else None)
    ax.set_yticks(range(len(doms)))
    ax.set_yticklabels([d.replace(" (Penyakit Infeksi)", "") +
                        ("  *" if d in kokoh else "") for d in doms], fontsize=10)
    ax.set_xlabel("Pergeseran pangsa atribusi domain (poin pangsa)", fontsize=10)
    ax.set_title("Ketahanan terhadap sumber: pergeseran ketika satu sumber ditinggalkan\n"
                 "(tanda bintang = domain berkelas kokoh)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_recap(rec, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"bertahan": "#2E9E8F", "sebagian": "#E0A458", "tidak": "#B4001E",
           "takada": "#D5DAE1"}
    MARK = {"bertahan": "bertahan", "sebagian": "sebagian", "tidak": "tidak", "takada": "-"}
    nr, nc = rec.shape
    fig, ax = plt.subplots(figsize=(2.6 + 2.9 * nc, 0.62 * nr + 1.7))
    for i in range(nr):
        for j in range(nc):
            v = str(rec.iloc[i, j])
            ax.add_patch(plt.Rectangle((j, nr - i - 1), 1, 1,
                                       facecolor=COL.get(v, "#D5DAE1"), edgecolor="white", lw=2))
            ax.text(j + 0.5, nr - i - 0.5, MARK.get(v, v), ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")
    ax.set_xlim(0, nc); ax.set_ylim(0, nr)
    ax.set_xticks([j + 0.5 for j in range(nc)])
    ax.set_xticklabels([c.replace(" & ", "\n& ") for c in rec.columns],
                       fontsize=10, fontweight="bold")
    ax.set_yticks([nr - i - 0.5 for i in range(nr)])
    ax.set_yticklabels(rec.index, fontsize=10)
    ax.set_xticks([], minor=True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Rekap ketahanan temuan organisasi", fontsize=12.5, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main(CONFIG)
