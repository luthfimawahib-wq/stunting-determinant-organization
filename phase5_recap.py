import os
import glob
import numpy as np
import pandas as pd

CONFIG = dict(
    phase3_dir="output_phase3",
    phase5_dir="output_phase5",
    loso_ratio_min=0.5,
    top_k=3,
)


def _cell(v, ok="bertahan", part="sebagian", no="tidak"):
    return v


def main(cfg):
    f3, f5 = cfg["phase3_dir"], cfg["phase5_dir"]
    klas = pd.read_csv(os.path.join(f3, "klasifikasi_headline.csv"), index_col=0)
    lm = pd.read_csv(os.path.join(f3, "lintas_model.csv"), index_col=0).iloc[:, 0]
    sensi = pd.read_csv(os.path.join(f3, "sensitivitas_variabel_batas.csv"), index_col=0)
    lo = pd.read_csv(os.path.join(f5, "loso_domain.csv"), index_col=0)
    fs = pd.read_csv(os.path.join(f5, "ruang_fitur.csv"), index_col=0)
    domains = [d for d in klas.index if str(klas.loc[d, "kelas_kokoh"]) != "-"]
    print(f"Domain berkelas kokoh: {domains}")

    rows = {}
    rows["Konsistensi arah tiga sumber"] = {
        d: ("bertahan" if str(klas.loc[d, "konsistensi"]) == "konsisten" else "tidak") for d in domains}
    rows["Kesepakatan tiga metode batas"] = {
        d: ("bertahan" if str(klas.loc[d, "kesepakatan"]) == "kokoh" else "sebagian") for d in domains}
    rows["Replikasi dua arsitektur model"] = {
        d: ("bertahan" if str(lm.get(d, "")) == "ya" else "tidak") for d in domains}
    def loso_cell(d):
        if str(lo.loc[d, "arah_bertahan"]) != "ya":
            return "tidak"
        r = lo.loc[d, "rasio_magnitudo_min"]
        return "takada" if pd.isna(r) else ("bertahan" if r >= cfg["loso_ratio_min"] else "sebagian")
    rows["Tinggalkan satu sumber"] = {d: loso_cell(d) for d in domains}
    rows["Ruang fitur (penuh vs per sumber)"] = {
        d: ("bertahan" if str(fs.loc[d, "arah_sama"]) == "ya" else "tidak") for d in domains}
    rows["Penempatan variabel batas"] = {
        d: ("bertahan" if str(sensi.loc[d, "stabil"]) == "ya" else "sebagian") for d in domains}

    def rank_cell(d):
        safe = d.split(" (")[0].replace(" & ", "_").replace(" ", "_").lower()
        p = os.path.join(f5, f"stabilitas_peringkat_{safe}.csv")
        if not os.path.exists(p):
            return "takada"
        t = pd.read_csv(p, index_col=0)
        col = [c for c in t.columns if c.startswith("top_")]
        if not len(t) or not col:
            return "takada"
        v = float(t.iloc[0][col[0]])
        return "bertahan" if v >= 0.8 else ("sebagian" if v >= 0.5 else "tidak")
    rows[f"Stabilitas penyumbang utama (top-{cfg['top_k']})"] = {d: rank_cell(d) for d in domains}

    pbob = os.path.join(f5, "bobot_perbandingan.csv")
    if os.path.exists(pbob):
        bob = pd.read_csv(pbob, index_col=0)
        def bob_cell(d):
            if d not in bob.index:
                return "takada"
            if str(bob.loc[d, "arah_sama"]) != "ya":
                return "tidak"
            r = bob.loc[d, "rasio_besaran"]
            if pd.isna(r):
                return "takada"
            return "bertahan" if 0.75 <= float(r) <= 1.25 else "sebagian"
        rows["Pembobotan survei"] = {d: bob_cell(d) for d in domains}
        print("Baris sensitivitas berbobot survei ditambahkan.")
    else:
        print("PERINGATAN: bobot_perbandingan.csv tidak ditemukan; rekap tanpa baris pembobotan.")

    rec = pd.DataFrame(rows).T[domains]
    rec.to_csv(os.path.join(f5, "rekap_ketahanan.csv"))
    print("\n" + rec.to_string())

    fig_recap(rec, os.path.join(f5, "gambar_rekap_ketahanan.png"))
    print(f"\nTersimpan: rekap_ketahanan.csv dan gambar_rekap_ketahanan.png di {f5}/")


def fig_recap(rec, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"bertahan": "#2E9E8F", "sebagian": "#E0A458", "tidak": "#B4001E", "takada": "#D5DAE1"}
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
    ax.set_xticklabels([c.replace(" & ", "\n& ") for c in rec.columns], fontsize=10, fontweight="bold")
    ax.set_yticks([nr - i - 0.5 for i in range(nr)])
    ax.set_yticklabels(rec.index, fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Rekap ketahanan temuan organisasi", fontsize=12.5, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main(CONFIG)
