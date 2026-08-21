import os
import json
import numpy as np
import pandas as pd
import phase3_shares as D

CONFIG = dict(
    matrix_dir="output_phase1/matrix",
    kamus_path="output_phase2/domain_dictionary.csv",
    output_dir="output_phase3",
    headline_model="xgb",
    check_model="rf",
    bootstrap_B=200,
    seed=42,
)


def run_layer(cfg, mapping, feats_fixed):
    md, model = cfg["matrix_dir"], cfg["headline_model"]
    shift_df, sb, st = D.shift_matrix(md, model, mapping, feats=feats_fixed)
    summary = D.summarize_domains(shift_df)
    fc = D.domain_feature_counts(feats_fixed, mapping) if feats_fixed is not None else None
    clsf, meta, boot_prop = D.classify(summary, md, model, mapping, B=cfg["bootstrap_B"],
                                       seed=cfg["seed"], feats_fixed=feats_fixed, fcounts=fc)
    return dict(shift=shift_df, share_bad=sb, share_bal=st, summary=summary,
                clsf=clsf, meta=meta, fcounts=fc, boot_prop=boot_prop)


def main(cfg):
    print("=" * 76)
    print("RUNNER FASE 3 - pangsa domain, pergeseran, kelas organisasi")
    print("=" * 76)
    od = cfg["output_dir"]
    os.makedirs(od, exist_ok=True)
    primer, sens, border = D.load_kamus(cfg["kamus_path"])
    model = cfg["headline_model"]
    md = cfg["matrix_dir"]
    print(f"Kamus: {len(set(primer.values()))} domain, {len(primer)} determinan, "
          f"{len(border)} variabel batas.")

    full = D.full_intersection_feats(md, model)
    unmapped = [f for f in full if primer.get(f) is None]
    if unmapped:
        print(f"  Fitur pada irisan penuh yang TIDAK dipetakan ke domain (dikeluarkan dari "
              f"pangsa dan dari penyebut kedua level): {unmapped}")
    per = {}
    for s in D.SOURCES:
        mb = D.load_matrix(md, s, "baduta", model)
        mt = D.load_matrix(md, s, "balita_tua", model)
        per[s] = D.cross_cohort_feats(mb, mt)
    union = sorted(set().union(*[set(D.feature_cols(D.load_matrix(md, s, c, model)))
                                 for s in D.SOURCES for c in D.COHORTS]))
    kb = pd.DataFrame({
        "ruang_uji_gabungan": D.domain_feature_counts(union, primer),
        "irisan_ssgi22": D.domain_feature_counts(per["ssgi22"], primer),
        "irisan_ssgi24": D.domain_feature_counts(per["ssgi24"], primer),
        "irisan_ski23": D.domain_feature_counts(per["ski23"], primer),
        "irisan_penuh": D.domain_feature_counts(full, primer),
    }).fillna(0).astype(int)
    kb["status"] = np.where(kb.irisan_penuh == 0, "tidak terbandingkan",
                            np.where(kb.irisan_penuh < 3, "basis sempit", "terbandingkan"))
    kb = kb.sort_values("ruang_uji_gabungan", ascending=False)
    kb.to_csv(os.path.join(od, "keterbandingan_domain.csv"))
    print(f"\n[0] Keterbandingan domain (irisan penuh = {len(full)} fitur):")
    print(kb.to_string())

    feats_mapped = [f for f in full if primer.get(f) is not None]
    print(f"\n[1] LAPIS HEADLINE: irisan penuh {len(full)} kolom, "
          f"{len(feats_mapped)} di antaranya determinan (sisanya poros struktural, "
          f"dikeluarkan dari pembilang dan penyebut). Model {model.upper()}")
    P = run_layer(cfg, primer, full)
    rows = []
    for s in D.SOURCES:
        for dom in P["summary"].index:
            rows.append(dict(sumber=s, domain=dom,
                             n_fitur=int(P["fcounts"].get(dom, 0)),
                             pangsa_baduta=round(float(P["share_bad"].loc[dom, s]), 4),
                             pangsa_balita_tua=round(float(P["share_bal"].loc[dom, s]), 4),
                             pergeseran=round(float(P["shift"].loc[dom, s]), 4)))
    pd.DataFrame(rows).to_csv(os.path.join(od, "pangsa_domain_headline.csv"), index=False)
    P["shift"].round(4).to_csv(os.path.join(od, "pergeseran_headline.csv"))
    P["summary"].round(4).to_csv(os.path.join(od, "ringkasan_domain_headline.csv"))
    P["clsf"].to_csv(os.path.join(od, "klasifikasi_headline.csv"))
    with open(os.path.join(od, "meta_batas.json"), "w") as f:
        json.dump(P["meta"], f, indent=2, default=float)
    P["boot_prop"].round(4).to_csv(os.path.join(od, "bootstrap_proporsi_kelas_headline.csv"))
    print(P["clsf"][["n_fitur", "magnitude", "arah", "konsistensi", "rentang_antar_sumber",
                     "kelas_persentil", "kelas_natural", "kelas_bootstrap",
                     "kesepakatan", "kelas_kokoh"]].to_string())
    print(f"  batas organisasi: k natural-break = {P['meta']['k_natural']}, "
          f"edges persentil = {[round(x, 4) for x in P['meta']['edges_persentil']]}")
    print("  CATATAN: selang bootstrap = presisi DALAM sumber, bukan penentu kelas; "
          "heterogenitas antar sumber dibaca dari rentang_antar_sumber.")

    print("\n[2] LAPIS DESKRIPTIF (irisan lintas kohort per sumber)")
    rows = []
    for s in D.SOURCES:
        r = D.shares_and_shift(md, s, model, primer)
        for dom in r["shift"].index:
            rows.append(dict(sumber=s, n_fitur_irisan=r["n_feats"], domain=dom,
                             pangsa_baduta=round(float(r["share_baduta"][dom]), 4),
                             pangsa_balita_tua=round(float(r["share_balita"][dom]), 4),
                             pergeseran=round(float(r["shift"][dom]), 4)))
    pd.DataFrame(rows).to_csv(os.path.join(od, "pangsa_domain_deskriptif.csv"), index=False)
    print(f"    tersimpan (irisan per sumber: { {s: len(per[s]) for s in D.SOURCES} })")

    print("\n[3] Domain tidak terbandingkan lintas kohort: komposisi baduta")
    comp_bad = D.cohort_only_composition(md, model, primer, cohort="baduta")
    comp_out = comp_bad.round(4).astype(object).where(comp_bad.notna(), "tidak diukur")
    comp_out.to_csv(os.path.join(od, "komposisi_baduta_takterbanding.csv"))
    tak = list(kb.index[kb.status == "tidak terbandingkan"])
    if tak:
        print(f"    domain: {tak}")
        avail = [d for d in tak if d in comp_bad.index]
        if avail:
            print(comp_out.loc[avail].to_string())
        print("    Dilaporkan deskriptif; kontras lintas kohort TIDAK dapat dihitung karena "
              "instrumen tidak mengukurnya pada balita tua (batas inferensi).")

    print(f"\n[4] Cek lintas model ({model.upper()} vs {cfg['check_model'].upper()})")
    full_rf = D.full_intersection_feats(md, cfg["check_model"])
    common = sorted(set(full) & set(full_rf))
    shift_rf, _, _ = D.shift_matrix(md, cfg["check_model"], primer, feats=common)
    shift_hl, _, _ = D.shift_matrix(md, model, primer, feats=common)
    xm = D.cross_model_check(shift_hl, shift_rf)
    xm.to_csv(os.path.join(od, "lintas_model.csv"))
    print(xm.to_string())

    print("\n[5] Sensitivitas variabel batas")
    S = run_layer(cfg, sens, full)
    S["clsf"].to_csv(os.path.join(od, "klasifikasi_sensitivitas.csv"))
    S["summary"].round(4).to_csv(os.path.join(od, "ringkasan_domain_sensitivitas.csv"))
    print("  klasifikasi pemetaan SENSITIVITAS (dilaporkan berdampingan):")
    print(S["clsf"][["n_fitur", "magnitude", "arah", "konsistensi",
                     "kesepakatan", "kelas_kokoh"]].to_string())
    cmp = pd.DataFrame({
        "kelas_primer": P["clsf"]["kelas_kokoh"],
        "kelas_sensitivitas": S["clsf"]["kelas_kokoh"].reindex(P["clsf"].index),
    })
    cmp["stabil"] = np.where(cmp.kelas_primer == cmp.kelas_sensitivitas, "ya", "BEDA")
    cmp.to_csv(os.path.join(od, "sensitivitas_variabel_batas.csv"))
    print(cmp.to_string())
    nb = int((cmp.stabil == "BEDA").sum())
    print(f"  domain berubah kelas: {nb}"
          + ("  -> klasifikasi INVARIANT terhadap penempatan variabel batas." if nb == 0
             else "  -> aturan diskordansi: laporkan kedua pemetaan, fokus bagian konsisten."))
    unchanged = [d for d in cmp.index
                 if int(P["fcounts"].get(d, 0)) == int(S["fcounts"].get(d, 0))
                 and cmp.loc[d, "stabil"] == "BEDA"]
    if unchanged:
        print(f"  CATATAN metode relatif: {unchanged} berpindah kelas meski komposisinya "
              f"tidak berubah; batas relatif bergeser karena domain lain berubah.")

    print("\n[6] LEVEL 2 (mekanisme): pergeseran determinan tunggal")
    det = D.determinant_shift_table(md, model, full, primer)
    det.to_csv(os.path.join(od, "pergeseran_determinan.csv"), float_format="%.6f")
    ver = D.verify_decomposition(det, P["summary"])
    ver.to_csv(os.path.join(od, "verifikasi_dekomposisi.csv"), index=False)
    print("  verifikasi dekomposisi (pergeseran domain = jumlah pergeseran determinannya):")
    print(ver.to_string(index=False))
    if (ver.cocok != "ya").any():
        print("  PERINGATAN: dekomposisi tidak cocok; periksa pemetaan.")
    for dom in P["clsf"].index[P["clsf"].kelas_kokoh != "-"]:
        sub = det[det.domain == dom].head(10)
        print(f"\n  {dom} (pergeseran domain {float(P['summary'].loc[dom,'shift_mean']):+.4f}):")
        print(sub[["shift_mean", "arah", "konsistensi", "kontribusi_persen"]].to_string())

    print("\n[7] Gambar")
    make_main_figure(P, os.path.join(od, "gambar_utama_pergeseran.png"), len(feats_mapped))
    print(f"    utama    -> gambar_utama_pergeseran.png")
    make_figure(P, os.path.join(od, "gambar_pendukung_pangsa.png"), len(feats_mapped))
    print(f"    pendukung-> gambar_pendukung_pangsa.png")
    make_mechanism_figure(det, P, os.path.join(od, "gambar_mekanisme_determinan.png"))
    print(f"    mekanisme-> gambar_mekanisme_determinan.png")
    print(f"\nSELESAI. Artefak di {od}/")


def make_main_figure(P, path, n_full):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    su = P["summary"].sort_values("shift_mean")
    domains = list(su.index)
    y = np.arange(len(domains))
    NAVY, TEAL, GREY = "#1F3864", "#2E9E8F", "#9AA5B1"
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.axvline(0, color="#444444", lw=1.2, zorder=1)
    for i, d in enumerate(domains):
        vals = [float(P["shift"].loc[d, s]) for s in D.SOURCES]
        ax.plot(vals, [i] * 3, "o", ms=7, color=GREY, alpha=0.85, zorder=2,
                label="Sumber (SSGI22, SSGI24, SKI23)" if i == 0 else None)
        m = float(su.loc[d, "shift_mean"])
        col = TEAL if m >= 0 else NAVY
        ax.plot([m], [i], "D", ms=11, color=col, zorder=3,
                label="Rerata lintas sumber" if i == 0 else None)
        ax.annotate(f"{m*100:+.1f} poin", (m, i), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=9, fontweight="bold", color=col)
    lbl = [f"{d.replace(' (Penyakit Infeksi)','')}  (n={int(P['fcounts'].get(d,0))})"
           for d in domains]
    ax.set_yticks(y); ax.set_yticklabels(lbl, fontsize=10)
    ax.set_xlabel("Pergeseran pangsa atribusi domain: balita tua dikurangi baduta "
                  "(poin pangsa)", fontsize=10)
    ax.set_title("Pergeseran organisasi determinan lintas tahap perkembangan\n"
                 f"(irisan penuh dua kohort dan tiga sumber; {n_full} determinan)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_ylim(-0.7, len(domains) - 0.15)
    xr = max(abs(P["shift"].values.min()), abs(P["shift"].values.max())) * 1.35
    ax.set_xlim(-xr, xr)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_mechanism_figure(det, P, path, top_n=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    doms = [d for d in P["clsf"].index if P["clsf"].loc[d, "kelas_kokoh"] != "-"]
    if not doms:
        doms = list(P["summary"].index[:2])
    fig, axes = plt.subplots(1, len(doms), figsize=(6.4 * len(doms), 4.4))
    if len(doms) == 1:
        axes = [axes]
    NAVY, TEAL, GREY = "#1F3864", "#2E9E8F", "#C3CAD3"
    for ax, dom in zip(axes, doms):
        sub = det[det.domain == dom].sort_values("shift_mean")
        sub = sub.tail(top_n) if len(sub) > top_n else sub
        y = np.arange(len(sub))
        cols = [TEAL if v >= 0 else NAVY for v in sub.shift_mean]
        edge = ["none" if c == "konsisten" else "#B4001E" for c in sub.konsistensi]
        ax.barh(y, sub.shift_mean.values * 100, color=cols, edgecolor=edge, linewidth=1.6)
        ax.axvline(0, color="#444444", lw=1)
        ax.set_yticks(y); ax.set_yticklabels(sub.index, fontsize=9)
        ax.set_title(f"{dom.replace(' (Penyakit Infeksi)','')}\n"
                     f"pergeseran domain {float(P['summary'].loc[dom,'shift_mean'])*100:+.1f} poin",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Pergeseran pangsa determinan (poin pangsa)", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Mekanisme: determinan penyumbang pergeseran domain "
                 "(tepi merah = arah tak konsisten lintas sumber)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_figure(P, path, n_full):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    domains = list(P["summary"].index)
    fig, axes = plt.subplots(1, len(D.SOURCES), figsize=(15, 5.8), sharey=True)
    NAVY, TEAL = "#1F3864", "#2E9E8F"
    for ax, s in zip(axes, D.SOURCES):
        yb = [float(P["share_bad"].loc[d, s]) for d in domains]
        yt = [float(P["share_bal"].loc[d, s]) for d in domains]
        x = np.arange(len(domains))
        ax.bar(x - 0.2, yb, 0.4, label="Baduta (0-23 bln)", color=NAVY)
        ax.bar(x + 0.2, yt, 0.4, label="Balita tua (24-59 bln)", color=TEAL)
        ax.set_title(s.upper(), fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        lbl = [d.replace(" & ", "\n& ").replace(" (Penyakit Infeksi)", "")
               + f"\n(n={int(P['fcounts'].get(d, 0))})" for d in domains]
        ax.set_xticklabels(lbl, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Pangsa atribusi domain (ternormalisasi)")
    axes[-1].legend(loc="upper right", fontsize=9)
    fig.suptitle("Pergeseran organisasi determinan lintas tahap perkembangan\n"
                 f"(irisan penuh dua kohort dan tiga sumber; {n_full} determinan)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main(CONFIG)
