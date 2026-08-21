import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import phase1_measure as u

CONFIG = dict(
    parquet_path="output_harmonisasi/stunting_harmonized.parquet",
    matriks_path="output_harmonisasi/matriks_ketersediaan.csv",
    schema_path="output_harmonisasi/skema_encoding.json",
    target="stunting_binary",
    group="id_ruta",
    kohort_col="kohort",
    source_col="source_flag",
    kohort_values=["baduta", "balita_tua"],
    sources=["ssgi22", "ssgi24", "ski23"],
    expected_source_rows=None,
    expected_total=None,
    models=["xgb", "rf"],
    matrix_sample=40000,
    conv_repeats=3,
    conv_top_k=10,
    shap_n_jobs=4,
    shap_backend="loky",
    output_dir="output_phase1",
    heritage_rank_dir="",
    source_flag_map={"ssgi22": "ssgi22", "ssgi24": "ssgi24", "ski23": "ski23"},
    only=None,
    resume=False,
    adopt_existing=False,
)


def _fail(msg):
    raise SystemExit("VALIDASI GAGAL: " + msg)


def tag_of(canon, coh, model):
    return f"{canon}_{coh}_{model}"


def manifest_path(matdir, tag):
    return os.path.join(matdir, tag + ".manifest.json")


def write_manifest(matdir, tag, cfg, extra):
    m = dict(tag=tag, code_version=u.CODE_VERSION, matrix_sample=cfg["matrix_sample"],
             conv_top_k=cfg["conv_top_k"], timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    m.update(extra)
    with open(manifest_path(matdir, tag), "w") as f:
        json.dump(m, f, indent=2)


def manifest_ok(matdir, tag, cfg):
    pq = os.path.join(matdir, tag + ".parquet")
    mp = manifest_path(matdir, tag)
    if not (os.path.exists(pq) and os.path.exists(mp)):
        return False
    try:
        m = json.load(open(mp))
    except Exception:
        return False
    return (m.get("code_version") == u.CODE_VERSION and
            m.get("matrix_sample") == cfg["matrix_sample"])


def auto_map_sources(df, cfg):
    sc = cfg["source_col"]
    if sc not in df.columns:
        _fail(f"kolom '{sc}' tidak ada. Kolom (30 pertama): {list(df.columns)[:30]}")
    counts = df[sc].value_counts().to_dict()
    print("  nilai source_flag ditemukan:", counts)
    if cfg["source_flag_map"]:
        mapping = {canon: lit for canon, lit in cfg["source_flag_map"].items()}
        for canon, lit in mapping.items():
            if lit not in counts:
                _fail(f"source_flag_map menyebut '{lit}' yang tidak ada di data.")
        return mapping
    mapping, used = {}, set()
    for canon, n in (cfg["expected_source_rows"] or {}).items():
        match = [v for v, c in counts.items() if c == n and v not in used]
        if len(match) == 1:
            mapping[canon] = match[0]
            used.add(match[0])
    if len(mapping) != 3:
        _fail("gagal memetakan source_flag otomatis; isi CONFIG['source_flag_map'].")
    print("  pemetaan source_flag (kanonik -> literal):", mapping)
    return mapping


def validate(df, mat, schema, cfg):
    print("VALIDASI SKEMA TERHADAP DATA NYATA")
    if cfg["expected_total"] and len(df) != cfg["expected_total"]:
        print(f"  PERINGATAN: total baris {len(df)} != {cfg['expected_total']} (dokumentasi).")
    else:
        print(f"  total baris {len(df)} sesuai.")
    for col in [cfg["target"], cfg["group"], cfg["kohort_col"], cfg["source_col"]]:
        if col not in df.columns:
            _fail(f"kolom wajib '{col}' tidak ada di Parquet.")
    kv = set(df[cfg["kohort_col"]].unique())
    for k in cfg["kohort_values"]:
        if k not in kv:
            _fail(f"nilai kohort '{k}' tidak ditemukan. Ada: {kv}")
    print(f"  kohort tersedia: {sorted(kv)}")
    tv = set(pd.unique(df[cfg["target"]].dropna()))
    if not tv <= {0, 1, 0.0, 1.0}:
        print(f"  PERINGATAN: target bukan biner murni: {tv}")
    print("  cek keberadaan kolom fitur beku per sel:")
    missing_any = False
    for canon in cfg["sources"]:
        for coh in cfg["kohort_values"]:
            feats = u.frozen_cell_features(mat, schema, canon, coh)
            miss = [f for f in feats if f not in df.columns]
            print(f"    {canon}-{coh}: {len(feats)} fitur, "
                  f"{'OK' if not miss else 'HILANG %d: %s' % (len(miss), miss)}")
            missing_any = missing_any or bool(miss)
    if missing_any:
        _fail("ada kolom fitur beku yang tidak ada di Parquet (lihat di atas).")
    print("  semua kolom fitur beku hadir.\n")


def build_plan(cfg):
    full = [(c, k, m) for c in cfg["sources"] for k in cfg["kohort_values"] for m in cfg["models"]]
    only = set(cfg["only"]) if cfg["only"] else None
    plan = []
    for c, k, m in full:
        if only is not None and tag_of(c, k, m) not in only:
            continue
        plan.append((c, k, m))
    return plan


def assemble_summary(cfg, matdir, perfdir, outdir):
    rows = []
    for c in cfg["sources"]:
        for k in cfg["kohort_values"]:
            for m in cfg["models"]:
                tag = tag_of(c, k, m)
                mp = manifest_path(matdir, tag)
                if not os.path.exists(mp):
                    continue
                man = json.load(open(mp))
                rows.append(dict(sel=f"{c}-{k}", model=m,
                                 n=man.get("n"), n_shap_rows=man.get("n_shap_rows"),
                                 n_fitur=man.get("n_features"), auc_cv=man.get("auc_cv"),
                                 konv_peringkat=man.get("konv_peringkat")))
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(outdir, "phase1_summary.csv"), index=False)
    return len(rows)


def adopt_existing(cfg, matdir, rankdir, convdir):
    print("ADOPSI BERKAS LAMA (mengasumsikan matriks dibuat versi kode saat ini):")
    n = 0
    for c in cfg["sources"]:
        for k in cfg["kohort_values"]:
            for m in cfg["models"]:
                tag = tag_of(c, k, m)
                pq = os.path.join(matdir, tag + ".parquet")
                if not (os.path.exists(pq) and not os.path.exists(manifest_path(matdir, tag))):
                    continue
                dfm = pd.read_parquet(pq)
                fc = [col for col in dfm.columns if col not in ("id_ruta", "y")]
                feat = dfm[fc].astype(float)
                ranking = u.ranking_from_matrix(feat)
                rk_path = os.path.join(rankdir, tag + ".csv")
                if not os.path.exists(rk_path):
                    ranking.to_csv(rk_path, index=False)
                conv = u.ranking_convergence(feat, repeats=cfg["conv_repeats"],
                                             top_k=cfg["conv_top_k"], full_ranking=ranking)
                conv.to_csv(os.path.join(convdir, tag + ".csv"), index=False)
                conv_pt = u.convergence_point(conv)
                write_manifest(matdir, tag, cfg,
                               dict(n=None, n_shap_rows=int(len(dfm)),
                                    n_features=int(len(fc)), auc_cv=None,
                                    konv_peringkat=conv_pt, adopted=True))
                print(f"  + {tag}: manifest + konvergensi (baris={len(dfm)}, fitur={len(fc)}, konv~{conv_pt})")
                n += 1
    print(f"Selesai. {n} unit diadopsi (turunan murah diregenerasi dari matriks).")
    print("Kini jalankan sisa yang belum jadi: python -u phase1_runner.py --resume\n")


def run_all(cfg):
    print("=" * 78)
    print(f"RUNNER FASE 1 (Epidemiologic Interpretability) - versi kode {u.CODE_VERSION}")
    print("=" * 78)
    import xgboost, shap, sklearn, imblearn
    print(f"Lingkungan: xgboost {xgboost.__version__} | shap {shap.__version__} "
          f"| sklearn {sklearn.__version__} | imblearn {imblearn.__version__}")

    outdir = cfg["output_dir"]
    matdir = os.path.join(outdir, "matrix")
    rankdir = os.path.join(outdir, "rankings")
    perfdir = os.path.join(outdir, "perf")
    convdir = os.path.join(outdir, "convergence")
    for d in (matdir, rankdir, perfdir, convdir):
        os.makedirs(d, exist_ok=True)

    if cfg["adopt_existing"]:
        adopt_existing(cfg, matdir, rankdir, convdir)
        return

    if not os.path.exists(cfg["parquet_path"]):
        _fail(f"Parquet tidak ditemukan: {cfg['parquet_path']}")
    schema = u.load_schema(cfg["schema_path"])
    mat = pd.read_csv(cfg["matriks_path"])
    print(f"Memuat Parquet: {cfg['parquet_path']}")
    df = pd.read_parquet(cfg["parquet_path"])
    print(f"  bentuk: {df.shape}\n")

    srcmap = auto_map_sources(df, cfg)
    validate(df, mat, schema, cfg)

    u.SHAP_N_JOBS = cfg["shap_n_jobs"]
    u.SHAP_BACKEND = cfg["shap_backend"]
    print(f"SHAP paralel: n_jobs={u.SHAP_N_JOBS} backend={u.SHAP_BACKEND} "
          f"(joblib {'ada' if u._HAS_JOBLIB else 'TIDAK ADA'})", flush=True)

    plan = build_plan(cfg)
    skipped = []
    if cfg["resume"]:
        keep = []
        for c, k, m in plan:
            tag = tag_of(c, k, m)
            if manifest_ok(matdir, tag, cfg):
                skipped.append(tag)
            else:
                keep.append((c, k, m))
        plan = keep
    if skipped:
        print(f"RESUME: {len(skipped)} unit dilewati (manifest cocok): {skipped}")
    total = len(plan)
    if total == 0:
        print("Tidak ada unit untuk dihitung (semua sudah jadi atau seleksi kosong).")
        n = assemble_summary(cfg, matdir, perfdir, outdir)
        print(f"Ringkasan mencakup {n} unit yang sudah jadi di {outdir}/phase1_summary.csv")
        return

    sizes = {}
    for c, k, m in plan:
        sizes[(c, k)] = int(((df[cfg["source_col"]] == srcmap[c]) &
                             (df[cfg["kohort_col"]] == k)).sum())
    print(f"\nRENCANA: {total} unit akan dihitung. matrix_sample={cfg['matrix_sample']}\n", flush=True)

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    def say(msg):
        (tqdm.write if tqdm is not None else print)(msg)

    rate = {}
    outer = tqdm(total=total, desc="UNIT", unit="unit", position=0, leave=True) if tqdm else None
    for i, (c, k, m) in enumerate(plan, 1):
        tag = tag_of(c, k, m)
        n_rows = sizes[(c, k)]
        feats = u.frozen_cell_features(mat, schema, c, k)
        known = {mm: sum(s for s, _ in v) / sum(r for _, r in v) for mm, v in rate.items() if v}
        eta = ""
        if known:
            spr = max(known.values())
            rem = sum(known.get(plan[j][2], spr) * sizes[(plan[j][0], plan[j][1])]
                      for j in range(i - 1, total))
            eta = f"   estimasi sisa ~{rem/60:.0f} mnt (kasar)"
        say(f"[{i}/{total}] {tag}: n={n_rows} fitur={len(feats)} (mulai){eta}")

        t0 = time.time()
        cell = df[(df[cfg["source_col"]] == srcmap[c]) &
                  (df[cfg["kohort_col"]] == k)].copy()
        res = u.measure_cell(cell, feats, schema, target=cfg["target"], group=cfg["group"],
                             model=m, matrix_sample=cfg["matrix_sample"],
                             verbose=True, label=tag)
        dt = time.time() - t0
        rate.setdefault(m, []).append((dt, n_rows))

        res["matrix"].to_parquet(os.path.join(matdir, tag + ".parquet"), index=False)
        res["ranking"].to_csv(os.path.join(rankdir, tag + ".csv"), index=False)
        with open(os.path.join(perfdir, tag + ".json"), "w") as f:
            json.dump(res["performance"], f, indent=2)
        conv = u.ranking_convergence(res["matrix"][res["feature_cols"]].astype(float),
                                     repeats=cfg["conv_repeats"], top_k=cfg["conv_top_k"],
                                     full_ranking=res["ranking"])
        conv.to_csv(os.path.join(convdir, tag + ".csv"), index=False)
        conv_pt = u.convergence_point(conv)

        write_manifest(matdir, tag, cfg,
                       dict(n=res["n"], n_shap_rows=res["n_shap_rows"],
                            n_features=res["n_features"],
                            auc_cv=round(res["performance"]["auc_mean"], 4),
                            konv_peringkat=conv_pt, seconds=round(dt, 1)))

        cmp_txt = ""
        if cfg["heritage_rank_dir"]:
            hpath = os.path.join(cfg["heritage_rank_dir"], tag + ".csv")
            if os.path.exists(hpath):
                cmp = u.compare_to_heritage(res["ranking"], hpath)
                cmp_txt = f"  vs-warisan: rho={cmp.get('spearman_rank')} jac10={cmp.get('jaccard_topk')}"
        say(f"   {tag}: SELESAI {dt/60:.1f} mnt  AUC={res['performance']['auc_mean']:.4f}"
            f"  baris_matriks={res['n_shap_rows']}  konv-peringkat~{conv_pt}{cmp_txt}")
        if outer:
            outer.update(1)
    if outer:
        outer.close()

    n = assemble_summary(cfg, matdir, perfdir, outdir)
    print(f"\nSELESAI run ini: {total} unit. Ringkasan mencakup {n}/12 unit yang sudah jadi.")
    print(f"Tersimpan di {outdir}/  (matrix, rankings, convergence, perf, manifest)")
    if n < 12:
        print(f"Masih {12 - n} unit belum jadi. Lanjutkan dengan: python -u phase1_runner.py --resume")


def parse_args(cfg):
    ap = argparse.ArgumentParser(description="Runner Fase 1 (per unit, resume aman).")
    ap.add_argument("--only", nargs="+", help="tag unit tertentu, mis. ski23_baduta_xgb")
    ap.add_argument("--sources", nargs="+", help="subset sumber, mis. ski23")
    ap.add_argument("--cohorts", nargs="+", help="subset kohort, mis. balita_tua")
    ap.add_argument("--models", nargs="+", help="subset model: xgb rf")
    ap.add_argument("--resume", action="store_true", help="lewati unit yang manifest-nya cocok")
    ap.add_argument("--adopt-existing", action="store_true",
                    help="tulis manifest untuk parquet lama yang sudah ada, lalu keluar")
    ap.add_argument("--shap-n-jobs", type=int, help="override paralelisasi SHAP RF")
    ap.add_argument("--shap-backend", choices=["loky", "threading"], help="backend SHAP RF")
    ap.add_argument("--matrix-sample", type=int, help="override ukuran matriks")
    a = ap.parse_args()
    if a.only:
        cfg["only"] = a.only
    if a.sources:
        cfg["sources"] = a.sources
    if a.cohorts:
        cfg["kohort_values"] = a.cohorts
    if a.models:
        cfg["models"] = a.models
    if a.resume:
        cfg["resume"] = True
    if a.adopt_existing:
        cfg["adopt_existing"] = True
    if a.shap_n_jobs is not None:
        cfg["shap_n_jobs"] = a.shap_n_jobs
    if a.shap_backend:
        cfg["shap_backend"] = a.shap_backend
    if a.matrix_sample is not None:
        cfg["matrix_sample"] = a.matrix_sample
    return cfg


if __name__ == "__main__":
    run_all(parse_args(dict(CONFIG)))
