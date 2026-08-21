import os
import numpy as np
import pandas as pd

SOURCES = ["ssgi22", "ssgi24", "ski23"]
COHORTS = ["baduta", "balita_tua"]
META_COLS = {"id_ruta", "y"}
EPS = 1e-12


def load_kamus(path):
    k = pd.read_csv(path)
    k = k[~k.domain.str.contains("struktural", case=False, na=False)]
    primer = dict(zip(k.feature, k.domain))
    sens = {}
    border = set()
    for _, r in k.iterrows():
        if isinstance(r.get("domain_sensitivitas"), str) and r["domain_sensitivitas"].strip():
            sens[r.feature] = r["domain_sensitivitas"]
            border.add(r.feature)
        else:
            sens[r.feature] = r["domain"]
    return primer, sens, border


def load_matrix(matdir, source, cohort, model):
    p = os.path.join(matdir, f"{source}_{cohort}_{model}.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return pd.read_parquet(p)


def feature_cols(matrix):
    return [c for c in matrix.columns if c not in META_COLS]


def importance(matrix, feats):
    return matrix[feats].abs().mean()


def domain_share(imp, mapping):
    dom = {}
    for f, v in imp.items():
        d = mapping.get(f)
        if d is None:
            continue
        dom[d] = dom.get(d, 0.0) + float(v)
    tot = sum(dom.values()) + EPS
    return pd.Series({d: v / tot for d, v in dom.items()})


def cross_cohort_feats(mat_bad, mat_bal):
    return sorted(set(feature_cols(mat_bad)) & set(feature_cols(mat_bal)))


def full_intersection_feats(matdir, model):
    sets = []
    for s in SOURCES:
        for c in COHORTS:
            sets.append(set(feature_cols(load_matrix(matdir, s, c, model))))
    return sorted(set.intersection(*sets))


def domain_feature_counts(feats, mapping):
    c = {}
    for f in feats:
        d = mapping.get(f)
        if d is not None:
            c[d] = c.get(d, 0) + 1
    return pd.Series(c, name="n_fitur")


def cohort_only_composition(matdir, model, mapping, cohort="baduta"):
    out = {}
    for s in SOURCES:
        m = load_matrix(matdir, s, cohort, model)
        fc = feature_cols(m)
        out[s] = domain_share(importance(m, fc), mapping)
    return pd.DataFrame(out)


def shares_and_shift(matdir, source, model, mapping, feats=None):
    mb = load_matrix(matdir, source, "baduta", model)
    mt = load_matrix(matdir, source, "balita_tua", model)
    if feats is None:
        feats = cross_cohort_feats(mb, mt)
    sb = domain_share(importance(mb, feats), mapping)
    st = domain_share(importance(mt, feats), mapping)
    domains = sorted(set(sb.index) | set(st.index))
    sb = sb.reindex(domains).fillna(0.0)
    st = st.reindex(domains).fillna(0.0)
    shift = st - sb
    return dict(share_baduta=sb, share_balita=st, shift=shift, n_feats=len(feats))


def shift_matrix(matdir, model, mapping, feats=None):
    shifts, sb_all, st_all = {}, {}, {}
    for s in SOURCES:
        r = shares_and_shift(matdir, s, model, mapping, feats=feats)
        shifts[s] = r["shift"]
        sb_all[s] = r["share_baduta"]
        st_all[s] = r["share_balita"]
    shift_df = pd.DataFrame(shifts)
    return shift_df, pd.DataFrame(sb_all), pd.DataFrame(st_all)


def summarize_domains(shift_df):
    out = pd.DataFrame(index=shift_df.index)
    out["shift_mean"] = shift_df.mean(axis=1)
    out["shift_min"] = shift_df.min(axis=1)
    out["shift_max"] = shift_df.max(axis=1)
    out["magnitude"] = out["shift_mean"].abs()
    signs = np.sign(shift_df.where(shift_df.abs() > 1e-4, 0.0))
    def consist(row):
        nz = row[row != 0]
        if len(nz) == 0:
            return "nol"
        return "konsisten" if (nz > 0).all() or (nz < 0).all() else "tak-konsisten"
    out["arah"] = np.where(out["shift_mean"] >= 0, "naik", "turun")
    out["konsistensi"] = signs.apply(consist, axis=1)
    out["n_sumber_searah"] = signs.apply(lambda r: int(max((r > 0).sum(), (r < 0).sum())), axis=1)
    return out.sort_values("magnitude", ascending=False)


def cross_model_check(shift_xgb, shift_rf):
    agree = {}
    for d in shift_xgb.index:
        sx = np.sign(shift_xgb.loc[d].where(shift_xgb.loc[d].abs() > 1e-4, 0.0))
        sr = np.sign(shift_rf.loc[d].where(shift_rf.loc[d].abs() > 1e-4, 0.0))
        both = [(a, b) for a, b in zip(sx, sr) if a != 0 and b != 0]
        agree[d] = "ya" if both and all(a == b for a, b in both) else ("takada" if not both else "tidak")
    return pd.Series(agree, name="lintas_model_searah")


_CLASSES = ["invariant", "transisi", "spesifik-tahap"]


def _assign_by_edges(mag, lo, hi):
    return pd.Series(np.where(mag <= lo, "invariant",
                     np.where(mag >= hi, "spesifik-tahap", "transisi")), index=mag.index)


def boundary_percentile(mag, q=(1/3, 2/3)):
    lo, hi = np.quantile(mag.values, q[0]), np.quantile(mag.values, q[1])
    return _assign_by_edges(mag, lo, hi), (float(lo), float(hi))


def boundary_natural(mag, ks=(2, 3), seed=42):
    x = mag.values.reshape(-1, 1)
    n = len(x)
    best = None
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        for k in ks:
            if k >= n:
                continue
            km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)
            lab = km.labels_
            sil = silhouette_score(x, lab) if len(set(lab)) > 1 else -1
            if best is None or sil > best[0]:
                best = (sil, k, km)
    except Exception:
        best = None
    if best is None:
        cls, edges = boundary_percentile(mag)
        return cls, dict(k=3, method="fallback-persentil", edges=edges)
    _, k, km = best
    centers = km.cluster_centers_.ravel()
    order = np.argsort(centers)
    rank = {c: i for i, c in enumerate(order)}
    lab_rank = pd.Series([rank[c] for c in km.labels_], index=mag.index)
    if k == 2:
        cls = lab_rank.map({0: "invariant", 1: "spesifik-tahap"})
    else:
        cls = lab_rank.map({0: "invariant", 1: "transisi", 2: "spesifik-tahap"})
    cs = np.sort(centers)
    edges = [float((cs[i] + cs[i + 1]) / 2) for i in range(len(cs) - 1)]
    return cls, dict(k=int(k), method="natural-break-kmeans", edges=edges)


def _cluster_resample_importance(matrix, feats, rng):
    ids = matrix["id_ruta"].values
    uniq = pd.unique(ids)
    pick = rng.choice(uniq, size=len(uniq), replace=True)
    by = {}
    for i, g in enumerate(ids):
        by.setdefault(g, []).append(i)
    idx = np.concatenate([by[g] for g in pick])
    sub = matrix.iloc[idx]
    return sub[feats].abs().mean()


def boundary_bootstrap(matdir, model, mapping, ref_edges, B=200, seed=42, use_percentile=True,
                       feats_fixed=None):
    rng = np.random.default_rng(seed)
    mats = {(s, c): load_matrix(matdir, s, c, model) for s in SOURCES for c in COHORTS}
    if feats_fixed is None:
        feats = {s: cross_cohort_feats(mats[(s, "baduta")], mats[(s, "balita_tua")]) for s in SOURCES}
    else:
        feats = {s: list(feats_fixed) for s in SOURCES}
    domains = None
    tally = None
    shift_draws = []
    for _ in range(B):
        shifts = {}
        for s in SOURCES:
            ib = _cluster_resample_importance(mats[(s, "baduta")], feats[s], rng)
            it = _cluster_resample_importance(mats[(s, "balita_tua")], feats[s], rng)
            sb = domain_share(ib, mapping)
            st = domain_share(it, mapping)
            ds = sorted(set(sb.index) | set(st.index))
            shifts[s] = (st.reindex(ds).fillna(0) - sb.reindex(ds).fillna(0))
        sh = pd.DataFrame(shifts)
        mean_shift = sh.mean(axis=1)
        shift_draws.append(mean_shift)
        mag = mean_shift.abs()
        if use_percentile:
            cls, _ = boundary_percentile(mag)
        else:
            cls = _assign_by_edges(mag, ref_edges[0], ref_edges[-1])
        if domains is None:
            domains = mag.index
            tally = pd.DataFrame(0, index=domains, columns=_CLASSES)
        for d in cls.index:
            tally.loc[d, cls[d]] += 1
    prop = tally.div(tally.sum(axis=1), axis=0)
    boot_cls = prop.idxmax(axis=1)
    boot_stab = prop.max(axis=1)
    draws = pd.DataFrame(shift_draws)
    ci = pd.DataFrame({
        "ci_lo": draws.quantile(0.025),
        "ci_hi": draws.quantile(0.975),
    })
    ci["terbedakan_dari_nol"] = np.where((ci.ci_lo > 0) | (ci.ci_hi < 0), "ya", "tidak")
    return boot_cls, boot_stab, prop, ci


def classify(summary, matdir, model, mapping, B=200, seed=42, feats_fixed=None, fcounts=None):
    mag = summary["magnitude"]
    cls_pct, edges_pct = boundary_percentile(mag)
    cls_nat, info_nat = boundary_natural(mag)
    boot_cls, boot_stab, boot_prop, ci = boundary_bootstrap(matdir, model, mapping, edges_pct,
                                                            B=B, seed=seed, feats_fixed=feats_fixed)
    df = pd.DataFrame(index=mag.index)
    df["magnitude"] = mag
    df["arah"] = summary["arah"]
    df["konsistensi"] = summary["konsistensi"]
    df["ci_dalam_sumber_lo"] = ci["ci_lo"].reindex(df.index).round(4)
    df["ci_dalam_sumber_hi"] = ci["ci_hi"].reindex(df.index).round(4)
    df["rentang_antar_sumber"] = (summary["shift_min"].round(4).astype(str) + " sd " +
                                  summary["shift_max"].round(4).astype(str))
    df["kelas_persentil"] = cls_pct
    df["kelas_natural"] = cls_nat.reindex(df.index)
    df["kelas_bootstrap"] = boot_cls.reindex(df.index)
    df["stabilitas_bootstrap"] = boot_stab.reindex(df.index).round(3)
    def agree(r):
        c = {r["kelas_persentil"], r["kelas_natural"], r["kelas_bootstrap"]}
        return "kokoh" if len(c) == 1 else "kasus-batas"
    df["kesepakatan"] = df.apply(agree, axis=1)
    df.loc[summary["konsistensi"] == "tak-konsisten", "kesepakatan"] = "arah-tak-konsisten"
    df["kelas_kokoh"] = np.where(df["kesepakatan"] == "kokoh", df["kelas_persentil"], "-")
    if fcounts is not None:
        df.insert(1, "n_fitur", fcounts.reindex(df.index).fillna(0).astype(int))
    meta = dict(k_natural=info_nat.get("k"), edges_persentil=edges_pct,
                edges_natural=info_nat.get("edges"), B=B)
    return df.sort_values("magnitude", ascending=False), meta, boot_prop


def determinant_shares(matdir, source, model, feats):
    mb = load_matrix(matdir, source, "baduta", model)
    mt = load_matrix(matdir, source, "balita_tua", model)
    ib, it = importance(mb, feats), importance(mt, feats)
    sb, st = ib / (ib.sum() + EPS), it / (it.sum() + EPS)
    return sb, st, st - sb


def determinant_shift_table(matdir, model, feats, mapping):
    feats = [f for f in feats if mapping.get(f) is not None]
    shifts, sbs, sts = {}, {}, {}
    for s in SOURCES:
        sb, st, sh = determinant_shares(matdir, s, model, feats)
        shifts[s], sbs[s], sts[s] = sh, sb, st
    sh_df = pd.DataFrame(shifts).reindex(feats).fillna(0.0)
    out = pd.DataFrame(index=sh_df.index)
    out["domain"] = [mapping.get(f, "-") for f in sh_df.index]
    out["pangsa_baduta"] = pd.DataFrame(sbs).reindex(feats).mean(axis=1).round(5)
    out["pangsa_balita_tua"] = pd.DataFrame(sts).reindex(feats).mean(axis=1).round(5)
    for s in SOURCES:
        out[f"shift_{s}"] = sh_df[s].round(5)
    out["shift_mean"] = sh_df.mean(axis=1)
    out["magnitude"] = out["shift_mean"].abs()
    out["arah"] = np.where(out["shift_mean"] >= 0, "naik", "turun")
    signs = np.sign(sh_df.where(sh_df.abs() > 1e-5, 0.0))
    def consist(row):
        nz = row[row != 0]
        if len(nz) == 0:
            return "nol"
        return "konsisten" if (nz > 0).all() or (nz < 0).all() else "tak-konsisten"
    out["konsistensi"] = signs.apply(consist, axis=1)
    dom_shift = out.groupby("domain")["shift_mean"].transform("sum")
    out["shift_domain"] = dom_shift
    with np.errstate(divide="ignore", invalid="ignore"):
        out["kontribusi_persen"] = np.where(dom_shift.abs() > EPS,
                                            (out["shift_mean"] / dom_shift * 100).round(1), np.nan)
    return out.sort_values(["domain", "magnitude"], ascending=[True, False])


def verify_decomposition(det_table, domain_summary, tol=1e-6):
    agg = det_table.groupby("domain")["shift_mean"].sum()
    rows = []
    for d in domain_summary.index:
        lv1 = float(domain_summary.loc[d, "shift_mean"])
        lv2 = float(agg.get(d, 0.0))
        rows.append(dict(domain=d, shift_level1=round(lv1, 6), jumlah_level2=round(lv2, 6),
                         selisih=round(lv1 - lv2, 8), cocok=("ya" if abs(lv1 - lv2) < tol else "TIDAK")))
    return pd.DataFrame(rows)
