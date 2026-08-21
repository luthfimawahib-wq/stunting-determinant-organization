import numpy as np
import pandas as pd
import phase3_shares as D

SOURCES = D.SOURCES
COHORTS = D.COHORTS


def _resampled_shift(matdir, model, mapping, feats, rng, level="domain"):
    shifts = {}
    for s in SOURCES:
        mb = D.load_matrix(matdir, s, "baduta", model)
        mt = D.load_matrix(matdir, s, "balita_tua", model)
        ib = D._cluster_resample_importance(mb, feats, rng)
        it = D._cluster_resample_importance(mt, feats, rng)
        if level == "domain":
            sb, st = D.domain_share(ib, mapping), D.domain_share(it, mapping)
            idx = sorted(set(sb.index) | set(st.index))
            shifts[s] = st.reindex(idx).fillna(0) - sb.reindex(idx).fillna(0)
        else:
            sb, st = ib / (ib.sum() + D.EPS), it / (it.sum() + D.EPS)
            shifts[s] = st - sb
    return pd.DataFrame(shifts).mean(axis=1)


def bootstrap_precision(matdir, model, mapping, feats, B=200, seed=42, level="domain"):
    rng = np.random.default_rng(seed)
    feats = [f for f in feats if mapping.get(f) is not None]
    draws = [_resampled_shift(matdir, model, mapping, feats, rng, level) for _ in range(B)]
    df = pd.DataFrame(draws)
    out = pd.DataFrame({
        "ci_lo": df.quantile(0.025).round(5),
        "ci_hi": df.quantile(0.975).round(5),
        "sd_bootstrap": df.std().round(5),
    })
    return out, df


def loso(matdir, model, mapping, feats):
    feats = [f for f in feats if mapping.get(f) is not None]
    per_source = {}
    for s in SOURCES:
        r = D.shares_and_shift(matdir, s, model, mapping, feats=feats)
        per_source[s] = r["shift"]
    sh = pd.DataFrame(per_source)
    out = pd.DataFrame(index=sh.index)
    out["penuh_3_sumber"] = sh.mean(axis=1).round(5)
    for left in SOURCES:
        keep = [s for s in SOURCES if s != left]
        out[f"tanpa_{left}"] = sh[keep].mean(axis=1).round(5)
    cols = [c for c in out.columns if c.startswith("tanpa_")]
    signs = np.sign(out[cols].where(out[cols].abs() > 1e-5, 0.0))
    ref = np.sign(out["penuh_3_sumber"].where(out["penuh_3_sumber"].abs() > 1e-5, 0.0))
    out["arah_bertahan"] = [
        "ya" if (ref[d] != 0 and (signs.loc[d] == ref[d]).all()) else
        ("takada" if ref[d] == 0 else "tidak") for d in out.index]
    out["magnitudo_min"] = out[cols].abs().min(axis=1).round(5)
    out["magnitudo_maks"] = out[cols].abs().max(axis=1).round(5)
    denom = out["penuh_3_sumber"].abs().replace(0, np.nan)
    out["rasio_magnitudo_min"] = (out["magnitudo_min"] / denom).round(3)
    return out


def determinant_rank_stability(matdir, model, mapping, feats, domains, B=200, seed=42,
                               top_k=3):
    rng = np.random.default_rng(seed)
    feats = [f for f in feats if mapping.get(f) is not None]
    by_dom = {d: [f for f in feats if mapping.get(f) == d] for d in domains}
    tally = {d: pd.DataFrame(0.0, index=by_dom[d], columns=["peringkat_1", f"top_{top_k}"])
             for d in domains}
    for _ in range(B):
        sh = _resampled_shift(matdir, model, mapping, feats, rng, level="determinant")
        for d in domains:
            sub = sh.reindex(by_dom[d]).fillna(0.0)
            dom_dir = np.sign(sub.sum()) or 1.0
            contrib = sub * dom_dir
            order = contrib.sort_values(ascending=False)
            if len(order):
                tally[d].loc[order.index[0], "peringkat_1"] += 1
                for f in order.index[:top_k]:
                    tally[d].loc[f, f"top_{top_k}"] += 1
    out = {}
    for d in domains:
        t = (tally[d] / B).round(3)
        t = t.sort_values("peringkat_1", ascending=False)
        out[d] = t
    return out


def feature_space_comparison(matdir, model, mapping, feats_full):
    feats_full = [f for f in feats_full if mapping.get(f) is not None]
    sh_full, _, _ = D.shift_matrix(matdir, model, mapping, feats=feats_full)
    sh_per, _, _ = D.shift_matrix(matdir, model, mapping, feats=None)
    doms = sorted(set(sh_full.index) | set(sh_per.index))
    out = pd.DataFrame(index=doms)
    out["irisan_penuh"] = sh_full.mean(axis=1).reindex(doms).round(5)
    out["irisan_per_sumber"] = sh_per.mean(axis=1).reindex(doms).round(5)
    out["selisih"] = (out["irisan_penuh"] - out["irisan_per_sumber"]).round(5)
    a = np.sign(out["irisan_penuh"].where(out["irisan_penuh"].abs() > 1e-5, 0.0))
    b = np.sign(out["irisan_per_sumber"].where(out["irisan_per_sumber"].abs() > 1e-5, 0.0))
    out["arah_sama"] = np.where((a != 0) & (a == b), "ya",
                                np.where((a == 0) | (b == 0), "takada", "tidak"))
    out["n_fitur_penuh"] = D.domain_feature_counts(feats_full, mapping).reindex(doms).fillna(0).astype(int)
    return out.sort_values("irisan_penuh", key=abs, ascending=False)


def robustness_recap(domains, klasifikasi, lintas_model, loso_tbl, fs_tbl,
                     rank_stab, sensitivitas, top_k=3, loso_ratio_min=0.5):
    rows = {}
    rows["Konsistensi arah tiga sumber"] = {
        d: ("bertahan" if str(klasifikasi.loc[d, "konsistensi"]) == "konsisten" else "tidak")
        for d in domains}
    rows["Kesepakatan tiga metode batas"] = {
        d: ("bertahan" if str(klasifikasi.loc[d, "kesepakatan"]) == "kokoh" else "sebagian")
        for d in domains}
    rows["Replikasi dua arsitektur model"] = {
        d: ("bertahan" if str(lintas_model.get(d, "")) == "ya" else "tidak") for d in domains}
    def loso_cell(d):
        if str(loso_tbl.loc[d, "arah_bertahan"]) != "ya":
            return "tidak"
        r = loso_tbl.loc[d, "rasio_magnitudo_min"]
        if pd.isna(r):
            return "takada"
        return "bertahan" if r >= loso_ratio_min else "sebagian"
    rows["Tinggalkan satu sumber"] = {d: loso_cell(d) for d in domains}
    rows["Ruang fitur (penuh vs per sumber)"] = {
        d: ("bertahan" if str(fs_tbl.loc[d, "arah_sama"]) == "ya" else "tidak")
        for d in domains}
    rows["Penempatan variabel batas"] = {
        d: ("bertahan" if str(sensitivitas.loc[d, "stabil"]) == "ya" else "sebagian")
        for d in domains}
    def rank_cell(d):
        if d not in rank_stab or not len(rank_stab[d]):
            return "takada"
        top = rank_stab[d].iloc[0]
        p = float(top[f"top_{top_k}"])
        return "bertahan" if p >= 0.8 else ("sebagian" if p >= 0.5 else "tidak")
    rows[f"Stabilitas penyumbang utama (top-{top_k})"] = {d: rank_cell(d) for d in domains}
    return pd.DataFrame(rows).T[list(domains)]
