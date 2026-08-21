import os
import numpy as np
import pandas as pd
import phase3_shares as D

SOURCES = D.SOURCES
COHORTS = D.COHORTS
SEED_FASE1 = 42
MATRIX_SAMPLE_FASE1 = 40000


def sample_size_from_manifest(matdir, source, cohort, model, default=MATRIX_SAMPLE_FASE1):
    import json
    p = os.path.join(matdir, f"{source}_{cohort}_{model}.manifest.json")
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return int(json.load(f).get("matrix_sample", default))
        except Exception:
            return default
    return default


def reconstruct_sampled_index(n_rows, sample=MATRIX_SAMPLE_FASE1, seed=SEED_FASE1):
    if n_rows <= sample:
        return np.arange(n_rows)
    dummy = pd.DataFrame(index=pd.RangeIndex(n_rows))
    return dummy.sample(sample, random_state=seed).index.to_numpy()


def cell_frame(df, source_col, source_literal, cohort_col, cohort):
    return df[(df[source_col] == source_literal) & (df[cohort_col] == cohort)]


def attach_weights(matrix, cell, idx, weight_col, group_col="id_ruta"):
    cell = cell.reset_index(drop=True)
    if len(idx) != len(matrix):
        raise SystemExit(f"VERIFIKASI GAGAL: baris rekonstruksi {len(idx)} != baris matriks {len(matrix)}")
    got = cell[group_col].astype(str).str.strip().to_numpy()[idx]
    want = matrix[group_col].astype(str).str.strip().to_numpy()
    n_mismatch = int((got != want).sum())
    if n_mismatch:
        raise SystemExit(f"VERIFIKASI GAGAL: {n_mismatch} id_ruta tidak cocok. "
                         "Rekonstruksi baris tidak sah; jangan lanjutkan.")
    w = pd.to_numeric(cell[weight_col], errors="coerce").to_numpy()[idx]
    w = np.where(np.isfinite(w) & (w > 0), w, np.nan)
    if np.isnan(w).all():
        raise SystemExit(f"VERIFIKASI GAGAL: kolom bobot '{weight_col}' kosong pada sel ini.")
    w = np.where(np.isnan(w), np.nanmedian(w), w)
    return w


def weighted_importance(matrix, feats, w):
    A = matrix[feats].abs().to_numpy(dtype=float)
    return pd.Series(np.average(A, axis=0, weights=w), index=feats)


def weighted_shift(matdir, df, cfg, model, mapping, feats):
    feats = [f for f in feats if mapping.get(f) is not None]
    rows_w, rows_u = {}, {}
    audit = []
    for s in SOURCES:
        share = {}
        for coh in COHORTS:
            mat = D.load_matrix(matdir, s, coh, model)
            cell = cell_frame(df, cfg["source_col"], cfg["source_flag_map"][s],
                              cfg["kohort_col"], coh)
            ss = sample_size_from_manifest(matdir, s, coh, model)
            idx = reconstruct_sampled_index(len(cell), sample=ss)
            w = attach_weights(mat, cell, idx, cfg["weight_col"])
            share[(coh, "berbobot")] = D.domain_share(weighted_importance(mat, feats, w), mapping)
            share[(coh, "takberbobot")] = D.domain_share(D.importance(mat, feats), mapping)
            audit.append(dict(sumber=s, kohort=coh, n_sel=len(cell), n_baris_matriks=len(mat),
                              bobot_min=round(float(np.min(w)), 4),
                              bobot_median=round(float(np.median(w)), 4),
                              bobot_maks=round(float(np.max(w)), 4)))
        doms = sorted(set(share[("baduta", "berbobot")].index) |
                      set(share[("balita_tua", "berbobot")].index))
        rows_w[s] = (share[("balita_tua", "berbobot")].reindex(doms).fillna(0) -
                     share[("baduta", "berbobot")].reindex(doms).fillna(0))
        rows_u[s] = (share[("balita_tua", "takberbobot")].reindex(doms).fillna(0) -
                     share[("baduta", "takberbobot")].reindex(doms).fillna(0))
    return pd.DataFrame(rows_w), pd.DataFrame(rows_u), pd.DataFrame(audit)


def compare(shift_w, shift_u):
    out = pd.DataFrame(index=sorted(set(shift_w.index) | set(shift_u.index)))
    out["takberbobot"] = shift_u.mean(axis=1).round(5)
    out["berbobot"] = shift_w.mean(axis=1).round(5)
    out["selisih"] = (out["berbobot"] - out["takberbobot"]).round(5)
    a = np.sign(out["takberbobot"].where(out["takberbobot"].abs() > 1e-5, 0.0))
    b = np.sign(out["berbobot"].where(out["berbobot"].abs() > 1e-5, 0.0))
    out["arah_sama"] = np.where((a != 0) & (a == b), "ya",
                                np.where((a == 0) | (b == 0), "takada", "tidak"))
    sg = np.sign(shift_w.where(shift_w.abs() > 1e-5, 0.0))
    def consist(r):
        nz = r[r != 0]
        return "nol" if len(nz) == 0 else ("konsisten" if (nz > 0).all() or (nz < 0).all()
                                           else "tak-konsisten")
    out["konsistensi_berbobot"] = sg.apply(consist, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["rasio_besaran"] = np.where(out["takberbobot"].abs() > 1e-9,
                                        (out["berbobot"] / out["takberbobot"]).round(3), np.nan)
    return out.sort_values("takberbobot", key=abs, ascending=False)
