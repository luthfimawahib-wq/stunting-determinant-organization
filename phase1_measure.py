import json
import numpy as np
import pandas as pd

import xgboost as xgb
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None
try:
    import joblib as _joblib
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

FIXED_XGB_PARAMS = dict(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5.0,
    reg_lambda=1.0, reg_alpha=0.0, gamma=0.0,
    objective="binary:logistic", eval_metric="auc",
    tree_method="hist", n_jobs=-1, random_state=42,
)
FIXED_RF_PARAMS = dict(
    n_estimators=600, max_depth=None, min_samples_leaf=20,
    max_features="sqrt", n_jobs=-1, random_state=42,
)

LEAKAGE = {"height_child_cm", "weight_child_kg"}
AVAIL_THRESHOLD = 0.5

SHAP_N_JOBS = 4
SHAP_BACKEND = "loky"

MATRIX_SAMPLE = 40000

CODE_VERSION = "phase1-1.0"


def load_schema(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def feature_types(schema):
    t = {}
    for grp in ("biner", "ordinal", "nominal", "kontinu"):
        for v in schema.get(grp, []):
            t[v] = grp
    return t


def frozen_cell_features(mat, schema, source, cohort):
    exc = (set(schema["meta"]) | set(schema["drop"])
           | {"stunting_binary", "haz_score", "waz_score", "whz_score"} | LEAKAGE)
    ncol, avail = f"n_sumber_{cohort}", f"{source}_{cohort}"
    m = mat[~mat.variabel.isin(exc)]
    sel = m[(m[ncol] >= 2) & (m[avail] >= AVAIL_THRESHOLD)]
    return sel.variabel.tolist()


def preprocess(df, feature_list, schema, add_missing_indicator=True):
    types = feature_types(schema)
    feats = [f for f in feature_list if f not in LEAKAGE and f in df.columns]
    frames, colmap = [], {}
    for f in feats:
        t = types.get(f, "kontinu")
        if t == "nominal":
            d = pd.get_dummies(df[f].astype("category"), prefix=f, prefix_sep="=",
                               dummy_na=bool(df[f].isna().any()))
            for c in d.columns:
                colmap[c] = f
            frames.append(d.astype(float))
        else:
            col = pd.to_numeric(df[f], errors="coerce").astype(float)
            if add_missing_indicator and col.isna().any():
                ind = col.isna().astype(float).rename(f + "__isNA")
                colmap[f + "__isNA"] = f
                col = col.fillna(col.median())
                frames.append(ind)
            colmap[f] = f
            frames.append(col.rename(f))
    X = pd.concat(frames, axis=1)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    return X, colmap


def _ctor(model):
    if model == "xgb":
        return lambda: xgb.XGBClassifier(**FIXED_XGB_PARAMS)
    if model == "rf":
        return lambda: RandomForestClassifier(**FIXED_RF_PARAMS)
    raise ValueError("model harus 'xgb' atau 'rf'")


def _pbar(iterable, verbose, **kw):
    if verbose and _tqdm is not None:
        return _tqdm(iterable, **kw)
    return iterable


def grouped_cv_performance(model, X, y, groups, n_splits=5, seed=42,
                           verbose=False, label=""):
    import time
    ctor = _ctor(model)
    try:
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(skf.split(X, y, groups))
    except Exception:
        splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    aucs, aps, pos_frac = [], [], []
    it = _pbar(splits, verbose, desc=f"CV  {label}", unit="fold", leave=False)
    for i, (tr, va) in enumerate(it, 1):
        t0 = time.time()
        ytr = y.iloc[tr]
        pos_frac.append(float(y.iloc[va].mean()))
        minority = int(min(ytr.sum(), len(ytr) - ytr.sum()))
        steps = []
        if minority > 5:
            steps.append(("smote", SMOTE(random_state=seed, k_neighbors=min(5, minority - 1))))
        steps.append(("clf", ctor()))
        pipe = ImbPipeline(steps)
        pipe.fit(X.iloc[tr], ytr)
        p = pipe.predict_proba(X.iloc[va])[:, 1]
        aucs.append(roc_auc_score(y.iloc[va], p))
        aps.append(average_precision_score(y.iloc[va], p))
        if verbose and _tqdm is not None and hasattr(it, "set_postfix_str"):
            it.set_postfix_str(f"AUC={aucs[-1]:.3f} ({time.time()-t0:.0f}s)")
    return dict(auc_mean=float(np.mean(aucs)), auc_sd=float(np.std(aucs)),
                ap_mean=float(np.mean(aps)), ap_sd=float(np.std(aps)),
                pos_frac_folds=[round(x, 4) for x in pos_frac],
                pos_frac_sd=float(np.std(pos_frac)))


def _fit_for_shap(model, X, y, seed=42):
    ctor = _ctor(model)
    minority = int(min(y.sum(), len(y) - y.sum()))
    steps = []
    if minority > 5:
        steps.append(("smote", SMOTE(random_state=seed, k_neighbors=min(5, minority - 1))))
    steps.append(("clf", ctor()))
    pipe = ImbPipeline(steps)
    pipe.fit(X, y)
    return pipe.named_steps["clf"]


def _is_xgb(clf):
    return clf.__class__.__name__.startswith("XGB") or \
        clf.__class__.__module__.split(".")[0] == "xgboost"


def _shap_chunk_worker(model, Xvals, columns):
    import shap as _shp
    import numpy as _np
    import pandas as _pd
    e = _shp.TreeExplainer(model)
    s = e.shap_values(_pd.DataFrame(Xvals, columns=columns), check_additivity=False)
    if isinstance(s, list):
        s = s[1]
    s = _np.asarray(s)
    if s.ndim == 3:
        s = s[:, :, -1]
    return s


def _rf_shap_parallel(clf, Xs, n_jobs, backend, verbose, label):
    from joblib import Parallel, delayed, cpu_count
    nj = cpu_count() if n_jobs in (-1, None) else int(n_jobs)
    nj = max(1, min(nj, cpu_count()))
    if nj == 1:
        return None
    ntask = max(nj, min(24, int(np.ceil(len(Xs) / 1500))))
    idxs = np.array_split(np.arange(len(Xs)), ntask)
    cols = list(Xs.columns)
    jobs = [delayed(_shap_chunk_worker)(clf, Xs.iloc[ix].values, cols) for ix in idxs]
    try:
        gen = Parallel(n_jobs=nj, backend=backend, return_as="generator")(jobs)
        if verbose and _tqdm is not None:
            gen = _tqdm(gen, total=len(jobs), desc=f"SHAP {label} [{backend} x{nj}]",
                        unit="chunk", leave=False)
        parts = list(gen)
    except TypeError:
        parts = Parallel(n_jobs=nj, backend=backend)(jobs)
    return np.vstack(parts)


def _signed_shap_encoded(clf, Xs, verbose=False, label="", batch=2000,
                         n_jobs=None, backend=None):
    n_jobs = SHAP_N_JOBS if n_jobs is None else n_jobs
    backend = SHAP_BACKEND if backend is None else backend
    nb = max(1, (len(Xs) + batch - 1) // batch)
    if _is_xgb(clf):
        booster = clf.get_booster()
        parts = []
        it = _pbar(range(nb), verbose, desc=f"SHAP {label}", unit="batch", leave=False)
        for b in it:
            sl = Xs.iloc[b * batch:(b + 1) * batch]
            dmat = xgb.DMatrix(sl, feature_names=list(Xs.columns), enable_categorical=False)
            c = np.asarray(booster.predict(dmat, pred_contribs=True))
            if c.ndim == 3:
                c = c[:, -1, :]
            parts.append(c[:, :-1])
        return np.vstack(parts)
    sv = None
    if _HAS_JOBLIB and n_jobs != 1:
        sv = _rf_shap_parallel(clf, Xs, n_jobs, backend, verbose, label)
    if sv is None:
        expl = shap.TreeExplainer(clf)
        parts = []
        it = _pbar(range(nb), verbose, desc=f"SHAP {label}", unit="batch", leave=False)
        for b in it:
            sl = Xs.iloc[b * batch:(b + 1) * batch]
            s = expl.shap_values(sl, check_additivity=False)
            if isinstance(s, list):
                s = s[1]
            s = np.asarray(s)
            if s.ndim == 3:
                s = s[:, :, -1]
            parts.append(s)
        sv = np.vstack(parts)
    return sv


def signed_feature_matrix(clf, X, colmap, sample=MATRIX_SAMPLE, seed=42,
                          verbose=False, label=""):
    Xs = X.sample(sample, random_state=seed) if len(X) > sample else X
    idx = Xs.index
    sv = _signed_shap_encoded(clf, Xs, verbose=verbose, label=label)
    enc = pd.DataFrame(sv, columns=list(Xs.columns), index=idx)
    groups = pd.Index([colmap.get(c, c) for c in enc.columns])
    feat_signed = enc.T.groupby(groups).sum().T
    return feat_signed, idx


def ranking_from_matrix(feat_signed):
    imp = feat_signed.abs().mean(axis=0)
    direction = feat_signed.mean(axis=0).reindex(imp.index)
    out = pd.DataFrame({"feature": imp.index,
                        "mean_abs_shap": imp.values,
                        "mean_signed_shap": direction.values})
    out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def measure_cell(df, feature_list, schema, target="stunting_binary", group="id_ruta",
                 model="xgb", matrix_sample=MATRIX_SAMPLE, cv=True,
                 verbose=False, label=""):
    X, colmap = preprocess(df, feature_list, schema)
    X = X.reset_index(drop=True)
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int).reset_index(drop=True)
    groups = df[group].astype(str).str.strip().reset_index(drop=True)

    perf = None
    if cv:
        perf = grouped_cv_performance(model, X, y, groups, verbose=verbose, label=label)

    clf = _fit_for_shap(model, X, y)
    feat_signed, idx = signed_feature_matrix(clf, X, colmap, sample=matrix_sample,
                                             verbose=verbose, label=label)
    ranking = ranking_from_matrix(feat_signed)

    meta = pd.DataFrame({"id_ruta": groups.loc[idx].values,
                         "y": y.loc[idx].values})
    feat32 = feat_signed.reset_index(drop=True).astype(np.float32)
    matrix = pd.concat([meta.reset_index(drop=True), feat32], axis=1)

    return dict(matrix=matrix, ranking=ranking, performance=perf,
                n=int(len(df)), n_shap_rows=int(len(feat_signed)),
                n_features=int(ranking.shape[0]), model=model,
                feature_cols=list(feat_signed.columns))


def _spearman(a, b):
    ra = pd.Series(a).rank()
    rb = pd.Series(b).rank()
    return float(ra.corr(rb, method="pearson"))


def compare_to_heritage(new_ranking, heritage_csv, top_k=10):
    her = pd.read_csv(heritage_csv)
    common = [f for f in new_ranking.feature if f in set(her.feature)]
    if len(common) < 3:
        return dict(status="tak_terbandingkan", n_common=len(common))
    nr = new_ranking.set_index("feature").loc[common, "rank"]
    hr = her.set_index("feature").loc[common, "rank"]
    top_new = set(new_ranking.feature.head(top_k))
    top_her = set(her.feature.head(top_k))
    jac = len(top_new & top_her) / len(top_new | top_her)
    return dict(status="ok", n_common=len(common),
                spearman_rank=round(_spearman(nr.values, hr.values), 4),
                jaccard_topk=round(jac, 4), top_k=top_k)


def ranking_convergence(feat_signed, sizes=None, repeats=3, top_k=10, seed=42,
                        full_ranking=None):
    n = len(feat_signed)
    if full_ranking is None:
        full_ranking = ranking_from_matrix(feat_signed)
    full_rank = full_ranking.set_index("feature")["rank"]
    full_top = set(full_ranking.feature.head(top_k))
    if sizes is None:
        base = [2000, 4000, 8000, 12000, 20000]
        sizes = [s for s in base if s < n] + [n]
    else:
        sizes = [s for s in sizes if s < n] + [n]
    rng = np.random.default_rng(seed)
    rows = []
    for s in sizes:
        if s >= n:
            rows.append(dict(size=n, spearman_mean=1.0, spearman_sd=0.0,
                             jaccard_topk_mean=1.0, jaccard_topk_sd=0.0, penuh=True))
            continue
        sp, jc = [], []
        for _ in range(repeats):
            idx = rng.choice(n, size=s, replace=False)
            rk = ranking_from_matrix(feat_signed.iloc[idx])
            common = [f for f in rk.feature if f in full_rank.index]
            a = rk.set_index("feature").loc[common, "rank"]
            b = full_rank.loc[common]
            sp.append(_spearman(a.values, b.values))
            top_s = set(rk.feature.head(top_k))
            jc.append(len(top_s & full_top) / len(top_s | full_top))
        rows.append(dict(size=int(s),
                         spearman_mean=round(float(np.mean(sp)), 5),
                         spearman_sd=round(float(np.std(sp)), 5),
                         jaccard_topk_mean=round(float(np.mean(jc)), 4),
                         jaccard_topk_sd=round(float(np.std(jc)), 4),
                         penuh=False))
    return pd.DataFrame(rows)


def convergence_point(conv_df, spearman_thr=0.99, jaccard_thr=1.0):
    ok = conv_df[(conv_df.spearman_mean >= spearman_thr) &
                 (conv_df.jaccard_topk_mean >= jaccard_thr) & (~conv_df.penuh)]
    if len(ok):
        return int(ok.iloc[0]["size"])
    return int(conv_df.iloc[-1]["size"])
