import numpy as np
import pandas as pd
import phase1_measure as u

UP = "/mnt/user-data/uploads"
RNG = np.random.default_rng(7)


def synth_cell(feats, schema, n=6000, n_groups=1500):
    types = u.feature_types(schema)
    cols = {}
    signal = np.zeros(n)
    for f in feats:
        t = types.get(f, "kontinu")
        if t == "nominal":
            k = 4
            v = RNG.integers(0, k, n).astype(float)
            cols[f] = v
            if f == "jkn_owned":
                signal += 0.9 * (v == 0)
        elif t in ("biner",):
            v = RNG.integers(0, 2, n).astype(float)
            cols[f] = v
        elif t == "ordinal":
            cols[f] = RNG.integers(0, 5, n).astype(float)
        else:
            v = RNG.normal(0, 1, n)
            cols[f] = v
            if f == "birth_weight_g":
                signal += 1.6 * v
            if f == "height_mother_cm":
                signal += 0.8 * v
    df = pd.DataFrame(cols)
    p = 1 / (1 + np.exp(-(signal - signal.mean())))
    df["stunting_binary"] = (RNG.random(n) < p).astype(int)
    df["id_ruta"] = RNG.integers(0, n_groups, n).astype(str)
    df["source_flag"] = "ssgi22"
    df["kohort"] = "baduta"
    if "gestational_age_wk" in df.columns:
        m = RNG.random(n) < 0.2
        df.loc[m, "gestational_age_wk"] = np.nan
    return df


def main():
    print("=" * 70)
    print("UJI SINTETIS FASE 1")
    print("=" * 70)
    schema = u.load_schema(f"{UP}/skema_encoding.json")
    mat = pd.read_csv(f"{UP}/matriks_ketersediaan.csv")

    expect = {("ssgi22", "baduta"): 75, ("ssgi24", "baduta"): 57, ("ski23", "baduta"): 79,
              ("ssgi22", "balita_tua"): 46, ("ssgi24", "balita_tua"): 56, ("ski23", "balita_tua"): 57}
    print("\n[1] frozen_cell_features:")
    ok_feat = True
    for (s, c), exp in expect.items():
        got = len(u.frozen_cell_features(mat, schema, s, c))
        tick = "OK" if got == exp else "BEDA"
        ok_feat = ok_feat and got == exp
        print(f"    {s}-{c}: {got} (harap {exp}) {tick}")

    feats = u.frozen_cell_features(mat, schema, "ssgi22", "baduta")
    df = synth_cell(feats, schema, n=6000)
    print(f"\n[2] data sintetis: {df.shape}, prevalensi={df.stunting_binary.mean():.3f}")

    u.SHAP_N_JOBS = 1
    for model in ["xgb", "rf"]:
        res = u.measure_cell(df, feats, schema, model=model, matrix_sample=6000,
                             verbose=False, label=f"{model}")
        M = res["matrix"]
        R = res["ranking"]
        fc = res["feature_cols"]

        rederived = u.ranking_from_matrix(M[fc].astype(float))
        same_order = list(rederived.feature) == list(R.feature)
        max_abs = float((rederived.set_index("feature").mean_abs_shap
                         - R.set_index("feature").mean_abs_shap).abs().max())
        top5 = list(R.feature.head(5))
        sig_ok = ("birth_weight_g" in top5) and (
            "jkn_owned" in list(R.feature.head(10)))
        has_meta = {"id_ruta", "y"}.issubset(M.columns)
        has_dir = "mean_signed_shap" in R.columns and R.mean_signed_shap.abs().sum() > 0
        is_f32 = str(M[fc[0]].dtype) == "float32"

        print(f"\n[{model}] matriks {M.shape}  AUC={res['performance']['auc_mean']:.3f}")
        print(f"    invarian peringkat=turunan-matriks: {same_order} (selisih maks {max_abs:.2e})")
        print(f"    sinyal tertanam di atas (birth_weight_g top5, jkn_owned top10): {sig_ok}")
        print(f"    matriks punya id_ruta&y: {has_meta} | arah(signed): {has_dir} | float32: {is_f32}")
        print(f"    top5: {top5}")

    print("\n[6] pembanding warisan:")
    res = u.measure_cell(df, feats, schema, model="xgb", matrix_sample=6000, cv=False)
    res["ranking"].to_csv("/tmp/heritage_fake.csv", index=False)
    cmp = u.compare_to_heritage(res["ranking"], "/tmp/heritage_fake.csv")
    print(f"    identik dengan diri sendiri: rho={cmp['spearman_rank']} jac={cmp['jaccard_topk']} "
          f"(harap 1.0/1.0)")

    print("\n[7] konvergensi peringkat (dari matriks, tanpa SHAP baru):")
    conv = u.ranking_convergence(res["matrix"][res["feature_cols"]].astype(float),
                                 sizes=[1000, 2000, 4000], repeats=3, full_ranking=res["ranking"])
    mono = conv.spearman_mean.is_monotonic_increasing or (conv.spearman_mean.iloc[-1] >= conv.spearman_mean.iloc[0])
    pt = u.convergence_point(conv)
    print(conv.to_string(index=False))
    print(f"    Spearman naik menuju penuh: {mono} | titik konvergen deskriptif: {pt}")
    assert conv.iloc[-1].penuh and conv.iloc[-1].spearman_mean == 1.0, "baris penuh harus 1.0"

    print("\nSELESAI. Semua jalur pengukuran Fase 1 berjalan.")


if __name__ == "__main__":
    main()
