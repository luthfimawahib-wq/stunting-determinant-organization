import os
import pandas as pd
import phase3_shares as D
import phase5_weights as B

CONFIG = dict(
    parquet_path="output_harmonisasi/stunting_harmonized.parquet",
    matrix_dir="output_phase1/matrix",
    kamus_path="output_phase2/domain_dictionary.csv",
    output_dir="output_phase5",
    headline_model="xgb",
    source_col="source_flag",
    kohort_col="kohort",
    source_flag_map={"ssgi22": "ssgi22", "ssgi24": "ssgi24", "ski23": "ski23"},
    weight_col="svy_weight",
)


def main(cfg):
    print("=" * 74)
    print("SENSITIVITAS BERBOBOT SURVEI (Fase 5, komponen tambahan)")
    print("=" * 74)
    od = cfg["output_dir"]
    os.makedirs(od, exist_ok=True)
    primer, _, _ = D.load_kamus(cfg["kamus_path"])
    md, model = cfg["matrix_dir"], cfg["headline_model"]

    print(f"Memuat Parquet: {cfg['parquet_path']}")
    df = pd.read_parquet(cfg["parquet_path"])
    if cfg["weight_col"] not in df.columns:
        raise SystemExit(f"Kolom bobot '{cfg['weight_col']}' tidak ada. "
                         f"Kolom yang mengandung 'weight': "
                         f"{[c for c in df.columns if 'weight' in c.lower() or 'wgt' in c.lower()]}")
    print(f"  bentuk: {df.shape}; kolom bobot: {cfg['weight_col']}")

    full = D.full_intersection_feats(md, model)
    feats = [f for f in full if primer.get(f) is not None]
    print(f"Ruang headline: {len(feats)} determinan pada irisan penuh.\n")

    print("Merekonstruksi baris sampel dan menempelkan bobot (dengan verifikasi id_ruta)...")
    shift_w, shift_u, audit = B.weighted_shift(md, df, cfg, model, primer, feats)
    print("  verifikasi id_ruta: LULUS pada seluruh sel.\n")
    print("Sebaran bobot per sel:")
    print(audit.to_string(index=False))

    cmp = B.compare(shift_w, shift_u)
    shift_w.round(5).to_csv(os.path.join(od, "bobot_pergeseran.csv"))
    audit.to_csv(os.path.join(od, "bobot_audit.csv"), index=False)
    cmp.to_csv(os.path.join(od, "bobot_perbandingan.csv"))

    print("\nPergeseran pangsa domain: berbobot dibanding tak berbobot")
    print(cmp.to_string())

    beda = cmp[cmp.arah_sama == "tidak"]
    print(f"\nDomain yang BERUBAH ARAH di bawah pembobotan: "
          f"{list(beda.index) if len(beda) else 'tidak ada'}")
    print(f"Artefak tersimpan di {od}/")
    print("Catatan: bobot menangani keterwakilan populasi. Objek ukur analisis utama adalah "
          "struktur atribusi relatif dalam tiap sel, sehingga hasil berbobot berperan sebagai "
          "uji ketahanan, bukan sebagai pengganti hasil utama.")


if __name__ == "__main__":
    main(CONFIG)
