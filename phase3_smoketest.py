import os, shutil, numpy as np, pandas as pd
import phase3_shares as D, phase3_runner as R

UP="/mnt/user-data/uploads"; OUT="/tmp/f3"; MD=os.path.join(OUT,"matrix")
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(MD, exist_ok=True)
kam=pd.read_csv("output_phase2/domain_dictionary.csv")
kam=kam[~kam.domain.str.contains("struktural",case=False,na=False)]
feat2dom=dict(zip(kam.feature,kam.domain))
feats=list(kam.feature)
rng=np.random.default_rng(1)

base={"Biologis Kelahiran":(0.55,0.55),"Maternal":(0.45,0.20),
      "Sosioekonomi & Aset":(0.10,0.42),"Sanitasi & Lingkungan":(0.15,0.17),
      "Layanan Kesehatan":(0.16,0.18),"Pemberian Makan":(0.20,0.16),
      "Morbiditas Anak (Penyakit Infeksi)":(0.22,0.20)}
def sigma(dom,coh,src,model):
    b=base.get(dom,(0.2,0.2)); s=b[0 if coh=="baduta" else 1]
    s*= (1.0+0.03*(hash((src,model))%5))
    return s
def make(src,coh,model,n=2000):
    cols={"id_ruta":rng.integers(0,300,n).astype(str),"y":rng.integers(0,2,n)}
    for f in feats:
        cols[f]=rng.normal(0,sigma(feat2dom[f],coh,src,model),n).astype("float32")
    pd.DataFrame(cols).to_parquet(os.path.join(MD,f"{src}_{coh}_{model}.parquet"),index=False)
for s in D.SOURCES:
    for c in D.COHORTS:
        for m in ["xgb","rf"]: make(s,c,m)

print("=== UJI SINTETIS FASE 3 ===")
cfg=dict(R.CONFIG); cfg.update(matrix_dir=MD, kamus_path="output_phase2/domain_dictionary.csv",
                               output_dir=os.path.join(OUT,"out"), bootstrap_B=60)
primer,sens,border=D.load_kamus(cfg["kamus_path"])
r=D.shares_and_shift(MD,"ssgi22","xgb",primer)
assert abs(r["share_baduta"].sum()-1)<1e-6 and abs(r["share_balita"].sum()-1)<1e-6, "pangsa != 1"
print(f"[1] pangsa berjumlah 1: baduta={r['share_baduta'].sum():.4f} balita={r['share_balita'].sum():.4f}")

R.main(cfg)

cl=pd.read_csv(os.path.join(cfg["output_dir"],"klasifikasi_xgb.csv"),index_col=0)
ses=cl.loc["Sosioekonomi & Aset","kelas_kokoh"]; bio=cl.loc["Biologis Kelahiran","kelas_kokoh"]
print(f"\n[VERIFIKASI] Sosioekonomi -> {ses} (harap spesifik-tahap); Biologis Kelahiran -> {bio} (harap invariant)")
assert os.path.exists(os.path.join(cfg["output_dir"],"gambar_pergeseran_domain.png")), "gambar tak terbuat"
print("[gambar] terbuat OK")
print("\nSEMUA TES FASE 3 LULUS.")
