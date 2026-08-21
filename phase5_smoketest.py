import os, shutil, numpy as np, pandas as pd, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase3_shares as D, phase3_runner as R3, phase5_robust as R, phase5_runner as R5

OUT="/tmp/f5"; MD=os.path.join(OUT,"matrix")
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(MD, exist_ok=True)
kam=pd.read_csv("output_phase2/domain_dictionary.csv")
kam=kam[~kam.domain.str.contains("struktural",case=False,na=False)]
f2d=dict(zip(kam.feature,kam.domain)); feats=list(kam.feature)
rng=np.random.default_rng(7)
base={"Biologis Kelahiran":(0.60,0.25),"Sosioekonomi & Aset":(0.15,0.15),
      "Maternal":(0.20,0.20),"Sanitasi & Lingkungan":(0.15,0.15),
      "Layanan Kesehatan":(0.16,0.16),"Pemberian Makan":(0.20,0.20),
      "Morbiditas Anak (Penyakit Infeksi)":(0.20,0.20)}
def sig(dom,coh,src):
    b=base.get(dom,(0.2,0.2)); v=b[0 if coh=="baduta" else 1]
    if dom=="Sosioekonomi & Aset" and src=="ssgi24" and coh=="balita_tua": v=0.55
    return v
for s in D.SOURCES:
    for c in D.COHORTS:
        for m in ["xgb","rf"]:
            n=1500; cols={"id_ruta":rng.integers(0,250,n).astype(str),"y":rng.integers(0,2,n)}
            for f in feats: cols[f]=rng.normal(0,sig(f2d[f],c,s),n).astype("float32")
            pd.DataFrame(cols).to_parquet(os.path.join(MD,f"{s}_{c}_{m}.parquet"),index=False)

print("=== UJI SINTETIS FASE 5 ===")
c3=dict(R3.CONFIG); c3.update(matrix_dir=MD, kamus_path="output_phase2/domain_dictionary.csv",
                              output_dir=os.path.join(OUT,"f3"), bootstrap_B=30)
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()): R3.main(c3)
print("[prasyarat] Fase 3 sintetis selesai")

c5=dict(R5.CONFIG); c5.update(matrix_dir=MD, kamus_path="output_phase2/domain_dictionary.csv",
                              phase3_dir=os.path.join(OUT,"f3"), output_dir=os.path.join(OUT,"f5"),
                              B=40)
R5.main(c5)

print("\n=== VERIFIKASI ===")
lo=pd.read_csv(os.path.join(OUT,"f5","loso_domain.csv"), index_col=0)
bio=lo.loc["Biologis Kelahiran","arah_bertahan"]
ses=lo.loc["Sosioekonomi & Aset"]
print(f"Biologis (tertanam konsisten) arah_bertahan = {bio}  (harap ya)")
print(f"Sosioekonomi (tertanam HANYA di ssgi24): penuh={ses['penuh_3_sumber']:+.4f} "
      f"tanpa_ssgi24={ses['tanpa_ssgi24']:+.4f}")
assert bio=="ya", "LOSO gagal pada struktur konsisten"
assert abs(ses["tanpa_ssgi24"]) < abs(ses["penuh_3_sumber"])/2, "LOSO tak menangkap ketergantungan sumber"
for f in ["rekap_ketahanan.csv","ruang_fitur.csv","presisi_bootstrap_domain.csv",
          "gambar_loso.png","gambar_rekap_ketahanan.png"]:
    assert os.path.exists(os.path.join(OUT,"f5",f)), f
print("\nLULUS: LOSO menangkap ketergantungan sumber; seluruh artefak & gambar terbuat.")
