import os
import pandas as pd
import phase2_dictionary as P2

CONFIG = dict(
    availability_matrix="output_harmonisasi/matriks_ketersediaan.csv",
    output_dir="output_phase2",
    threshold=P2.AVAIL_THRESHOLD,
)


def main(cfg):
    print("=" * 74)
    print("PHASE 2. DETERMINANT DOMAINS AND COMPARABILITY")
    print("=" * 74)
    mat = pd.read_csv(cfg["availability_matrix"])
    available = set(mat.variabel)
    dictionary = P2.build_dictionary(available)
    comp, full = P2.comparability(dictionary, mat, cfg["threshold"])

    n_det = int((dictionary.domain != P2.STRUCTURAL).sum())
    n_bound = int((dictionary.boundary_variable == "yes").sum())
    print(f"Dictionary: {len(P2.DOMAINS)} domains, {n_det} determinants, "
          f"{n_bound} boundary variables.")
    print(f"Common analytic space: {len(full)} columns, "
          f"{len([f for f in full if f not in P2.STRUCTURAL_FEATURES])} determinants.\n")
    print(comp.to_string(index=False))

    P2.write(dictionary, comp, full, cfg["output_dir"])
    print(f"\nWritten to {cfg['output_dir']}/: domain_dictionary.csv, "
          "domain_comparability.csv, common_analytic_space.json")


if __name__ == "__main__":
    main(CONFIG)
