# Stunting Determinant Organization Across Developmental Stages

Analysis code for a study of how the **attribution structure** of stunting determinants
differs between two developmental stages, using three Indonesian national surveys
(SSGI 2022, SSGI 2024, SKI 2023). The question is not which determinants cause
stunting, but whether the relative weight carried by groups of determinants is the same
throughout early childhood.

Machine learning is used here as a **measurement instrument**, not as a predictor.
Models are fitted to quantify how attribution is distributed among determinants; their
discrimination is reported as context for measurement quality, not as a finding.

> **DOI:** `10.5281/zenodo.XXXXXXX`

## Data availability

The microdata are governed by the Ministry of Health of the Republic of Indonesia and
are **not redistributed** here. See `raw_data/README.md` for how to request access. The
harmonized master Parquet is a derivative of that microdata and is never committed.

This repository does not generate data. The harmonized master Parquet, its encoding
schema, and the feature availability matrix all come from the companion repository
[`stunting-harmonization`](https://github.com/luthfimawahib-wq/stunting-harmonization)
(DOI `10.5281/zenodo.22015038`), which also ships the **synthetic data generator**
(`data_uji/buat_data_uji.py`). That generator writes a Parquet with the same schema and
structural-missingness pattern as the real file, so the whole analysis chain here can be
executed and verified without restricted data. Any statistical relationship in the
synthetic data is artificial and must not be interpreted as a finding.

## Repository structure

```
stunting-determinant-organization/
  README.md
  LICENSE                       # MIT
  requirements.txt
  .gitignore
  .zenodo.json                  # Zenodo record metadata (DOI)
  CITATION.cff                  # citation metadata

  phase1_measure.py             # SHAP measurement engine (signed matrices)
  phase1_runner.py              # runs the twelve measurement units
  phase1_smoketest.py           # synthetic check of the measurement engine

  phase2_dictionary.py          # determinant-to-domain dictionary, comparability
  phase2_runner.py

  phase3_shares.py              # domain attribution shares and decomposition
  phase3_runner.py
  phase3_smoketest.py

  phase4_interpretation.py      # prespecified hypotheses, evidence tiers
  phase4_runner.py

  phase5_robust.py              # leave-one-survey-out, feature space, rank stability
  phase5_runner.py
  phase5_weights.py             # sampling-weighted recomputation
  phase5_weights_runner.py
  phase5_recap.py               # robustness recap table and figure
  phase5_smoketest.py

  docs/
    phase2_domains.md
    phase4_interpretation.md
  raw_data/                     # (git-ignored) restricted microdata goes here
    README.md
  output_harmonisasi/           # inputs from the harmonization repository land here
    .gitkeep                    # stunting_harmonized.parquet, skema_encoding.json,
                                # matriks_ketersediaan.csv (git-ignored)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Without microdata (synthetic run)

Generate the synthetic Parquet in the harmonization repository first, then copy the
three files it produces into `output_harmonisasi/` here:

```bash
# in the stunting-harmonization repository
python data_uji/buat_data_uji.py
# then copy stunting_harmonized.parquet, skema_encoding.json,
# and matriks_ketersediaan.csv into output_harmonisasi/ of this repository
```

The chain below then runs end to end and takes a few minutes.

```bash
python phase1_runner.py              # twelve measurement units (set parquet_path first)
python phase2_runner.py              # domain dictionary and comparability
python phase3_runner.py              # attribution shares and decomposition
python phase4_runner.py              # hypotheses and evidence tiers
python phase5_runner.py              # robustness
python phase5_weights_runner.py      # sampling-weighted sensitivity
python phase5_recap.py               # robustness recap including the weighted row
```

For a synthetic run, lower `matrix_sample` in the `CONFIG` of `phase1_runner.py`; 1500
is enough for a structural check. `phase5_weights_runner.py` reads the sample size back
from the phase 1 manifest, so it stays consistent automatically.

### With real microdata

Place `stunting_harmonized.parquet`, `skema_encoding.json`, and
`matriks_ketersediaan.csv` in `output_harmonisasi/`, then run the same sequence with the
default `CONFIG` values.

Phase 1 is the expensive step. It supports `--only`, `--resume`, and
`--adopt-existing`, and writes a manifest per unit recording the code version and
sample size, so a resumed run never mixes outputs from different code versions.

## The five phases

| Phase | Question | Key output |
|---|---|---|
| 1 | How is attribution distributed among determinants in each cell? | signed SHAP matrices, one per cell and model |
| 2 | Which determinants form which domains, and which domains are comparable? | dictionary, comparability table, common analytic space |
| 3 | How does each domain's attribution share differ across stages? | domain differences, exact determinant decomposition |
| 4 | What do the prespecified hypotheses say, and how strong is the evidence? | hypothesis verdicts, evidence tiers |
| 5 | Does the result survive perturbation of its assumptions? | eight robustness tests and a recap |

Phases 2 and 4 are documented in `docs/`, because their reasoning is not obvious from
the code alone.

## Two design decisions worth knowing

**Expensive artefacts are saved; cheap ones are derived.** SHAP is computed once per
cell and model, and the signed per-row matrix is stored. Rankings, domain shares,
bootstrap resamples, and the weighted recomputation are all derived from those stored
matrices. SHAP is never computed twice.

**The common analytic space is held identical on both axes.** Comparing attribution
across stages requires that the determinant set be the same; otherwise a change in what
was measured is confounded with a change in how attribution is distributed. The same
applies across surveys, because a replication claim is meaningful only when the
compared quantity is the same.

## Anti-leakage protocol

- Child anthropometry (`height_child_cm`, `weight_child_kg`) is removed from the
  predictor set, since it enters the definition of the outcome.
- Cross-validation folds are grouped by household, because children in one household
  are not independent.
- Class balancing is applied only within training folds; SHAP is computed on real rows,
  never on oversampled ones.
- Model hyperparameters are fixed a priori and identical across all cells, because the
  objective is comparability of measurement, not predictive optimization.

## License and citation

Released under the MIT License (see `LICENSE`). If you use this code, please cite the
DOI release (see `CITATION.cff` and `.zenodo.json`).
