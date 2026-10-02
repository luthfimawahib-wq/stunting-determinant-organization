# Composition of Attributed Risk in Stunting Across Developmental Stages

Analysis code for a study of how the **composition of attributed risk** in stunting
differs between two developmental stages, using three Indonesian national surveys
(SSGI 2022, SSGI 2024, SKI 2023). The question is not which determinants cause
stunting, but whether the relative weight carried by groups of determinants is the same
throughout early childhood. The published article calls this quantity the composition of
attributed risk; earlier versions of this repository called it the attribution structure.
The measured quantity is the same in both.

Machine learning is used here as a **measurement instrument**, not as a predictor.
Models are fitted to quantify how attribution is distributed among determinants; their
discrimination is reported as context for measurement quality, not as a finding.

> **DOI:** [`10.5281/zenodo.22039068`](https://doi.org/10.5281/zenodo.22039068)

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
python phase1_runner.py              # twelve measurement units
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

## Model settings

Hyperparameters are fixed a priori and identical across all twelve measurement units,
because the objective is comparability of measurement rather than predictive
optimization. They are defined in `phase1_measure.py`; the `CONFIG` block of
`phase1_runner.py` carries paths, cell lists, and sample sizes only.

| Setting | Value |
|---|---|
| XGBoost | n_estimators 400, max_depth 5, learning_rate 0.05, subsample 0.8, colsample_bytree 0.8, lambda 1.0, alpha 0.0, gamma 0.0, objective binary:logistic |
| Random forest | n_estimators 600, max_depth None, min_samples_leaf 20, max_features sqrt |
| Cross-validation | five folds, StratifiedGroupKFold stratified on the outcome and grouped by `id_ruta`, falling back to GroupKFold |
| Class balancing | SMOTE inside training folds only, k_neighbors min(5, minority - 1), applied when the minority class exceeds five rows |
| SHAP sample | up to 40 000 rows per cell (`matrix_sample`); the smallest cell uses all 30 333 |
| Random seed | 42 throughout cross-validation, SMOTE, and sampling |
| Python | 3.10.4 |

AUC and average precision are computed per fold and stored with each unit. They describe
measurement quality and are not a finding of the study.

## Relation to the published article

The article reports the same measurements under a different frame. Phase 4 here evaluates
three hypotheses stated before the analysis and sorts domains into evidence tiers. The
article reports no hypothesis verdicts. It reports the measured cross-stage differences,
the eight robustness analyses whose decision rules were fixed in advance, and how much
evidential weight each finding can carry.

Two consequences are worth stating plainly. H1 expected relative stability in the
birth-related share, and the measurement shows the largest decrease of any domain. H3
expected the maternal share to decrease at the older stage, and the measurement shows an
increase of 1.5 share points whose sign is inconsistent across surveys; the article
reports that domain as not supporting a claim. Both verdicts are in the Phase 4 output
(`hypotheses.csv`) and neither is restated in the article, because the article argues
from the measured differences and their robustness rather than from hypothesis verdicts.

The vocabulary also differs. The evidence tiers used here (robust, directionally robust,
boundary case, insufficient evidence, insufficient basis) describe how much weight a
finding can bear. The class labels used in the article (stage-specific, transition,
invariant) come from the three boundary methods of Phase 3 and serve only to organize the
pattern; the article states that they are not the basis of any claim.

## License and citation

Released under the MIT License (see `LICENSE`). If you use this code, please cite the
DOI release (see `CITATION.cff` and `.zenodo.json`).
