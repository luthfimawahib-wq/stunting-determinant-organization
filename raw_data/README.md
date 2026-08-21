# Restricted microdata

This repository does not read raw microdata. It reads the **harmonized master Parquet**
produced by the companion repository
[`stunting-harmonization`](https://github.com/luthfimawahib-wq/stunting-harmonization)
(DOI `10.5281/zenodo.22015038`).

The source microdata for the Indonesian Nutritional Status Survey (SSGI) 2022 and 2024
and the Indonesian Health Survey (SKI) 2023 are held by the Ministry of Health of the
Republic of Indonesia and are **not publicly redistributable**. They may be requested
from the Ministry under its data-access terms.

## What to place in `output_harmonisasi/`

| File | Produced by |
|---|---|
| `stunting_harmonized.parquet` | `harmonisasi_pipeline.py` in the harmonization repository |
| `skema_encoding.json` | the same pipeline |
| `matriks_ketersediaan.csv` | `matriks_ketersediaan.py` in the harmonization repository |

## Running without the microdata

The harmonization repository ships a synthetic data generator that reproduces the
schema, cohort structure, and structural-missingness pattern of the real file:

```bash
# in the stunting-harmonization repository
python data_uji/buat_data_uji.py
```

Copy the three files it produces into `output_harmonisasi/` here. The entire analysis
chain then runs unchanged. Any statistical relationship in the synthetic data is
artificial and must not be interpreted as a finding.
