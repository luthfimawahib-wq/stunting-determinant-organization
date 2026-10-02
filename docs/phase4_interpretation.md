# Phase 4. Interpretation

> **Note on the published article.** The article derived from this analysis does not use
> the hypothesis frame documented below. It reports the measured cross-stage differences
> and their robustness, and does not restate the verdicts on H1 to H3. The hypotheses and
> their verdicts are kept here as the record of what was planned and what was found,
> including the predictions that failed. See "Relation to the published article" in the
> README.

## Purpose

Phase 3 produces measured quantities: attribution shares by domain, their cross-stage
differences, and the exact decomposition of those differences into determinants.
Phase 4 does not compute new quantities. It evaluates those measurements against
commitments made before the analysis, and it sorts domains into tiers of evidence.

Separating this from Phase 3 is deliberate. Phase 3 reports; Phase 4 judges. Keeping
the two apart makes it visible that no measurement was adjusted after the judgement
was formed.

## Prespecified hypotheses

Three expectations were fixed before any attribution share was computed:

- **H1**: the birth-related attribution structure will show relative stability across
  developmental stages.
- **H2**: the socioeconomic attribution share will be higher in the older stage.
- **H3**: the maternal attribution share will be lower in the older stage.

Two properties of this evaluation matter.

First, the hypotheses are evaluated against **relative attribution share**, not against
causal magnitude. A domain whose share falls has not been shown to matter less for
growth. Attribution share is a ratio, so a determinant can retain an unchanged absolute
contribution while its share declines, if other determinants accumulate contribution
and the denominator grows.

Second, the hypotheses are reported as fixed, including the one that fails. H1 is not
renamed after the fact to match the result. Renaming a hypothesis to fit the finding is
hypothesising after the results are known, and it destroys the value of having
prespecified anything.

A hypothesis is marked **Not evaluable** rather than rejected when the domain rests on
fewer than two determinants in the analytic space, or when its direction is not
consistent across surveys. In those cases the evidence cannot bear on the claim in
either direction.

## Evidence tiers

Robustness is not a pass or fail verdict. Reporting it as binary would suggest that
surviving all tests makes a finding true, and that failing one makes it false. Neither
follows.

Phase 4 therefore sorts domains into tiers:

| Tier | Meaning |
|---|---|
| Robust | consistent in direction, large in magnitude, agreed by both architectures, and stable under the alternative boundary mapping |
| Directionally robust | consistent in direction and across surveys, but sensitive in magnitude or class to boundary placement |
| Boundary case | small magnitude, or the two model architectures disagree |
| Insufficient evidence | direction not consistent across the three surveys |
| Insufficient basis | domain rests on fewer than two determinants |

The tiers describe what the evidence supports, not what the finding is. A domain in the
directionally robust tier still carries a claim, but a narrower one: the direction can
be asserted, the magnitude cannot be asserted independently of the classification
choice.

## Inputs and outputs

Inputs: `output_phase2/domain_comparability.csv` and, from `output_phase3/`,
`pergeseran_headline.csv`, `klasifikasi_headline.csv`, `lintas_model.csv`, and
`sensitivitas_variabel_batas.csv`.

Outputs in `output_phase4/`:

| File | Contents |
|---|---|
| `hypotheses.csv` | each prespecified hypothesis, the measured difference, the verdict, and its basis |
| `evidence_tiers.csv` | each domain with its tier and the reason for that tier |

## Running

```bash
python phase4_runner.py
```
