# Phase 2. Determinant domains and comparability

## Purpose

Phase 1 measures attribution for individual determinants. Phase 2 defines the groups
in which those determinants are read, and establishes which groups can legitimately be
compared across developmental stages and across surveys.

This phase produces no model output. It produces a dictionary and a comparability
table, both of which are inputs to every later phase.

## Why the dictionary is fixed before any result is seen

The mapping from determinants to domains was fixed before attribution shares were
computed. If domains were adjusted after seeing which grouping produced a larger
difference, the resulting difference would be an artefact of the grouping choice. The
dictionary is therefore a prespecification, not a tuning parameter.

The seven domains follow the layered causal framework for child undernutrition:
determinants at birth, maternal, socioeconomic and assets, environment and sanitation,
health services, feeding, and child morbidity.

## Boundary variables

Nine determinants can defensibly belong to more than one domain. Household health
insurance ownership, for example, marks socioeconomic position and also potential
access to care. Maternal education marks household socioeconomic position and also a
maternal characteristic.

Each such determinant is given a primary placement and an alternative placement. Both
mappings are carried forward, and Phase 5 repeats the entire analysis under the
alternative mapping. The purpose is not to choose the correct placement, which is a
question the data cannot settle, but to report how much the result depends on the
choice.

## Comparability

A domain can only be compared across developmental stages if its determinants are
measured in both stages. A domain can only be claimed to replicate across surveys if
its determinants are measured in all three. The comparability table records, for each
domain, how many determinants survive each restriction.

Three statuses are assigned:

- **Comparable**: at least three determinants present in the common analytic space.
- **Narrow basis**: one or two determinants present. Direction may be reported, but the
  domain is too thin to carry a claim.
- **Not comparable**: no determinants present. The domain is outside the scope of
  inference and is reported as such rather than silently dropped.

The feeding domain is not comparable, because all three surveys administer feeding
items only to children under 24 months. This is a property of the measurement
instrument, not of the analysis, and it means that of the two proximal causes of
undernutrition in the conceptual framework, only disease can be examined across stages.

## The common analytic space

Comparing attribution across stages requires that the set of determinants entering the
comparison be identical. If membership differs, a change in what was measured is
confounded with a change in how attribution is distributed. The same requirement
applies across surveys, because a replication claim is meaningful only when the
compared quantity is the same.

The common analytic space is therefore the intersection: determinants measured in both
developmental stages and in all three surveys. Child age in months is present in every
space but is excluded throughout as a structural axis, since it defines the contrast
rather than competing within it; it enters neither the numerator nor the denominator of
any share.

## Inputs and outputs

Input: `output_harmonisasi/matriks_ketersediaan.csv` from the harmonization pipeline.

Outputs in `output_phase2/`:

| File | Contents |
|---|---|
| `domain_dictionary.csv` | determinant, primary domain, boundary flag, alternative domain |
| `domain_comparability.csv` | determinants per domain in each analytic space, with status |
| `common_analytic_space.json` | the determinant list defining the primary analytic space |

## Running

```bash
python phase2_runner.py
```
