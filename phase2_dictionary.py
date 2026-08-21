import json
import os
import pandas as pd

DOMAINS = [
    "Determinants at birth",
    "Maternal",
    "Socioeconomic and assets",
    "Environment and sanitation",
    "Health services",
    "Feeding",
    "Child morbidity",
]

STRUCTURAL = "Structural (excluded)"

DOMAIN_RULES = {
    "Determinants at birth": [
        "sex_child", "birth_weight_g", "birth_length_cm", "gestational_age_wk",
    ],
    "Maternal": [
        "age_mother_yr", "height_mother_cm", "weight_mother_kg", "gravida",
        "pregnancy_class",
    ],
    "Socioeconomic and assets": [
        "area_type", "edu_mother", "occupation_mother", "marital_status",
        "lighting_source", "jkn_owned", "building_ownership", "floor_area_m2",
        "asset_car", "asset_motorcycle", "asset_phone", "asset_computer",
        "asset_tv", "asset_fridge", "asset_washing_machine", "asset_gas",
        "asset_gold", "asset_land", "asset_livestock",
        "bansos_kks", "bansos_pkh", "bansos_bpnt", "bansos_blt",
    ],
    "Environment and sanitation": [
        "water_source", "toilet_type", "sanitation_own", "feces_disposal",
        "cooking_fuel",
    ],
    "Health services": [
        "anc_received", "anc_place", "anc_freq_mid_t1", "anc_freq_mid_t2",
        "anc_freq_mid_t3", "anc_hb_tested", "anc_lila_measured",
        "delivery_place", "delivery_assistant", "kia_book", "jkn_used",
        "ttd_received", "ttd_count", "pmt_received",
        "imm_bcg", "imm_hepb0", "imm_dpt1", "imm_dpt2", "imm_dpt3",
        "imm_dpt_boost", "imm_measles_9mo", "imm_measles_boost",
        "imm_vit_a_count", "weigh_freq_12mo", "height_measure_12mo",
    ],
    "Feeding": [
        "breastfed_current",
        "breastfed_ever",
        "colostrum_action",
        "food_cereal",
        "food_egg",
        "food_fish",
        "food_formula",
        "food_green_veg",
        "food_legume",
        "food_meat",
        "food_organ_meat",
        "food_vit_a_fruit",
        "food_vit_a_veg",
        "food_water",
        "imd_duration",
        "imd_skin_contact",
        "meal_frequency",
        "prelacteal_feed",
    ],
    "Child morbidity": [
        "diarrhea_1month", "ispa_1month", "pneumonia_1year", "tb_1year",
    ],
}

BOUNDARY = {
    "edu_mother": ("Socioeconomic and assets", "Maternal"),
    "occupation_mother": ("Socioeconomic and assets", "Maternal"),
    "marital_status": ("Socioeconomic and assets", "Maternal"),
    "area_type": ("Socioeconomic and assets", "Environment and sanitation"),
    "lighting_source": ("Socioeconomic and assets", "Environment and sanitation"),
    "jkn_owned": ("Socioeconomic and assets", "Health services"),
    "jkn_used": ("Health services", "Socioeconomic and assets"),
    "ttd_received": ("Health services", "Maternal"),
    "ttd_count": ("Health services", "Maternal"),
}

STRUCTURAL_FEATURES = ["age_child_months"]

SOURCES = ["ssgi22", "ssgi24", "ski23"]
COHORTS = ["baduta", "balita_tua"]
AVAIL_THRESHOLD = 0.5


def build_dictionary(available_features=None):
    rows = []
    for domain, feats in DOMAIN_RULES.items():
        for f in feats:
            if available_features is not None and f not in available_features:
                continue
            primary, alternative = BOUNDARY.get(f, (domain, ""))
            rows.append(dict(feature=f, domain=primary,
                             boundary_variable="yes" if f in BOUNDARY else "no",
                             alternative_domain=alternative))
    for f in STRUCTURAL_FEATURES:
        if available_features is None or f in available_features:
            rows.append(dict(feature=f, domain=STRUCTURAL,
                             boundary_variable="no", alternative_domain=""))
    out = pd.DataFrame(rows)
    return out.sort_values(["domain", "feature"]).reset_index(drop=True)


def cell_features(mat, source, cohort, threshold=AVAIL_THRESHOLD):
    ncol, avail = f"n_sumber_{cohort}", f"{source}_{cohort}"
    if ncol not in mat.columns or avail not in mat.columns:
        return []
    sel = mat[(mat[ncol] >= 2) & (mat[avail] >= threshold)]
    return sel.variabel.tolist()


def per_survey_intersection(mat, source, threshold=AVAIL_THRESHOLD):
    a = set(cell_features(mat, source, "baduta", threshold))
    b = set(cell_features(mat, source, "balita_tua", threshold))
    return a & b


def comparability(dictionary, mat, threshold=AVAIL_THRESHOLD):
    dom = dict(zip(dictionary.feature, dictionary.domain))
    per = {s: per_survey_intersection(mat, s, threshold) for s in SOURCES}
    full = set.intersection(*per.values()) if per else set()
    rows = []
    for d in DOMAINS:
        members = {f for f, dd in dom.items() if dd == d}
        row = dict(domain=d, dictionary=len(members))
        for s in SOURCES:
            row[s] = len(members & per[s])
        row["all_three"] = len(members & full)
        n = row["all_three"]
        row["status"] = ("Not comparable" if n == 0
                         else "Narrow basis" if n < 3 else "Comparable")
        rows.append(row)
    out = pd.DataFrame(rows)
    total = dict(domain="Total determinants",
                 dictionary=int(out.dictionary.sum()),
                 all_three=int(out.all_three.sum()), status="")
    for s in SOURCES:
        total[s] = int(out[s].sum())
    return pd.concat([out, pd.DataFrame([total])], ignore_index=True), full


def write(dictionary, comp, full, outdir):
    os.makedirs(outdir, exist_ok=True)
    dictionary.to_csv(os.path.join(outdir, "domain_dictionary.csv"), index=False)
    comp.to_csv(os.path.join(outdir, "domain_comparability.csv"), index=False)
    with open(os.path.join(outdir, "common_analytic_space.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(n_determinants=len(
            [x for x in full if x not in STRUCTURAL_FEATURES]),
            n_columns=len(full),
            features=sorted(full)), f, indent=2)
