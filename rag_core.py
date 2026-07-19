

import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# LOCAL SETUP (no Colab). API keys come from .env or environment.
# ============================================================
_HERE = Path(__file__).resolve().parent        # 0715/
_BASE = _HERE.parent                           # iam+llm/


def _load_dotenv(path=None):
    """Minimal .env loader (KEY=VALUE lines); does not override existing env vars."""
    path = path or (_HERE / ".env")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v

_load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError(
        "Missing OPENAI_API_KEY. Put it in 0715/.env (OPENAI_API_KEY=sk-...) "
        "or export it before running."
    )

from openai import OpenAI
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
client = openai_client

# Gemini is optional: only initialized if GEMINI_API_KEY is set.
gemini_client = None
if os.environ.get("GEMINI_API_KEY"):
    try:
        from google import genai
        gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    except ImportError:
        pass

_DATA = _HERE / "data"                         # 0715/data/

EMBED_MODEL = "text-embedding-3-large"  # change to text-embedding-3-small if needed
FINGERPRINT_PATH = os.environ.get(
    "IAM_LLM_FINGERPRINT_PATH",
    str(_DATA / "model_fingerprint.xlsx"),
)

CACHE_DIR = os.environ.get(
    "IAM_LLM_CACHE_DIR",
    str(_HERE / "_cache"),
)
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
SCEN_EMB_CACHE = os.path.join(CACHE_DIR, "cache_scenario_desc_embeddings.json")
FP_EMB_CACHE   = os.path.join(CACHE_DIR, "cache_model_fingerprint_embeddings.json")

# Retrieval budgets; override per-run via env vars, e.g.:
#   IAM_LLM_TOPK_SCEN=0  -> no same-model-family neighbors (STEP1 off)
#   IAM_LLM_TOPK_MODEL=0 -> no same-scenario neighbors (STEP2 off)
TOPK_SCEN  = int(os.environ.get("IAM_LLM_TOPK_SCEN",  "3"))
TOPK_MODEL = int(os.environ.get("IAM_LLM_TOPK_MODEL", "3"))
TOPK_CROSS = int(os.environ.get("IAM_LLM_TOPK_CROSS", "3"))
RUN_STEP3  = False

# Which sheet of model_fingerprint.xlsx to use (recorded in the eval report)
FP_SHEET_NAME = "fingerprint"
# Whether the TARGET's scenario description text is given to the LLM
INCLUDE_SCENARIO_DESC = True

VARS_14 = [
    "Primary Energy|Coal",
    "Primary Energy|Coal|w/ CCS",
    "Primary Energy|Coal|w/o CCS",
    "Primary Energy|Gas",
    "Primary Energy|Gas|w/ CCS",
    "Primary Energy|Gas|w/o CCS",
    "Primary Energy|Oil",
    "Primary Energy|Oil|w/ CCS",
    "Primary Energy|Oil|w/o CCS",
    "Primary Energy|Solar",
    "Primary Energy|Wind",
    "Primary Energy|Hydro",
    "Primary Energy|Nuclear",
    "Primary Energy|Biomass",
]
YEARS = [str(y) for y in range(2010, 2110, 10)]  # 2010..2100

"""Set target"""

TARGET = {
    "region": "Countries of centrally-planned Asia; primarily China",          # must exactly match the region value in sheet
    "model_family": "IMAGE",                # must exactly match the model value in sheet
    "scenario": "CO_BAU",     # must exactly match the scenario value in sheet
}

# New 6351-based 8:2 split (no val). Old 4085 splits remain in data/split_outputs/.
BATCH_SHEET_PATH = os.environ.get(
    "IAM_LLM_BATCH_SHEET_PATH",
    str(_DATA / "split_outputs_6351" / "AR6_6351_test_list.xlsx"),
)
GT_PATH = os.environ.get(
    "IAM_LLM_GT_PATH",
    str(_DATA / "split_outputs_6351" / "AR6_6351_test_gt.xlsx"),
)
# embedded neighbour pool: train70 (7/10), same information budget as cVAE / KNN
# (the full 8/10 file is still available via this env var if ever needed)
EXCEL_PATH = os.environ.get(
    "IAM_LLM_TRAIN_GT_PATH",
    str(_DATA / "split_outputs_6351" / "AR6_6351_train70_ground_truth.xlsx"),
)
EXCEL_SHEET = int(os.environ.get("IAM_LLM_EXCEL_SHEET", "0"))

MODE = "batch"  # "single" or "batch"
# Paths
OUTPUTS_DIR = str(_HERE / "outputs")
Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
# Single-mode outputs (distinct names: OUT_XLSX is reused later for the eval report)
OUT_SINGLE_JSON = os.path.join(OUTPUTS_DIR, "single_target_output.json")
OUT_SINGLE_XLSX = os.path.join(OUTPUTS_DIR, "single_target_output.xlsx")
OUT_JSON = OUT_SINGLE_JSON  # backward-compat alias
OUT_BATCH_CSV = os.path.join(OUTPUTS_DIR, "test6351_pred.csv")

# Batch config
MAX_CONCURRENCY = 20
RETRY_MAX = 10
RETRY_BACKOFF_BASE = 1.8

"""function define"""

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _norm_col(c: str) -> str:
    c = str(c).strip().lower()
    c = re.sub(r"\s+", "_", c)
    c = c.replace("-", "_")
    return c

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_norm_col(c) for c in df.columns]
    return df

def find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_str(x):
    return "" if pd.isna(x) else str(x)

def embed_texts(texts, cache: dict, prefix: str, batch_size: int = 128):
    """
    Embed texts with caching (JSON).
    Cache key = f"{prefix}:{text}"
    """
    _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    keys = [f"{prefix}:{t}" for t in texts]
    missing_texts = []
    missing_keys = []
    for t, k in zip(texts, keys):
        if k not in cache:
            missing_texts.append(t)
            missing_keys.append(k)

    if missing_texts:
        for i in tqdm(range(0, len(missing_texts), batch_size), desc=f"Embedding {prefix}"):
            batch = missing_texts[i:i+batch_size]
            resp = _client.embeddings.create(model=EMBED_MODEL, input=batch)
            for emb_obj, k in zip(resp.data, missing_keys[i:i+batch_size]):
                cache[k] = emb_obj.embedding

    embs = np.array([cache[k] for k in keys], dtype=np.float32)
    return embs

def cosine_rank(query_emb: np.ndarray, cand_embs: np.ndarray):
    sims = cosine_similarity(query_emb.reshape(1, -1), cand_embs).ravel()
    order = np.argsort(-sims)
    return order, sims

def build_fp_text(row: pd.Series) -> str:
    parts = [
        f"Mitigation Preference: {safe_str(row.get('mitigation_preference',''))}",
        f"Responds: {safe_str(row.get('responds',''))}",
    ]
    return "\n".join(parts).strip()

"""Load data"""

# Read scenario data
df = pd.read_excel(EXCEL_PATH, sheet_name=EXCEL_SHEET)
df = normalize_columns(df)

# Read model fingerprint data
fp = pd.read_excel(FINGERPRINT_PATH, sheet_name=FP_SHEET_NAME)
fp = normalize_columns(fp)

col_model      = find_col(df, ["model"])
col_scenario   = find_col(df, ["scenario"])
col_region     = find_col(df, ["region"])
col_variable   = find_col(df, ["variable"])
col_unit       = find_col(df, ["unit"])
col_family     = find_col(df, ["model_family"])
col_desc       = find_col(df, ["scenario_description"])
col_c_category = find_col(df, ["category_c"])

# Eval targets table
df_gt = pd.read_excel(GT_PATH, sheet_name=EXCEL_SHEET)
scenario2desc = dict(zip(df[col_scenario], df[col_desc]))
family2model = dict(zip(df[col_family], df[col_model]))

# Supplement scenario descriptions from the FULL (unsplit) dataset, so test
# scenarios that never appear in train still get a description.
# (Descriptions are model INPUT, not ground truth -> not leakage.)
DESC_SOURCE_PATH = os.environ.get(
    "IAM_LLM_DESC_SOURCE_PATH",
    str(_DATA / "AR6_cleaned_6351.xlsx"),
)
_desc_supplement = {}
if os.path.exists(DESC_SOURCE_PATH):
    _dxl = pd.ExcelFile(DESC_SOURCE_PATH)
    _dsheet = 0
    for _sn in _dxl.sheet_names:   # file may contain a stray empty sheet
        _probe = pd.read_excel(_dxl, sheet_name=_sn, nrows=1)
        if "scenario" in [str(c).strip() for c in _probe.columns]:
            _dsheet = _sn
            break
    _ddf = normalize_columns(pd.read_excel(_dxl, sheet_name=_dsheet))
    _ds = find_col(_ddf, ["scenario"])
    _dd = find_col(_ddf, ["scenario_description"])
    if _ds and _dd:
        for _s, _d in zip(_ddf[_ds], _ddf[_dd]):
            _s, _d = str(_s).strip(), safe_str(_d).strip()
            if _s and _d:
                _desc_supplement.setdefault(_s, _d)
    del _ddf
for _s, _d in _desc_supplement.items():
    scenario2desc.setdefault(_s, _d)

df_gt[col_desc] = df_gt[col_scenario].map(scenario2desc)
df_gt[col_model] = df_gt[col_scenario].map(family2model)

# Neighbor retrieval pool: TRAIN ONLY by default.
# (The old Colab pipeline concatenated eval GT into the pool; with the new
#  6351 split we keep the pool clean - if a test case has no same-scenario
#  train neighbor, that neighbor category is simply absent.)
INCLUDE_EVAL_GT_IN_NEIGHBOR_POOL = False
if INCLUDE_EVAL_GT_IN_NEIGHBOR_POOL:
    df = pd.concat(
       [df, df_gt.reindex(columns=df.columns)],
       axis=0
    )

req = {
    "model": col_model,
    "scenario": col_scenario,
    "region": col_region,
    "variable": col_variable,
    "unit": col_unit,
    "model_family": col_family,
    "scenario_description": col_desc,
    "category_c": col_c_category,
}
missing = [k for k, v in req.items() if v is None]
if missing:
    raise KeyError(
        f"Missing columns in Excel sheet '{EXCEL_SHEET}': {missing}\n"
        f"Available columns: {list(df.columns)}"
    )

# Convert year columns if read as int
for y in YEARS:
    yi = int(y)
    if y not in df.columns and yi in df.columns:
        df = df.rename(columns={yi: y})

fp_col_model = find_col(fp, ["model_family"])
fp_col_pref  = find_col(fp, ["mitigation_preference"])
fp_col_resp  = find_col(fp, ["responds"])

if fp_col_model is None:
    raise KeyError(
        f"Fingerprint sheet missing 'model_family' column. Available: {list(fp.columns)}"
    )

# Build fp_text (must exist before you embed fingerprints)
fp["fp_text"] = fp.apply(build_fp_text, axis=1)
FP_FAMILY_COL = "model_family"
if FP_FAMILY_COL not in fp.columns:
    raise KeyError(f"fp missing '{FP_FAMILY_COL}'. Available: {list(fp.columns)}")
if "fp_text" not in fp.columns:
    raise KeyError(
        "fp missing 'fp_text'. Did you run fp['fp_text'] = fp.apply(build_fp_text, axis=1)?"
    )

_fp2 = fp[[FP_FAMILY_COL, "fp_text"]].copy()
_fp2[FP_FAMILY_COL] = _fp2[FP_FAMILY_COL].fillna("").astype(str).str.strip()
_fp2["fp_text"]     = _fp2["fp_text"].fillna("").astype(str)

model_family2fp_text = dict(zip(_fp2[FP_FAMILY_COL], _fp2["fp_text"]))

# Ensure TARGET family exists
tf = str(TARGET["model_family"]).strip()
if tf not in model_family2fp_text or not str(model_family2fp_text[tf]).strip():
    raise KeyError(
        f"TARGET model_family '{tf}' not found or has empty fp_text in model_family2fp_text."
    )
print("model_family2fp_text size:", len(model_family2fp_text))
print("TARGET fp_text preview:", str(model_family2fp_text[tf])[:200], "...")

df_gt.shape

"""Embeddings (with caches)"""

# Re-enabled JSON caches for local runs (cache key includes the full text,
# so edited fingerprints/descriptions still get fresh embeddings automatically).
scen_cache = load_json(SCEN_EMB_CACHE)
fp_cache   = load_json(FP_EMB_CACHE)

df_desc_global = df[[col_scenario, col_desc]].copy()
df_desc_global[col_scenario] = df_desc_global[col_scenario].astype(str).str.strip()
df_desc_global[col_desc]     = df_desc_global[col_desc].fillna("").astype(str)

df_desc_global = (
    df_desc_global
    .sort_values([col_scenario])
    .groupby(col_scenario, as_index=False)
    .agg({col_desc: lambda x: next((t for t in x if str(t).strip()), "")})
)

scenario2desc = dict(zip(df_desc_global[col_scenario], df_desc_global[col_desc]))
# Re-apply full-dataset description supplement (test-only scenarios etc.)
for _s, _d in _desc_supplement.items():
    if not str(scenario2desc.get(_s, "")).strip():
        scenario2desc[_s] = _d
print("scenario2desc size:", len(scenario2desc))


all_desc = sorted({str(v).strip() for v in scenario2desc.values() if str(v).strip()})
desc_embs = embed_texts(all_desc, scen_cache, prefix="SCEN_DESC")
save_json(scen_cache, SCEN_EMB_CACHE)
desc2emb = {t: desc_embs[i] for i, t in enumerate(all_desc)}


FP_FAMILY_COL = "model_family"
fp_families = fp[FP_FAMILY_COL].fillna("").astype(str).str.strip().tolist()
fp_texts    = fp["fp_text"].fillna("").astype(str).tolist()

fp_embs = embed_texts(fp_texts, fp_cache, prefix="MODEL_FP")
save_json(fp_cache, FP_EMB_CACHE)


family2fp = {fam: emb for fam, emb in zip(fp_families, fp_embs) if fam}
print("family2fp keys:", list(family2fp.keys()))

def infer_family_for_model(model_name: str) -> str | None:
    fams = (
        df[df[col_model].astype(str) == str(model_name)][col_family]
        .dropna().astype(str).unique().tolist()
    )
    return fams[0] if fams else None

"""Retrieval"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


YEARS = [str(y) for y in range(2010, 2101, 10)]
VARS_14_NORM = [str(v).strip() for v in VARS_14]
VARS_14_SET  = set(VARS_14_NORM)

def ensure_year_cols(df: pd.DataFrame, years=YEARS) -> pd.DataFrame:
    df = df.copy()
    for y in years:
        yi = int(y)
        if y not in df.columns and yi in df.columns:
            df = df.rename(columns={yi: y})
    return df

df = ensure_year_cols(df, YEARS)

def get_scenario_desc(scenario: str) -> str | None:
    d = scenario2desc.get(str(scenario))
    if d is None:
        return None
    d = str(d).strip()
    return d if d else None

# Step1: rank scenarios by desc similarity within same region+family
def step1_rank_scenarios_region_family(target_region: str, target_family: str, target_scenario: str):
    tdesc = get_scenario_desc(target_scenario)
    if not tdesc or tdesc not in desc2emb:
        raise ValueError(f"Target scenario description missing/not embedded: {target_scenario}")

    q = desc2emb[tdesc].reshape(1, -1)

    cand_s = (
        df[
            (df[col_region].astype(str) == str(target_region)) &
            (df[col_family].astype(str) == str(target_family))
        ][col_scenario]
        .astype(str).unique().tolist()
    )
    keep = []
    for s in cand_s:
        if s == str(target_scenario):
            continue
        d = get_scenario_desc(s)
        if not d or d not in desc2emb:
            continue
        keep.append((str(s), d))

    if not keep:
        return []

    cand_emb = np.vstack([desc2emb[d] for _, d in keep])
    sims = cosine_similarity(q, cand_emb).ravel()
    order = np.argsort(-sims)
    print("Step 1: Target Family ({}) Target Region ({}) Target Scenario ({})".format(target_family, target_region, target_scenario), [(keep[i][0], float(sims[i])) for i in order[: 3]])
    return [(keep[i][0], float(sims[i])) for i in order]


def extract_single_model_18vars(region: str, family: str, model: str, scenario: str,
                               step_tag: str, similarity_score: float,
                               years=YEARS) -> pd.DataFrame:
    sub = df[
        (df[col_region].astype(str) == str(region)) &
        (df[col_model].astype(str) == str(model)) &
        (df[col_scenario].astype(str) == str(scenario)) &
        (df[col_variable].astype(str).str.strip().isin(VARS_14_SET))
    ].copy()

    sub["_var"] = sub[col_variable].astype(str).str.strip()

    for y in years:
        if y in sub.columns:
            sub[y] = pd.to_numeric(sub[y], errors="coerce")

    if not sub.empty:
        sub = sub.sort_values(["_var"]).drop_duplicates(subset=["_var"], keep="first")

    lookup = {str(r["_var"]): r for _, r in sub.iterrows()}

    rows = []
    for v in VARS_14_NORM:
        r0 = lookup.get(v)
        out = {
            "step": step_tag,
            "similarity": float(similarity_score),
            "region": str(region),
            "model_family": str(family),
            "model": str(model),
            "scenario": str(scenario),
            "category_c": None,
            "variable": str(v),
            "unit": None
        }

        if r0 is not None:
            # category_c (source column may be category_c or category_C)
            cat = r0.get(col_c_category, None)
            if isinstance(cat, pd.Series):
                cat = cat.iloc[0] if not cat.empty else None
            out["category_c"] = None if (cat is None or pd.isna(cat)) else str(cat)

            # unit
            unit = r0.get(col_unit, None)
            if isinstance(unit, pd.Series):
                unit = unit.iloc[0] if not unit.empty else None
            out["unit"] = None if (unit is None or pd.isna(unit)) else unit

            # years
            for y in years:
                if y in sub.columns:
                    val = r0.get(y, None)
                else:
                    yi = int(y)
                    val = r0.get(yi, None) if yi in sub.columns else None
                out[y] = None if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val)
        else:
            for y in years:
                out[y] = None

        rows.append(out)

    return pd.DataFrame(rows)

def extract_family_models_for_scenario_limited(region: str, family: str, scenario: str,
                                               step_tag: str, similarity_score: float,
                                               remaining_budget: int):
    models = (
        df[
            (df[col_region].astype(str) == str(region)) &
            (df[col_family].astype(str) == str(family)) &
            (df[col_scenario].astype(str) == str(scenario))
        ][col_model]
        .astype(str).unique().tolist()
    )

    tables = []
    used = 0
    for m in models:
        if used >= remaining_budget:
            break
        tables.append(
            extract_single_model_18vars(
                region=region,
                family=family,
                model=str(m),
                scenario=str(scenario),
                step_tag=step_tag,
                similarity_score=similarity_score
            )
        )
        used += 1

    out_df = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return out_df, used

# Step2: choose top-k model×scenario using family similarity (family2fp)
def topk_models_same_region_scenario_by_family_similarity(
    target_region: str,
    target_scenario: str,
    target_family: str,
    k_models: int = 3,
    exclude_target_family: bool = True
):
    if target_family not in family2fp:
        raise ValueError(f"target_family not in family2fp: {target_family}")

    sub = df[
        (df[col_region].astype(str) == str(target_region)) &
        (df[col_scenario].astype(str) == str(target_scenario))
    ][[col_model, col_family]].dropna().copy()

    if sub.empty:
        return []

    # model -> family (first)
    m2f = {}
    for m, f in zip(sub[col_model].astype(str).tolist(), sub[col_family].astype(str).tolist()):
        if m not in m2f:
            m2f[str(m)] = str(f)

    q = family2fp[target_family].reshape(1, -1)

    fam2sim = {}
    for fam in sorted(set(m2f.values())):
        if exclude_target_family and fam == str(target_family):
            continue
        if fam not in family2fp:
            continue
        fam2sim[fam] = float(cosine_similarity(q, family2fp[fam].reshape(1, -1))[0, 0])

    if not fam2sim:
        return []

    scored = [(m, fam, fam2sim[fam]) for m, fam in m2f.items() if fam in fam2sim]
    scored.sort(key=lambda x: x[2], reverse=True)
    print("Step 2: Target Family ({}) Target Region ({}) Target Scenario ({})".format(target_family, target_region, target_scenario), scored[: 3])
    return scored[:k_models]  # (model, family, fam_sim)

def step3_cross_with_budget(
    target_region: str,
    step1_used_scenarios: list[tuple[str, float]],     # [(scenario, scen_sim)] in order
    step2_models: list[tuple[str, str, float]],        # [(model, family, fam_sim)]
    topk_cross: int
):
    # unique families from step2, keep best similarity per family
    fam_best = {}
    for _, fam, fam_sim in step2_models:
        fam = str(fam)
        fam_best[fam] = max(fam_best.get(fam, -1e9), float(fam_sim))

    remaining = int(topk_cross)
    tables = []

    if remaining <= 0 or not fam_best or not step1_used_scenarios:
        return pd.DataFrame()

    for scen, scen_sim in step1_used_scenarios:
        if remaining <= 0:
            break
        scen = str(scen)
        scen_sim = float(scen_sim)

        for fam, fam_sim in sorted(fam_best.items(), key=lambda x: x[1], reverse=True):
            if remaining <= 0:
                break

            exists = df[
                (df[col_region].astype(str) == str(target_region)) &
                (df[col_family].astype(str) == str(fam)) &
                (df[col_scenario].astype(str) == str(scen))
            ]
            if exists.empty:
                continue

            combo_sim = 0.5 * float(fam_sim) + 0.5 * float(scen_sim)

            block_df, used = extract_family_models_for_scenario_limited(
                region=target_region,
                family=fam,
                scenario=scen,
                step_tag="STEP3_CROSS",
                similarity_score=combo_sim,
                remaining_budget=remaining
            )

            if not block_df.empty and used > 0:
                tables.append(block_df)
                remaining -= used

    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

def assign_combo_id_sequential(df_in: pd.DataFrame, start_id: int):
    """
    Returns (df_out, next_id)
    """
    if df_in is None or df_in.empty:
        return df_in, start_id

    df_out = df_in.copy()

    combos = df_out[["model", "scenario"]].drop_duplicates(keep="first").reset_index(drop=True)
    combos["combo_id"] = range(start_id, start_id + len(combos))

    df_out = df_out.merge(combos, on=["model", "scenario"], how="left")

    cols = ["combo_id"] + [c for c in df_out.columns if c != "combo_id"]
    df_out = df_out[cols]

    next_id = start_id + len(combos)
    return df_out, next_id

def run_retrieval_with_stepwise_ids(
    target_region: str,
    target_family: str,
    target_scenario: str,
    topk_scen: int,
    topk_model: int,
    run_step3: bool,
    topk_cross: int,
    exclude_target_family_in_step2: bool = True
):
    # ----- Step1 (budgeted by combos) -----
    ranked_scen = step1_rank_scenarios_region_family(target_region, target_family, target_scenario)

    remaining = int(topk_scen)
    step1_tables = []
    step1_used_scenarios = []  # [(scenario, sim)] ONLY those used (in order)

    for scen, sim in ranked_scen:
        if remaining <= 0:
            break

        block_df, used = extract_family_models_for_scenario_limited(
            region=target_region,
            family=target_family,
            scenario=scen,
            step_tag="STEP1_SCEN",
            similarity_score=float(sim),
            remaining_budget=remaining
        )

        if not block_df.empty and used > 0:
            step1_tables.append(block_df)
            step1_used_scenarios.append((str(scen), float(sim)))
            remaining -= used

    step1_df = pd.concat(step1_tables, ignore_index=True) if step1_tables else pd.DataFrame()

    # ----- Step2 (budgeted by combos directly) -----
    step2_models = topk_models_same_region_scenario_by_family_similarity(
        target_region=target_region,
        target_scenario=target_scenario,
        target_family=target_family,
        k_models=int(topk_model),
        exclude_target_family=exclude_target_family_in_step2
    )

    step2_tables = []
    for m, fam, sim in step2_models:
        step2_tables.append(
            extract_single_model_18vars(
                region=target_region,
                family=fam,
                model=m,
                scenario=target_scenario,
                step_tag="STEP2_MODEL",
                similarity_score=float(sim)
            )
        )
    step2_df = pd.concat(step2_tables, ignore_index=True) if step2_tables else pd.DataFrame()

    # ----- Step3 (optional) -----
    if run_step3:
        step3_df = step3_cross_with_budget(
            target_region=target_region,
            step1_used_scenarios=step1_used_scenarios,
            step2_models=step2_models,
            topk_cross=int(topk_cross)
        )
    else:
        step3_df = pd.DataFrame()

    next_id = 1
    step1_df, next_id = assign_combo_id_sequential(step1_df, next_id)
    step2_df, next_id = assign_combo_id_sequential(step2_df, next_id)
    step3_df, next_id = assign_combo_id_sequential(step3_df, next_id)

    return step1_df, step2_df, step3_df, next_id


def compute_steps_for_target(target: dict) -> dict:
    """
    Build STEP1_DF / STEP2_DF / STEP3_DF for a given target combo.
    Requires globals: TOPK_SCEN, TOPK_MODEL, RUN_STEP3, TOPK_CROSS
    """
    step1_df, step2_df, step3_df, _ = run_retrieval_with_stepwise_ids(
        target_region=target["region"],
        target_family=target["model_family"],
        target_scenario=target["scenario"],
        topk_scen=TOPK_SCEN,
        topk_model=TOPK_MODEL,
        run_step3=RUN_STEP3,
        topk_cross=TOPK_CROSS,
        exclude_target_family_in_step2=True
    )
    return {"STEP1_DF": step1_df, "STEP2_DF": step2_df, "STEP3_DF": step3_df}


"""# Executor"""

import os
import json
import re
import ast
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from openai import OpenAI

# ============================================================
# CONFIG (HARD MODE SWITCHES)
# ============================================================
# Default LLM: Gemini 3.1 Flash-Lite. Override per-run without editing this file:
#   IAM_LLM_PROVIDER=openai IAM_LLM_MODEL=gpt-4.1-mini python3 rag_core.py
PROVIDER = os.environ.get("IAM_LLM_PROVIDER", "gemini")   # "openai" | "gemini"

# ---- OpenAI models ----
MODEL_NAME = os.environ.get("IAM_LLM_MODEL", "gpt-4.1-mini")

# ---- Gemini models ----
GEMINI_MODEL_NAME = os.environ.get("IAM_LLM_GEMINI_MODEL", "gemini-3.1-flash-lite")

# ============================================================
# RUN TAG: date + model + topk + fp sheet + scenario-desc on/off.
# Used in output filenames so runs never overwrite each other.
# ============================================================
from datetime import datetime as _dt
# fp tag: sheet name by default; if a custom fingerprint file is supplied via env,
# use its filename so runs with different fingerprint sources are distinguishable
_fp_env = os.environ.get("IAM_LLM_FINGERPRINT_PATH", "")
_FP_TAG = (Path(_fp_env).stem.replace("fingerprint_from_papers_", "papers-")
           if _fp_env else FP_SHEET_NAME)
RUN_TAG = (
    f"{_dt.now():%Y%m%d_%H%M}"
    f"_{(MODEL_NAME if PROVIDER == 'openai' else GEMINI_MODEL_NAME).replace('/', '-')}"
    f"_topk{TOPK_SCEN}{TOPK_MODEL}{TOPK_CROSS if RUN_STEP3 else 0}"
    f"_fp-{_FP_TAG}"
    f"_desc-{'on' if INCLUDE_SCENARIO_DESC else 'off'}"
)
# Dataset tag derived from the test list filename
# (AR6_6351_test / AR6_6351_sub01_test / ...)
DATASET_TAG = Path(BATCH_SHEET_PATH).stem.replace("_list", "")
OUT_BATCH_CSV = os.path.join(OUTPUTS_DIR, f"{DATASET_TAG}_pred_{RUN_TAG}.csv")

TEMPERATURE = 0
N_RUNS = 1

INCLUDE_CGROUP_RATIONALE = True
INCLUDE_PER_VAR_RATIONALE = True

# HARD MODE: choose what to expose to the LLM (raw values OR curve features)
# - "raw"       => pass ONLY decade values (2010..2100)
# - "features"  => pass ONLY curve features (requires FEATURES_DF)
ELECTRICITY_EVIDENCE_MODE = "raw"      # "raw" | "features"

# For NON-electricity variables, choose what to expose.
OTHER_EVIDENCE_MODE = "raw"            # "raw" | "features"

# Years
YEARS = [str(y) for y in range(2010, 2101, 10)]

# ============================================================
# HARD CONSTRAINT VARIABLES (must match your exact variable strings)
# ============================================================
TOTAL_ELECTRICITY_VAR = None   # No single total variable for Primary Energy

TECH_ELECTRICITY_VARS = []     # Not used (no total balance constraint for Primary Energy)

AGG_IDENTITIES = [
    {
        "parent": "Primary Energy|Coal",
        "parts": [
            "Primary Energy|Coal|w/ CCS",
            "Primary Energy|Coal|w/o CCS",
        ],
    },
    {
        "parent": "Primary Energy|Gas",
        "parts": [
            "Primary Energy|Gas|w/ CCS",
            "Primary Energy|Gas|w/o CCS",
        ],
    },
    {
        "parent": "Primary Energy|Oil",
        "parts": [
            "Primary Energy|Oil|w/ CCS",
            "Primary Energy|Oil|w/o CCS",
        ],
    },
]

# All primary energy variables share the same evidence mode
PARENT2PARTS = {it["parent"]: list(it["parts"]) for it in AGG_IDENTITIES}
ELECTRICITY_SET = set(VARS_14)
for parent, parts in PARENT2PARTS.items():
    ELECTRICITY_SET.add(parent)
    for p in parts:
        ELECTRICITY_SET.add(p)

# ============================================================
# REQUIRED INPUTS (must exist from your previous cells)
# - VARS_14
# - TARGET dict with keys: region, model_family, scenario
# - scenario2desc
# - model_family2fp_text
# - STEP1_DF, STEP2_DF, (optional STEP3_DF)
# - (optional) FEATURES_DF if you choose any *MODE="features"
# ============================================================
VARS_OUT = [str(v).strip() for v in VARS_14]
VARS_SET = set(VARS_OUT)

TARGET_SCENARIO = str(TARGET["scenario"])
TARGET_FAMILY   = str(TARGET["model_family"]).strip()
TARGET_REGION   = str(TARGET["region"])

if "CACHE_DIR" not in globals():
    CACHE_DIR = "."

# ============================================================
# Basic getters
# ============================================================
def get_model_family_fp_text(model_family: str) -> str:
    if "model_family2fp_text" not in globals():
        raise KeyError("Missing global dict: model_family2fp_text (keyed by model_family).")
    t = model_family2fp_text.get(str(model_family))
    if t is None:
        raise KeyError(f"No fingerprint found for model_family='{model_family}'.")
    t = str(t).strip()
    if not t:
        raise ValueError(f"Empty fingerprint text for model_family='{model_family}'.")
    return t

def get_scenario_desc_text(scenario: str) -> str | None:
    if "scenario2desc" not in globals():
        return None
    t = scenario2desc.get(str(scenario))
    if t is None:
        return None
    t = str(t).strip()
    return t if t else None

def _nan_to_none(x):
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    return x

# ============================================================
# Consistency checks
# ============================================================
def assert_constraints_consistent(vars_set: set[str]):
    missing = []
    for it in AGG_IDENTITIES:
        if it["parent"] not in vars_set:
            missing.append(it["parent"])
        for p in it["parts"]:
            if p not in vars_set:
                missing.append(p)
    missing = sorted(set(missing))
    if missing:
        raise ValueError(
            "Constraint variables missing from VARS_14 (NO implicit residuals).\n"
            + "\n".join([f" - {m}" for m in missing])
        )

assert_constraints_consistent(VARS_SET)

# ============================================================
# Neighbor DF assembly (raw decade data)
# ============================================================
def ensure_year_cols(df: pd.DataFrame, years=YEARS) -> pd.DataFrame:
    df = df.copy()
    for y in years:
        yi = int(y)
        if y not in df.columns and yi in df.columns:
            df = df.rename(columns={yi: y})
    return df

def concat_neighbor_raw(step1_df: pd.DataFrame, step2_df: pd.DataFrame, step3_df: pd.DataFrame | None):
    frames = []
    for x in [step1_df, step2_df, step3_df]:
        if x is not None and not x.empty:
            frames.append(x.copy())
    if not frames:
        # All neighbor categories empty is ALLOWED: the target may have no
        # same-scenario / same-family train neighbors under the new split.
        cols = ["combo_id", "step", "similarity", "region", "model_family",
                "scenario", "category_c", "variable", "unit"] + list(YEARS)
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    out = ensure_year_cols(out, YEARS)
    return out

# ============================================================
# Feature extraction helper (from FEATURES_DF) – only used if MODE="features"
# ============================================================
def require_features_df():
    if (
        ELECTRICITY_EVIDENCE_MODE == "features"
        or OTHER_EVIDENCE_MODE == "features"
    ):
        if "FEATURES_DF" not in globals() or FEATURES_DF is None or FEATURES_DF.empty:
            raise KeyError("You selected evidence_mode='features' but FEATURES_DF is missing/empty.")

def get_features_for_combo_variable(features_df: pd.DataFrame, combo_id: int, variable: str) -> dict:
    sub = features_df[
        (features_df["combo_id"].astype(int) == int(combo_id)) &
        (features_df["variable"].astype(str).str.strip() == str(variable).strip())
    ]
    if sub.empty:
        return {}
    r = sub.iloc[0].to_dict()
    drop = {
        "combo_id", "step", "similarity", "region", "model_family", "scenario",
        "category_c", "variable", "unit"
    }
    feats = {}
    for k, v in r.items():
        if k in drop:
            continue
        feats[k] = _nan_to_none(v)
    return feats

# ============================================================
# HARD MODE: build neighbors evidence – ONLY expose what mode requires
# ============================================================
def var_mode(variable: str) -> str:
    v = str(variable).strip()
    if v in ELECTRICITY_SET:
        return ELECTRICITY_EVIDENCE_MODE
    return OTHER_EVIDENCE_MODE

def build_neighbors_all_evidence_hard(raw_df: pd.DataFrame, vars_out: list[str]) -> list[dict]:
    """
    NEIGHBOR_ALL:
    - per variable: either "values" OR "features" (hard mode)
    - keep STEP1 / STEP2 / STEP3 separated in ordering because similarity meanings differ
    """
    require_features_df()
    neighbors = []

    if raw_df is None or raw_df.empty:
        return neighbors   # no neighbors available for this target - allowed

    need_cols = {"combo_id", "region", "model_family", "scenario", "similarity", "category_c", "variable", "unit"}
    miss = [c for c in need_cols if c not in raw_df.columns]
    if miss:
        raise KeyError(f"NEIGHBOR_RAW_DF missing columns: {miss}")

    def _step_rank(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 999
        s = str(x).strip().lower()
        if s in {"1", "step1", "step_1", "step1_scen"}:
            return 1
        if s in {"2", "step2", "step_2", "step2_model"}:
            return 2
        if s in {"3", "step3", "step_3", "step3_cross"}:
            return 3
        return 999

    raw_df = raw_df.copy()

    # group by step + combo_id if step exists, otherwise keep old behavior
    if "step" in raw_df.columns:
        group_iter = raw_df.groupby([raw_df["step"], raw_df["combo_id"].astype(int)], dropna=False)
    else:
        group_iter = raw_df.groupby(raw_df["combo_id"].astype(int), dropna=False)

    for group_key, g in group_iter:
        meta = g.iloc[0]

        if isinstance(group_key, tuple):
            step_val, combo_id = group_key
        else:
            step_val, combo_id = None, group_key

        scenario = str(meta["scenario"])
        fam = str(meta["model_family"]).strip()

        case = {
            "combo_id": int(combo_id),
            "similarity": _nan_to_none(float(meta["similarity"])) if pd.notna(meta["similarity"]) else None,
            "category_c": (lambda v: None if (v is None or (not isinstance(v, pd.Series) and pd.isna(v)))
                           else (str(v.iloc[0]) if isinstance(v, pd.Series) and not v.empty else str(v)))
                          (meta.get("category_c", None)),
            "region": str(meta["region"]),
            "model_family": fam,
            "scenario": scenario,
            "model_fingerprint_text": get_model_family_fp_text(fam),
            "scenario_description_text": get_scenario_desc_text(scenario),
            "series": [],
        }

        g2 = g.copy()
        g2["_var"] = g2["variable"].astype(str).str.strip()
        lookup = {str(r["_var"]): r for _, r in g2.iterrows()}

        for v in vars_out:
            v = str(v).strip()

            r0 = lookup.get(v)

            unit = None
            values = None
            feats = None

            mode = var_mode(v)

            if r0 is not None:
                unit = _nan_to_none(r0.get("unit", None))
                if mode == "raw":
                    vv = {}
                    for y in YEARS:
                        val = r0.get(y, None)
                        # raw mode: missing -> 0
                        vv[y] = 0.0 if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val)
                    values = vv
                elif mode == "features":
                    feats = get_features_for_combo_variable(FEATURES_DF, int(combo_id), v)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
            else:
                if mode == "raw":
                    # raw mode: variable missing entirely -> all 0
                    values = {y: 0.0 for y in YEARS}
                elif mode == "features":
                    feats = {}
                else:
                    raise ValueError(f"Unknown mode: {mode}")

            item = {"variable": v, "unit": unit}
            if mode == "raw":
                item["values"] = values
            else:
                item["features"] = feats

            case["series"].append(item)

        # internal sort helpers only; remove before return
        case["_step_rank"] = _step_rank(step_val)
        neighbors.append(case)

    # separate ordering by step first, then by similarity within each step
    neighbors.sort(
        key=lambda d: (
            d["_step_rank"],
            -(d["similarity"] if d["similarity"] is not None else -1e9)
        )
    )

    for d in neighbors:
        d.pop("_step_rank", None)

    return neighbors

# ============================================================
# TARGET block
# ============================================================
def build_target_block(target_family: str, target_scenario: str) -> dict:
    return {
        "model_family": str(target_family),
        "scenario": str(target_scenario),
        "model_fingerprint_text": get_model_family_fp_text(str(target_family)),
        "scenario_description_text": (
            get_scenario_desc_text(str(target_scenario))
            if INCLUDE_SCENARIO_DESC else None
        ),
    }

# ============================================================
# PROMPT BUILDER
# ============================================================
def build_json_schema(
    include_cgroup_rationale: bool,
    include_per_var_rationale: bool,
) -> Dict[str, Any]:
    schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "trajectories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "variable": {
                            "type": "string",
                            "description": "Exact variable name",
                        },
                        "unit": {
                            "type": "string",
                            "nullable": True,
                            "description": "Unit or null",
                        },
                        "values": {
                            "type": "object",
                            "properties": {
                                "2010": {"type": "number"},
                                "2020": {"type": "number"},
                                "2030": {"type": "number"},
                                "2040": {"type": "number"},
                                "2050": {"type": "number"},
                                "2060": {"type": "number"},
                                "2070": {"type": "number"},
                                "2080": {"type": "number"},
                                "2090": {"type": "number"},
                                "2100": {"type": "number"},
                            },
                            "required": [
                                "2010", "2020", "2030", "2040", "2050",
                                "2060", "2070", "2080", "2090", "2100",
                            ],
                        },
                    },
                    "required": ["variable", "unit", "values"],
                },
            },
            "c_group": {
                "type": "string",
                "enum": ["C1-C4", "C5-C6", "C7-C8"],
            },
        },
        "required": ["trajectories", "c_group"],
    }

    if include_cgroup_rationale:
        schema["properties"]["c_group_rationale"] = {
            "type": "string",
            "description": (
                "Detailed 3-5 sentence rationale for the c_group choice, grounded in the target "
                "scenario description and same-model-family neighbor comparisons"
            ),
        }
        schema["required"].append("c_group_rationale")

    if include_per_var_rationale:
        schema["properties"]["per_variable_rationales"] = {
            "type": "object",
            "description": (
                "Mapping from exact variable name to a detailed 3-5 sentence evidence-grounded rationale"
            ),
            "additionalProperties": {
                "type": "string"
            },
        }
        schema["required"].append("per_variable_rationales")

    return schema


def build_prompt(
    target_block: dict,
    neighbors_all: list[dict],
    vars_out: list[str],
    total_var: str,
    tech_vars: list[str],
    agg_identities: list[dict],
    include_cgroup_rationale: bool,
    include_per_var_rationale: bool,
) -> str:

    c_notes = (
        "We only output coarse AR6 category groups:\n"
        "- C1-C4 (<= 2°C)\n"
        "- C5-C6 (~2–3°C)\n"
        "- C7-C8 (> 3°C)\n"
    )

    cgroup_few_shots = """
C-GROUP FEW-SHOT EXAMPLES (follow this reasoning style):

Example 1:
TARGET scenario_description_text:
"2CNow_Gradual represents a scenario aimed at limiting global warming to 2 °C through a gradual implementation of mitigation measures."

Relevant same-family neighbors:
[]

Correct output:
{
  "c_group": "C1-C4",
  "c_group_rationale": "The TARGET scenario description directly encodes a 2 °C climate goal, which is strong direct evidence of a stringent mitigation pathway. Under the coarse AR6 grouping used here, scenarios explicitly aiming at 2 °C fall within C1-C4. This classification is supported even without neighbor comparison because the target itself provides a clear mitigation signal. Therefore, the TARGET should be assigned to C1-C4."
}

Example 2:
TARGET scenario_description_text:
"EN_INDCi2030_1000f_NDCp represents an EN scenario with nationally determined contributions until 2030, a full carbon budget of 1000 GtCO₂ that permits the budget to be temporarily overspent, and an NDCp specification."

Relevant same-family neighbors:
[]

Correct output:
{
  "c_group": "C1-C4",
  "c_group_rationale": "The TARGET scenario description does not state an explicit temperature outcome, so temperature-based direct classification is not available. However, it provides carbon-budget evidence: the scenario has a full carbon budget of 1000 GtCO₂, which falls within the range used here as approximate evidence for a likely C1-C4 pathway. The full-budget formulation permits temporary overshoot, so the budget should be treated as guidance rather than an absolute rule. Overall, this evidence supports assigning the TARGET to the stringent coarse group C1-C4."
}

Example 3:
TARGET scenario_description_text:
"LINKS_INDC2030i_1000 represents a CD-LINKS scenario implementing intended nationally determined contributions until 2030 followed by mitigation constrained by a carbon budget of 1000 GtCO₂."

Relevant same-family neighbors:
[
  {
    "scenario": "CD-LINKS_INDC2030i_1600",
    "scenario_description_text": "CD-LINKS_INDC2030i_1600 represents a CD-LINKS scenario implementing intended nationally determined contributions until 2030 followed by mitigation constrained by a carbon budget of 1600 GtCO₂.",
    "category_c": "C4"
  }
]

Correct output:
{
  "c_group": "C1-C4",
  "c_group_rationale": "The TARGET scenario description does not provide an explicit temperature outcome, so temperature-based direct classification is not available. It does provide carbon-budget evidence: a 1000 GtCO₂ budget falls within the range used here as approximate evidence for a likely C1-C4 pathway. The same-model-family neighbor provides additional comparative evidence because the 1600 GtCO₂ neighbor is classified as C4, and the TARGET has a smaller and therefore more stringent carbon budget. Combining the carbon-budget guidance with the same-family comparison, the TARGET should be classified as C1-C4."
}
""".strip()

    if not include_cgroup_rationale:
        # Batch mode (rationales off): strip the rationale field from the few-shot
        # examples, otherwise the model tends to emit it anyway (wasted output tokens)
        # even though the OUTPUT JSON FORMAT omits it.
        cgroup_few_shots = re.sub(
            r',\s*\n\s*"c_group_rationale": "[^"]*"', "", cgroup_few_shots
        )

    ident_lines = []
    for item in agg_identities:
        parent = item["parent"]
        parts = item["parts"]
        ident_lines.append(
            f' "{parent}"[y] = '
            + " + ".join([f'"{p}"[y]' for p in parts])
            + " for every year y."
        )

    identities_block = "\n".join(ident_lines)

    cgroup_rationale_rules = (
        "Provide a concise, evidence-grounded rationale for the c_group classification in 3–5 sentences. "
        "The rationale should follow this logic: "
        "(1) First, check whether the TARGET scenario_description_text contains an explicit temperature outcome "
        "(e.g., 1.5°C, 2°C, 2.5°C, 3°C, or higher). "
        "If present, use the temperature outcome as approximate evidence: approximately 1.5–2°C maps to C1–C4, "
        "approximately 2–3°C maps to C5–C6, and above 3°C maps to C7–C8. "
        "(2) If a carbon budget is provided, use it as approximate evidence for the possible c_group: "
        "budgets around 500–1350 GtCO2 suggest warming of approximately 1.5–2.0°C and therefore likely indicate C1–C4; "
        "budgets around 1500–2050 GtCO2 suggest warming of approximately 2.1–2.4°C and may indicate C5–C6, depending on the context; "
        "Carbon-budget evidence should guide the likely c_group but should not be treated as an absolute mapping rule. "
        "(3) If SAME-MODEL-FAMILY neighbors with known category_c or c_group values are available, use them as comparative evidence "
        "to determine whether the TARGET is more stringent, similar, or less stringent. "
        "Explain how the available temperature evidence, carbon-budget evidence, and/or neighbor comparisons support the assigned c_group. "
        "The rationale must be specific and grounded in the provided inputs. "
        "Do not produce vague statements or unsupported assumptions."
    )

    per_var_rationale_rules = (
        "Provide a concise, evidence-grounded rationale for EACH variable trajectory in 3–5 sentences per variable. "
        "The rationale MUST follow this structure: "
        "(1) Describe the quantitative pattern inferred from the neighbors, "
        "such as the onset of change, peak timing, decline rate, long-run level, or stabilization behavior. "
        "(2) Explain how the TARGET scenario_description_text modifies the inferred trajectory in terms of HOW FAST and HOW FAR "
        "the variable evolves. For example, stronger mitigation may imply a steeper decline and a lower long-run level. "
        "Describe these modifications relative to the relevant same-model, different-scenario neighbor (if available). "
        "(3) Explain how the TARGET model_fingerprint_text modifies the trajectory of each variable relative to the relevant "
        "same-scenario, different-model neighbor if such a neighbor is available; "
        "if no same-scenario neighbor exists, ground the comparison in the same-family neighbors instead. "
        "Address all available components of the model fingerprint: "
        "(a) Mitigation Preference — structural behavior across primary-energy sources and CCS configurations, "
        "such as coal w/ CCS, coal w/o CCS, gas w/ CCS, gas w/o CCS, oil w/ CCS, oil w/o CCS, solar, wind, hydro, nuclear, and biomass; "
        "(b) Responds — response dynamics, such as how carbon-price responsiveness affects the depth and speed of change. "
        "The rationale must be specific and grounded in observable patterns from the provided inputs. "
        "Do not produce vague statements, generic reasoning, or assumptions that are not directly supported by the inputs."
    )


    neighbors_json_all = json.dumps(neighbors_all, ensure_ascii=False, indent=2)
    target_json = json.dumps(target_block, ensure_ascii=False, indent=2)

    vars_json = json.dumps([str(v) for v in vars_out], ensure_ascii=False)
    tech_json = json.dumps([str(v) for v in tech_vars], ensure_ascii=False)

    cgroup_block = ""
    if include_cgroup_rationale:
        cgroup_block = f"""
C-GROUP RATIONALE REQUIREMENT:
- {cgroup_rationale_rules}
""".rstrip()

    per_var_block = ""
    if include_per_var_rationale:
        per_var_block = f"""
PER-VARIABLE RATIONALES REQUIREMENT:
- {per_var_rationale_rules}
""".rstrip()

    extra_fields_schema = ""

    if include_per_var_rationale:
        extra_fields_schema += ',\n  "per_variable_rationales": { "<exact variable name>": "<evidence-grounded rationale>", "...": "..." }'

    extra_fields_schema += ',\n  "c_group": "C1-C4" | "C5-C6" | "C7-C8"'

    if include_cgroup_rationale:
        extra_fields_schema += ',\n  "c_group_rationale": "<3-5 sentence evidence-grounded rationale>"'

    prompt = f"""
You are an expert in integrated assessment model (IAM) scenario synthesis for IPCC AR6-style databases.
Your task is to generate decade-step numeric trajectories and AR6 category group for a TARGET model_family × scenario using the provided TARGET information and NEIGHBOR evidence.

You must reason carefully and step by step internally, but you must output JSON only.
Do NOT include any extra text before or after the JSON.

TASK:
Using NEIGHBOR evidence and TARGET fingerprint/description texts, generate for the TARGET model_family × scenario:
1) the coarse AR6 category group ONLY: one of ["C1-C4", "C5-C6", "C7-C8"]
2) full trajectories (no interpolation) for ALL required variables for years: {", ".join(YEARS)}

AR6 CATEGORY NOTES:
{c_notes}

{cgroup_few_shots}

DECISION PROCESS (FOLLOW INTERNALLY IN THIS ORDER):

VARIABLE TRAJECTORY INFERENCE RULES
Step 1: Infer variable trajectories using quantitative neighbor patterns.
- Use neighbors as quantitative references for trajectory shape, timing, slope, peak timing, decline timing, long-run level, and stabilization behavior.
- SAME-MODEL-FAMILY / DIFFERENT-SCENARIO neighbors are the primary references for the structural behavior of the TARGET model family. Use the neighbors' and the TARGET's scenario_description_text to translate the neighbor patterns into the TARGET trajectory.
- SAME-SCENARIO / DIFFERENT-MODEL-FAMILY neighbors are scenario-level references. Use them to understand how the target scenario responds under different models, and then translate these patterns to the TARGET model family by comparing the neighbors' and the TARGET model's Mitigation Preference and Responds fingerprints.
- Use both the neighbors' and the TARGET's scenario_description_text to determine their relative mitigation stringency and constraints, such as a tighter carbon budget or a more ambitious climate target. These factors influence HOW FAST and HOW FAR the TARGET system should change relative to the neighbors.
- Use the TARGET model_fingerprint_text to determine structural behavior and response dynamics:
  - Mitigation Preference determines the model's structural preferences for achieving mitigation through different variables, primary-energy sources, and CCS configurations, such as coal w/ CCS, coal w/o CCS, gas w/ CCS, gas w/o CCS, oil w/ CCS, oil w/o CCS, solar, wind, hydro, nuclear, and biomass.
  - Responds, or carbon-price responsiveness, determines how variables adjust, including the steepness of changes and the depth of long-run reductions.

Step 2: Generate trajectories subject to all hard consistency constraints.
- Ensure exact algebraic consistency.
- Ensure non-negativity.
- Avoid unrealistic decade-to-decade oscillations.
- If adjustments are needed to satisfy the constraints, adjust the child variables (w/ CCS and w/o CCS) to preserve the aggregation identities.

C-GROUP INFERENCE RULES
Step 1: Use explicit temperature outcomes as approximate classification evidence.
- Read the TARGET scenario_description_text carefully.
- If the TARGET explicitly states a temperature outcome or temperature goal, use it as approximate evidence for the possible c_group.
- Map explicit temperature outcomes to the coarse AR6 groups as follows:
  - Approximately 1.5–2°C suggests C1–C4.
  - Approximately 2–3°C suggests C5–C6.
  - Above 3°C suggests C7–C8.

Step 2: Use carbon-budget information as approximate c_group evidence.
- Carbon budgets are not absolute c_group classification rules, but they can help indicate the most likely c_group.
- Smaller budgets suggest more stringent mitigation and stronger c_groups, whereas larger budgets suggest weaker mitigation and weaker c_groups.
- As rough guidance:
  - Around 500–1350 GtCO₂, corresponding roughly to 1.5–2°C warming, suggests a likely C1–C4 pathway.
  - Around 1500–2050 GtCO₂, corresponding roughly to 2.1–2.4°C warming, may suggest a possible C5–C6 pathway.

Step 3: Use SAME-MODEL-FAMILY neighbors as comparative evidence.
- SAME-MODEL-FAMILY neighbors with known category_c or c_group values should be used to calibrate the TARGET's relative stringency within the same model family.
- Determine whether the TARGET is more stringent, similarly stringent, or less stringent than these neighbors.
- Relevant comparison signals include carbon-budget size, explicit temperature targets, carbon-price levels, mitigation start years, NDC/INDC policy timing, and whether overshoot or full-budget use is allowed.
- If the TARGET is more stringent than a same-family neighbor, it should generally be assigned to the same or a stronger coarse group.
- If the TARGET is less stringent than a same-family neighbor, determine whether it should remain in the same group or move to a weaker group.

Step 4: Combine the evidence and assign the most plausible c_group.
- If the evidence is mixed, select the most plausible coarse group based on the combined evidence.


HARD NUMERIC CONSISTENCY CONSTRAINTS (MUST HOLD EXACTLY FOR EACH YEAR y):

Aggregation identities (parent equals sum of parts):
{identities_block}

NON-NEGATIVITY:
- Every value must be >= 0 for every year.

GENERAL OUTPUT RULES:
- Output MUST be valid JSON.
- Values must be numeric only.
- Do not output NaN.
- Do not output strings for numeric values.
- Avoid null whenever possible.
- Do not omit required variables.
- Use exact variable names.
- Use only the provided evidence.
- If evidence is weak, stay conservative and rely on the closest visible reference pattern rather than inventing unsupported detail.

{cgroup_block}

{per_var_block}

VARIABLE LIST (must include EXACTLY these variables, each exactly once):
{vars_json}

OUTPUT JSON FORMAT (STRICT; FOLLOW EXACTLY):
{{
  "trajectories": [
    {{
      "variable": "<exact variable name>",
      "unit": "<unit or null>",
      "values": {{
        "2010": <number>,
        "2020": <number>,
        "2030": <number>,
        "2040": <number>,
        "2050": <number>,
        "2060": <number>,
        "2070": <number>,
        "2080": <number>,
        "2090": <number>,
        "2100": <number>
      }}
    }}
  ]{extra_fields_schema}
}}

TARGET (JSON):
{target_json}

NEIGHBOR_ALL (JSON):
{neighbors_json_all}

Return JSON only.
""".strip()

    return prompt

# ============================================================
# JSON extraction (robust)
# ============================================================
def extract_json(text: str) -> dict:
    s = (text or "").strip()

    s = re.sub(r"^```(?:json)?\s*", "", s.strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s.strip())

    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j <= i:
        raise ValueError(f"Model output does not contain a JSON object.\nRAW:\n{s[:800]}")
    s = s[i:j+1].strip()

    s = re.sub(r",\s*([}\]])", r"\1", s)

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(s)
            return json.loads(json.dumps(obj, ensure_ascii=False))
        except Exception as e:
            raise ValueError(
                "Failed to parse JSON even after repairs.\n"
                f"Original JSONDecodeError: {repr(e)}\n"
                f"RAW (first 1200 chars):\n{s[:1200]}"
            )

# ============================================================
# Projection to satisfy constraints exactly
# ============================================================
def project_constraints_inplace(traj_map: dict):
    # 0) Non-negativity + fill missing
    for var, item in traj_map.items():
        for y in YEARS:
            v = item["values"].get(y, None)

            if v is None or (isinstance(v, float) and np.isnan(v)):
                item["values"][y] = 0.0
                continue

            vv = float(v)
            item["values"][y] = max(vv, 0.0)

    # 1) Enforce identities: parent = sum(parts)
    for y in YEARS:
        for parent, parts in PARENT2PARTS.items():
            P = float(traj_map[parent]["values"][y])
            s_parts = float(sum(float(traj_map[p]["values"][y]) for p in parts))

            if s_parts > 0:
                traj_map[parent]["values"][y] = s_parts
            else:
                if P > 0:
                    n = len(parts)
                    for p in parts:
                        traj_map[p]["values"][y] = P / n if n > 0 else 0.0
                    traj_map[parent]["values"][y] = float(
                        sum(float(traj_map[p]["values"][y]) for p in parts)
                    )
                else:
                    for p in parts:
                        traj_map[p]["values"][y] = 0.0
                    traj_map[parent]["values"][y] = 0.0

    # 2) Final exact identities again
    for y in YEARS:
        for parent, parts in PARENT2PARTS.items():
            traj_map[parent]["values"][y] = float(
                sum(float(traj_map[p]["values"][y]) for p in parts)
            )

# ============================================================
# Helpers: output <-> map
# ============================================================
def trajectories_to_map(out_obj: dict) -> dict:
    """
    Parse model output to internal map.
    All variables must have numeric values for every year.
    """
    m = {}
    trajs = out_obj.get("trajectories", [])
    if not isinstance(trajs, list):
        raise ValueError("Output 'trajectories' must be a list.")

    for item in trajs:
        var = str(item.get("variable", "")).strip()
        if not var:
            raise ValueError("Trajectory item missing variable.")

        unit = item.get("unit", None)
        values_obj = item.get("values", {}) or {}
        if not isinstance(values_obj, dict):
            raise ValueError(f"Invalid values dict for var={var}")

        vals = {}

        for y in YEARS:
            if y not in values_obj:
                raise KeyError(f"Missing year '{y}' for variable '{var}'")

            v = values_obj.get(y, None)

            if v is None:
                raise ValueError(f"Null value not allowed for variable '{var}' year {y}")

            try:
                vals[y] = float(v)
            except Exception:
                raise ValueError(f"Non-numeric value for variable '{var}' year {y}: {v}")

        m[var] = {"unit": unit, "values": vals}

    for var in VARS_OUT:
        if var not in m:
            raise KeyError(f"Model output missing variable '{var}' entirely.")

    return m

def map_to_output(
    avg_map: dict,
    c_group: str,
    c_group_rationale: str | None,
    per_var_rationales: dict | None,
    prompt: str | None = None,
) -> dict:

    def _json_safe_number(x):
        if x is None:
            return None
        if isinstance(x, float) and np.isnan(x):
            return None
        return float(x)

    out = {
        "c_group": c_group,
        "prompt": prompt,
        "trajectories": [
            {
                "variable": var,
                "unit": avg_map[var]["unit"],
                "values": {y: _json_safe_number(avg_map[var]["values"][y]) for y in YEARS},
            }
            for var in VARS_OUT
        ],
    }

    if INCLUDE_CGROUP_RATIONALE:
        out["c_group_rationale"] = (c_group_rationale or "").strip()
    if INCLUDE_PER_VAR_RATIONALE:
        out["per_variable_rationales"] = per_var_rationales or {}

    return out



"""Run"""

# ============================================================
# 1) Core function: synthesize_one_target (KEEP THIS)
# ============================================================

from decimal import Decimal, ROUND_HALF_UP
import math

def round_sig(x, sig=3):
    if x == 0:
        return 0.0
    return float(
        Decimal(x).quantize(
            Decimal(f'1e{int(math.floor(math.log10(abs(x)))) - sig + 1}'),
            rounding=ROUND_HALF_UP
        )
    )

def round_dict_floats(d, sig=3):
    if isinstance(d, dict):
        return {k: round_dict_floats(v, sig) for k, v in d.items()}
    elif isinstance(d, list):
        return [round_dict_floats(v, sig) for v in d]
    elif isinstance(d, tuple):
        return tuple(round_dict_floats(v, sig) for v in d)
    elif isinstance(d, float):
        return round_sig(d, sig)
    else:
        return d

def synthesize_one_target(target: dict, include_rationales: bool = True) -> dict:
    """
    Synthesize trajectories for one target combo.
    Batch-safe: does NOT rely on global TARGET_* variables.
    """

    global INCLUDE_CGROUP_RATIONALE, INCLUDE_PER_VAR_RATIONALE

    target_region = str(target["region"]).strip()
    target_family = str(target["model_family"]).strip()
    target_scenario = str(target["scenario"]).strip()

    # Ensure TARGET scenario has a description (only if we intend to use it)
    if INCLUDE_SCENARIO_DESC:
        td = get_scenario_desc_text(target_scenario)
        if td is None or not str(td).strip():
            raise KeyError(f"TARGET scenario '{target_scenario}' has no scenario_description in scenario2desc.")

    # Compute retrieval steps for THIS target
    steps = compute_steps_for_target({
        "region": target_region,
        "model_family": target_family,
        "scenario": target_scenario
    })
    step1_df = steps["STEP1_DF"]
    step2_df = steps["STEP2_DF"]
    step3_df = steps["STEP3_DF"]

    # Neighbor raw df
    neighbor_raw_df = concat_neighbor_raw(
        step1_df=step1_df,
        step2_df=step2_df,
        step3_df=step3_df if (step3_df is not None and not step3_df.empty) else None
    )

    # Build hard-mode neighbor evidence
    neighbors_all = build_neighbors_all_evidence_hard(neighbor_raw_df, VARS_OUT)
    target_block = build_target_block(target_family, target_scenario)
    neighbors_all = round_dict_floats(neighbors_all, sig=2)
    print("Neighbors All: ", neighbors_all)

    prompt = build_prompt(
        target_block=target_block,
        neighbors_all=neighbors_all,
        vars_out=VARS_OUT,
        total_var=TOTAL_ELECTRICITY_VAR,
        tech_vars=TECH_ELECTRICITY_VARS,
        agg_identities=AGG_IDENTITIES,
        include_cgroup_rationale=INCLUDE_CGROUP_RATIONALE,
        include_per_var_rationale=INCLUDE_PER_VAR_RATIONALE,
    )
    gemini_json_schema = build_json_schema(
        INCLUDE_CGROUP_RATIONALE,
        INCLUDE_PER_VAR_RATIONALE,
    )

    # ----------------------------
    # LLM call (OpenAI / Gemini)
    # ----------------------------
    from openai import OpenAI
    openai_client = OpenAI()  # uses OPENAI_API_KEY from env

    def _ensure_gemini_client():
        """
        New SDK: google-genai (google.genai)
        Returns a cached client stored in global 'gemini_client'.
        """
        global gemini_client
        if "gemini_client" in globals() and gemini_client is not None:
            return gemini_client

        if not os.environ.get("GEMINI_API_KEY", ""):
            raise ValueError("Missing GEMINI_API_KEY in env.")

        from google import genai
        gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        return gemini_client

    def _ensure_hf_model():
        global hf_model, hf_tokenizer
        if "hf_model" in globals() and hf_model is not None:
            return hf_model, hf_tokenizer

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        # Use HF_MODEL_NAME if available, else MODEL_NAME
        model_name = globals().get("HF_MODEL_NAME", globals().get("MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf"))

        hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        hf_model.eval()
        return hf_model, hf_tokenizer

    def _run_once() -> dict:
        if PROVIDER == "openai":
            resp = openai_client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                seed=globals().get("SEED", 0),
                messages=[
                    {"role": "system", "content": "Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
            )
            return extract_json(resp.choices[0].message.content)

        if PROVIDER == "gemini":
            gc = _ensure_gemini_client()
            gemini_prompt = "Return JSON only.\n\n" + prompt
            from google.genai import types
            # Thinking is ON by default (billed as output tokens - the hidden cost
            # driver). Gemini 3.x cannot fully disable it and uses thinking_level
            # ("minimal" is the floor); Gemini 2.5 uses thinking_budget=0.
            if GEMINI_MODEL_NAME.startswith("gemini-3"):
                thinking_cfg = types.ThinkingConfig(thinking_level="minimal")
            else:
                thinking_cfg = types.ThinkingConfig(thinking_budget=0)
            resp = gc.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=gemini_prompt,
                config=types.GenerateContentConfig(
                    thinking_config=thinking_cfg,
                    response_mime_type="application/json",
                    response_json_schema=gemini_json_schema,
                    temperature=TEMPERATURE,
                    max_output_tokens=40000,
                ),
            )
            return extract_json((resp.text or "").strip())

        if PROVIDER == "huggingface":
            model, tokenizer = _ensure_hf_model()
            messages = [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ]

            try:
                input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
            except Exception:
                # Fallback if chat template is unsupported
                hf_prompt = "System: Return JSON only.\nUser: " + prompt + "\nAssistant:"
                input_ids = tokenizer(hf_prompt, return_tensors="pt").input_ids.to(model.device)

            gen_kwargs = {
                "max_new_tokens": 4000,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if getattr(globals(), "TEMPERATURE", 0) > 0.0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = globals()["TEMPERATURE"]
            else:
                gen_kwargs["do_sample"] = False

            import torch
            print(input_ids.shape)
            with torch.no_grad():
                outputs = model.generate(input_ids, **gen_kwargs)

            input_length = input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            text_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return extract_json(text_output.strip())

        raise ValueError(f"Unknown PROVIDER: {PROVIDER}")

    # reset cached gemini client for safety (optional)
    if "gemini_client" in globals():
        gemini_client = None

    outs = [_run_once() for _ in range(N_RUNS)]

    # c_group majority vote
    c_groups = [o.get("c_group") for o in outs]
    c_group_mode = max(set(c_groups), key=c_groups.count)

    # Average trajectories across runs
    maps = [trajectories_to_map(o) for o in outs]
    avg_map = {}

    for var in VARS_OUT:
        unit = next((m[var]["unit"] for m in maps if m[var].get("unit", None) is not None), None)
        avg_vals = {y: float(np.mean([m[var]["values"][y] for m in maps])) for y in YEARS}
        avg_map[var] = {"unit": unit, "values": avg_vals}

    # Enforce constraints exactly
    project_constraints_inplace(avg_map)

    # Rationales (optional)
    c_group_rationale_text = None
    if INCLUDE_CGROUP_RATIONALE:
        c_group_rationale_text = str(outs[0].get("c_group_rationale", "")).strip()

    per_var_rationales = None
    if INCLUDE_PER_VAR_RATIONALE:
        prv = outs[0].get("per_variable_rationales", {})
        per_var_rationales = prv if isinstance(prv, dict) else {}

    out_obj = map_to_output(
        avg_map=avg_map,
        c_group=c_group_mode,
        c_group_rationale=c_group_rationale_text,
        per_var_rationales=per_var_rationales,
        prompt=prompt,
    )

    return out_obj


# ============================================================
# 2) MODE SWITCH (ONLY SINGLE or ONLY BATCH)
# Put everything below in the LAST notebook cell
# ============================================================

import os
import json
import asyncio
import time
import numpy as np
import pandas as pd



def load_validation_xlsx(path: str) -> pd.DataFrame:
    dfv = pd.read_excel(path)
    required = ["id", "region", "model_family", "scenario"]
    missing = [c for c in required if c not in dfv.columns]
    if missing:
        raise KeyError(f"Missing required columns in validation sheet: {missing}")

    dfv = dfv.copy()
    dfv["id"] = dfv["id"].astype(int)
    for c in ["region", "model_family", "scenario"]:
        dfv[c] = dfv[c].astype(str).str.strip()
    return dfv


def run_single() -> dict:
    """Run exactly one TARGET and save JSON/XLSX (prints rationales)."""
    out_obj = synthesize_one_target(TARGET, include_rationales=True)

    print("c_group:", out_obj.get("c_group"))
    if INCLUDE_CGROUP_RATIONALE:
        print("\nc_group_rationale:", out_obj.get("c_group_rationale", ""))

    if INCLUDE_PER_VAR_RATIONALE:
        print("\nPer-variable rationales (FULL):")
        for v in VARS_OUT:
            txt = out_obj.get("per_variable_rationales", {}).get(v, "")
            print(f"\n- {v}:\n{txt}")

    # Save JSON
    with open(OUT_SINGLE_JSON, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print("\nSaved JSON:", OUT_SINGLE_JSON)

    # Save XLSX (variables × years)
    rows = []
    for item in out_obj["trajectories"]:
        row = {"variable": item["variable"], "unit": item.get("unit", None)}
        for y in YEARS:
            row[y] = item["values"][y]
        row['prompt'] = out_obj.get('prompt', '')
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_excel(OUT_SINGLE_XLSX, index=False)
    print("Saved XLSX:", OUT_SINGLE_XLSX)

    return out_obj


def run_one_target_no_rationale(target: dict) -> dict:
    """Batch mode: one target run with rationales disabled."""
    return synthesize_one_target(target, include_rationales=False)


def run_with_retry_blocking(target: dict) -> dict:
    """Blocking retry wrapper (used inside asyncio.to_thread)."""
    last_err = None
    for k in range(RETRY_MAX):
        try:
            return run_one_target_no_rationale(target)
        except Exception as e:
            last_err = e
            wait = (RETRY_BACKOFF_BASE ** k) + np.random.random()
            wait = 1
            time.sleep(float(wait))
            print("Retry due to error: ", e)
    raise RuntimeError(f"Failed after retries. Last error: {repr(last_err)}")


async def run_batch(dfv: pd.DataFrame) -> pd.DataFrame:
    """Run batch synthesis with bounded concurrency."""
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _one(row: pd.Series) -> pd.DataFrame:
        target = {
            "region": row["region"],
            "model_family": row["model_family"],
            "scenario": row["scenario"],
        }

        async with sem:
            out_obj = await asyncio.to_thread(run_with_retry_blocking, target)

        rows = []
        for item in out_obj["trajectories"]:
            r = {
                "id": int(row["id"]),
                "region": row["region"],
                "model_family": row["model_family"],
                "scenario": row["scenario"],
                "c_group": out_obj.get("c_group"),
                "variable": item["variable"],
                "unit": item.get("unit", None),
                "prompt": out_obj.get("prompt", ""),
            }
            for y in YEARS:
                r[y] = item["values"][y]
            rows.append(r)

        return pd.DataFrame(rows)

    parts = await asyncio.gather(*[_one(dfv.iloc[i]) for i in range(len(dfv))])
    return pd.concat(parts, ignore_index=True)


# ----------------------------
# EXECUTE: mutually exclusive
# ----------------------------
if __name__ == "__main__":
    if MODE == "single":
        _ = run_single()

    elif MODE == "batch":
        INCLUDE_CGROUP_RATIONALE = False
        INCLUDE_PER_VAR_RATIONALE = False

        dfv = load_validation_xlsx(BATCH_SHEET_PATH)
        _lim = int(os.environ.get("TEST_LIMIT", "0"))
        if _lim > 0:
            dfv = dfv.head(_lim)
            print(f"[NOTE] TEST_LIMIT={_lim}: running only the first {_lim} test targets.")
        out_df = asyncio.run(run_batch(dfv))
        out_df.to_csv(OUT_BATCH_CSV, index=False)
        print("Saved batch CSV:", OUT_BATCH_CSV)

    else:
        raise ValueError("MODE must be 'single' or 'batch'.")




"""# validation (phase1)"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# ============================================================
# PATHS
# ============================================================
# Evaluate the CSV from THIS run by default; override with IAM_LLM_PRED_PATH
# to re-evaluate an older prediction file.
PRED_PATH = os.environ.get("IAM_LLM_PRED_PATH", OUT_BATCH_CSV)

OUT_DIR   = OUTPUTS_DIR
OUT_XLSX  = os.path.join(OUT_DIR, f"{DATASET_TAG}_eval_{RUN_TAG}.xlsx")

# ============================================================
# HELPERS
# ============================================================
def norm_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def find_year_cols(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        cs = str(c).strip()
        if cs.isdigit():
            y = int(cs)
            if 1900 <= y <= 2200:
                cols.append(cs)
    cols = sorted(cols, key=lambda s: int(s))
    return cols

def smape(y_true, y_pred, eps=1e-12):
    """
    sMAPE. Points where both true and pred are 0 contribute near-zero (via eps).
    This keeps all points in the mean, pulling down overall sMAPE by the zero-error samples.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(denom, eps)) * 100.0)

def safe_corr(x, y, method="pearson"):
    """
    Correlation on already-filtered 1D arrays.
    Skip invalid cases:
    - n < 3
    - either side is constant (including all-zero)
    Returns (corr, n)
    """
    x2 = np.asarray(x, dtype=float)
    y2 = np.asarray(y, dtype=float)
    n = x2.size

    if n < 3:
        return np.nan, n

    if np.std(x2) == 0 or np.std(y2) == 0:
        return np.nan, n

    if method == "pearson":
        return pearsonr(x2, y2)[0], n
    elif method == "spearman":
        return spearmanr(x2, y2)[0], n
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")

def prepare_arrays_gt_only(y_true, y_pred):
    """
    Final evaluation mode:
    - keep only GT finite points
    - if pred is NaN at kept points, treat as 0
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(yt)
    yt2 = yt[mask]
    yp2 = yp[mask]
    yp2 = np.where(np.isfinite(yp2), yp2, 0.0)

    return yt2, yp2

def compute_metrics_flat(y_true, y_pred):
    """
    Final evaluation mode:
    - keep only GT finite points
    - pred NaN -> 0
    """
    yt, yp = prepare_arrays_gt_only(y_true, y_pred)

    n = yt.size
    if n == 0:
        return {
            "n_points": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "smape_%": np.nan,
            "pearson": np.nan,
            "spearman": np.nan,
        }

    mae = float(np.mean(np.abs(yp - yt)))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    sm = float(smape(yt, yp))
    pr, _ = safe_corr(yt, yp, "pearson")
    sr, _ = safe_corr(yt, yp, "spearman")

    return {
        "n_points": int(n),
        "mae": mae,
        "rmse": rmse,
        "smape_%": sm,
        "pearson": float(pr) if np.isfinite(pr) else np.nan,
        "spearman": float(sr) if np.isfinite(sr) else np.nan,
    }

def compute_metrics_timeseries_row(row, years):
    """
    Compute metrics for ONE (id, variable) trajectory across time.
    Final mode:
    - keep only GT finite points
    - pred NaN -> 0
    """
    yt = np.array([row[f"{y}_true"] for y in years], dtype=float)
    yp = np.array([row[f"{y}_pred"] for y in years], dtype=float)
    m = compute_metrics_flat(yt, yp)
    return pd.Series(m)

def summarize_ts_metrics(df_ts, group_cols):
    """
    Summarize trajectory-level metrics.
    Each row in df_ts is one (id, variable) trajectory.
    """
    rows = []
    for key, sub in df_ts.groupby(group_cols, dropna=False):
        rec = {}
        if isinstance(key, tuple):
            rec.update(dict(zip(group_cols, key)))
        else:
            rec[group_cols[0]] = key

        rec["n_series"] = int(len(sub))
        rec["avg_n_points"] = float(sub["n_points"].mean()) if len(sub) else np.nan
        rec["mae"] = float(sub["mae"].mean()) if len(sub) else np.nan
        rec["rmse"] = float(sub["rmse"].mean()) if len(sub) else np.nan
        rec["smape_%"] = float(sub["smape_%"].mean()) if len(sub) else np.nan
        rec["pearson"] = float(sub["pearson"].mean(skipna=True)) if len(sub) else np.nan
        rec["spearman"] = float(sub["spearman"].mean(skipna=True)) if len(sub) else np.nan
        rec["n_valid_pearson"] = int(sub["pearson"].notna().sum())
        rec["n_valid_spearman"] = int(sub["spearman"].notna().sum())

        rows.append(rec)

    out = pd.DataFrame(rows)
    out = out.sort_values(group_cols).reset_index(drop=True)
    return out

def cgroup_accuracy_by_id(df_merged):
    def mode_or_empty(s):
        s = s.dropna().astype(str).str.strip()
        s = s[s != ""]
        if len(s) == 0:
            return ""
        return s.value_counts().index[0]

    tmp = df_merged.copy()
    tmp["c_group_pred"] = tmp["c_group_pred"].astype(str).str.strip().str.lower()
    tmp["c_group_true"] = tmp["c_group_true"].astype(str).str.strip().str.lower()

    agg = tmp.groupby("id", as_index=False).agg(
        c_group_pred=("c_group_pred", mode_or_empty),
        c_group_true=("c_group_true", mode_or_empty),
        model_family=("model_family_true", mode_or_empty),
        region=("region_true", mode_or_empty),
    )

    valid = (agg["c_group_true"] != "") & (agg["c_group_pred"] != "")
    acc = float((agg.loc[valid, "c_group_pred"] == agg.loc[valid, "c_group_true"]).mean()) if valid.any() else np.nan

    # Version-proof pattern: newer pandas excludes the grouping column from
    # the frames passed to groupby().apply(), so compute correctness first.
    _v = agg.loc[valid].copy()
    _v["_correct"] = (_v["c_group_pred"] == _v["c_group_true"]).astype(float)

    by_family = _v.groupby("model_family")["_correct"].mean().reset_index(name="c_group_acc")
    by_region = _v.groupby("region")["_correct"].mean().reset_index(name="c_group_acc")
    by_cg     = _v.groupby("c_group_true")["_correct"].mean().reset_index(name="c_group_acc")

    # Balanced accuracy = mean per-class recall (handles class imbalance)
    c_group_classes = ["c1-c4", "c5-c6", "c7-c8"]
    per_class_recalls = []
    for cls in c_group_classes:
        n_cls = int((agg.loc[valid, "c_group_true"] == cls).sum())
        if n_cls == 0:
            continue
        tp_cls = int(((agg.loc[valid, "c_group_true"] == cls) & (agg.loc[valid, "c_group_pred"] == cls)).sum())
        per_class_recalls.append(float(tp_cls) / float(n_cls))
    bacc = float(np.mean(per_class_recalls)) if per_class_recalls else np.nan

    summary = pd.DataFrame([{
        "n_ids_total": int(len(agg)),
        "n_ids_valid": int(valid.sum()),
        "c_group_acc": acc,
        "c_group_bacc": bacc,
    }])

    return summary, agg, by_family, by_region, by_cg

def assert_no_dupe_keys(df, side_name, key_cols, show_n=20):
    dup = df.duplicated(subset=key_cols, keep=False)
    if dup.any():
        top = df.loc[dup, key_cols].value_counts().head(show_n)
        raise ValueError(
            f"[ERROR] {side_name}: duplicated keys found for {key_cols}. "
            f"Refusing to aggregate.\nTop duplicated keys:\n{top.to_string()}"
        )

def assert_exact_key_match(pred_df, gt_df, key_cols, show_n=20):
    pred_keys = set(map(tuple, pred_df[key_cols].to_numpy()))
    gt_keys   = set(map(tuple, gt_df[key_cols].to_numpy()))

    only_in_pred = sorted(pred_keys - gt_keys)
    only_in_gt   = sorted(gt_keys - pred_keys)

    if only_in_pred or only_in_gt:
        msg = ["[ERROR] PRED and GT keys do not match exactly."]
        msg.append(f"PRED unique keys: {len(pred_keys)}")
        msg.append(f"GT unique keys: {len(gt_keys)}")
        msg.append(f"Only in PRED: {len(only_in_pred)}")
        msg.append(f"Only in GT: {len(only_in_gt)}")

        if only_in_pred:
            msg.append("\nExamples only in PRED:")
            msg.extend([str(x) for x in only_in_pred[:show_n]])

        if only_in_gt:
            msg.append("\nExamples only in GT:")
            msg.extend([str(x) for x in only_in_gt[:show_n]])

        raise ValueError("\n".join(msg))

# ============================================================
# LOAD DATA
# ============================================================
def run_evaluation():
    pred = pd.read_csv(PRED_PATH)
    gt   = pd.read_excel(GT_PATH)

    pred.columns = [str(c).strip() for c in pred.columns]
    gt.columns   = [str(c).strip() for c in gt.columns]

    required_base = ["id", "region", "model_family", "scenario", "c_group", "variable", "unit"]
    for name, d in [("PRED", pred), ("GT", gt)]:
        miss = [c for c in required_base if c not in d.columns]
        if miss:
            raise KeyError(f"{name} is missing required columns: {miss}")

    for d in [pred, gt]:
        d["id"] = pd.to_numeric(d["id"], errors="coerce").astype("Int64")
        for c in ["region", "model_family", "scenario", "c_group", "variable", "unit"]:
            d[c] = d[c].apply(norm_str)

    pred["c_group"] = pred["c_group"].astype(str).str.strip().str.lower()
    gt["c_group"]   = gt["c_group"].astype(str).str.strip().str.lower()

    pred_years = find_year_cols(pred)
    gt_years   = find_year_cols(gt)
    years = [y for y in pred_years if y in gt_years]
    if len(years) == 0:
        raise ValueError(f"No overlapping year columns found.\nPRED years: {pred_years}\nGT years: {gt_years}")

    pred2 = pred[required_base + years].copy()
    gt2   = gt[required_base + years].copy()

    # Partial runs (TEST_LIMIT): evaluate only the ids that were predicted.
    _pred_ids = set(pred2["id"].astype(int).tolist())
    _gt_ids   = set(gt2["id"].astype(int).tolist())
    if _pred_ids < _gt_ids:
        print(f"[NOTE] Partial run: evaluating {len(_pred_ids)}/{len(_gt_ids)} test ids.")
        gt2 = gt2[gt2["id"].astype(int).isin(_pred_ids)].copy()

    for y in years:
        pred2[y] = pd.to_numeric(pred2[y], errors="coerce")
        gt2[y]   = pd.to_numeric(gt2[y], errors="coerce")

    # ============================================================
    # STRICT DUPLICATE CHECK
    # ============================================================
    key_cols = ["id", "variable"]
    assert_no_dupe_keys(pred2, "PRED", key_cols)
    assert_no_dupe_keys(gt2, "GT", key_cols)

    # ============================================================
    # STRICT KEY MATCH CHECK
    # ============================================================
    assert_exact_key_match(pred2, gt2, key_cols)

    print(f"Matched rows (id, variable): {len(pred2)}")
    print(f"PRED unique keys: {len(pred2)} | GT unique keys: {len(gt2)}")

    # ============================================================
    # MERGE
    # ============================================================
    merged = pred2.merge(
        gt2,
        on=key_cols,
        how="outer",
        suffixes=("_pred", "_true"),
        indicator=True
    )

    if not (merged["_merge"] == "both").all():
        bad = merged.loc[merged["_merge"] != "both", key_cols + ["_merge"]].head(20)
        raise ValueError(
            "[ERROR] Unexpected non-matching rows found after strict key check.\n"
            f"{bad.to_string(index=False)}"
        )

    merged = merged.drop(columns=["_merge"])

    # ============================================================
    # TRUE TIME-SERIES METRICS
    # ============================================================
    ts_meta = [
        "id", "variable",
        "region_true", "model_family_true", "scenario_true", "c_group_true",
        "region_pred", "model_family_pred", "scenario_pred", "c_group_pred"
    ]
    df_ts = merged[ts_meta].copy()

    ts_metrics = merged.apply(lambda row: compute_metrics_timeseries_row(row, years), axis=1)
    df_ts = pd.concat([df_ts, ts_metrics], axis=1)

    # overall time-series summary
    overall_timeseries = pd.DataFrame([{
        "scope": "ALL_timeseries_mean_over_(id,variable)_gt_only",
        "n_series": int(len(df_ts)),
        "avg_n_points": float(df_ts["n_points"].mean()) if len(df_ts) else np.nan,
        "mae": float(df_ts["mae"].mean()) if len(df_ts) else np.nan,
        "rmse": float(df_ts["rmse"].mean()) if len(df_ts) else np.nan,
        "smape_%": float(df_ts["smape_%"].mean()) if len(df_ts) else np.nan,
        "pearson": float(df_ts["pearson"].mean(skipna=True)) if len(df_ts) else np.nan,
        "spearman": float(df_ts["spearman"].mean(skipna=True)) if len(df_ts) else np.nan,
        "n_valid_pearson": int(df_ts["pearson"].notna().sum()),
        "n_valid_spearman": int(df_ts["spearman"].notna().sum()),
    }])

    # grouped summaries
    by_variable_ts    = summarize_ts_metrics(df_ts, ["variable"])
    by_modelfamily_ts = summarize_ts_metrics(df_ts, ["model_family_true"])
    by_region_ts      = summarize_ts_metrics(df_ts, ["region_true"])
    by_cgroup_ts      = summarize_ts_metrics(df_ts, ["c_group_true"])

    by_var_family_ts  = summarize_ts_metrics(df_ts, ["model_family_true", "variable"])
    by_var_region_ts  = summarize_ts_metrics(df_ts, ["region_true", "variable"])

    # ============================================================
    # C-group accuracy
    # ============================================================
    cacc_summary, cacc_by_id, cacc_by_family, cacc_by_region, cacc_by_cgroup = cgroup_accuracy_by_id(merged)

    # ============================================================
    # WRITE EXCEL REPORT
    # ============================================================
    os.makedirs(OUT_DIR, exist_ok=True)

    run_config = pd.DataFrame(
        [
            ("run_datetime",          _dt.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("provider",              PROVIDER),
            ("llm_model",             MODEL_NAME if PROVIDER == "openai" else GEMINI_MODEL_NAME),
            ("temperature",           TEMPERATURE),
            ("topk_scen",             TOPK_SCEN),
            ("topk_model",            TOPK_MODEL),
            ("topk_cross",            TOPK_CROSS),
            ("run_step3",             RUN_STEP3),
            ("fingerprint_path",      FINGERPRINT_PATH),
            ("fingerprint_sheet",     FP_SHEET_NAME),
            ("scenario_desc_used",    INCLUDE_SCENARIO_DESC),
            ("embed_model",           EMBED_MODEL),
            ("test_list",             BATCH_SHEET_PATH),
            ("test_gt",               GT_PATH),
            ("train_gt",              EXCEL_PATH),
            ("pred_csv",              PRED_PATH),
            ("neighbor_pool",         "train_only" if not INCLUDE_EVAL_GT_IN_NEIGHBOR_POOL else "train+eval_gt"),
            ("n_pred_ids",            int(pred2["id"].nunique())),
        ],
        columns=["key", "value"],
    )

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        run_config.to_excel(w, index=False, sheet_name="run_config")
        overall_timeseries.to_excel(w, index=False, sheet_name="overall_timeseries")

        by_variable_ts.to_excel(w, index=False, sheet_name="by_variable_ts")
        by_modelfamily_ts.to_excel(w, index=False, sheet_name="by_model_family_ts")
        by_region_ts.to_excel(w, index=False, sheet_name="by_region_ts")
        by_cgroup_ts.to_excel(w, index=False, sheet_name="by_c_group_true_ts")

        by_var_family_ts.to_excel(w, index=False, sheet_name="by_family_x_variable_ts")
        by_var_region_ts.to_excel(w, index=False, sheet_name="by_region_x_variable_ts")

        df_ts.to_excel(w, index=False, sheet_name="trajectory_metrics")

        cacc_summary.to_excel(w, index=False, sheet_name="cgroup_acc_overall")
        cacc_by_family.to_excel(w, index=False, sheet_name="cgroup_acc_by_family")
        cacc_by_region.to_excel(w, index=False, sheet_name="cgroup_acc_by_region")
        cacc_by_cgroup.to_excel(w, index=False, sheet_name="cgroup_acc_by_cgroup")
        cacc_by_id.to_excel(w, index=False, sheet_name="cgroup_by_id")

        merged_out_cols = [
            "id", "variable",
            "region_true", "model_family_true", "scenario_true", "c_group_true", "unit_true",
            "region_pred", "model_family_pred", "scenario_pred", "c_group_pred", "unit_pred"
        ]
        for y in years:
            merged_out_cols += [f"{y}_true", f"{y}_pred"]
        merged[merged_out_cols].to_excel(w, index=False, sheet_name="merged_wide")

    print(f"\nDone. Report saved to:\n{OUT_XLSX}")

    print("\nOverall timeseries metrics (mean over each (id, variable) trajectory):")
    print(overall_timeseries.to_string(index=False))

    print("\nC-group accuracy (id-level):")
    print(cacc_summary.to_string(index=False))


if __name__ == "__main__" and MODE == "batch":
    run_evaluation()
