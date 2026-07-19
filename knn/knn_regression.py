"""
knn_regression.py
─────────────────
KNN regression baseline on the AR6_6351 split, mirroring the cVAE setup.

Two condition modes (same as cVAE; components embedded SEPARATELY so model
and scenario information contribute through independent channels):
  1 = model_family + scenario names        (name-based)
  2 = model_fingerprint + scenario_description (content-based)

Method:
  - embed each component text (text-embedding-3-large, shared cache with cVAE)
  - neighbors MUST be from the same region (hard filter), so similarity uses the
    two informative channels: sim = (cos_model + cos_scenario) / 2
  - for a query case, find k nearest train70 cases within its region
  - trajectory prediction = (uniform | similarity-weighted) mean of neighbors' 14x10 data
  - c_group prediction   = majority vote of neighbors (ties -> higher summed similarity)

Hyperparameter selection by VAL sMAPE:
  grid: k in {1,3,5,10,20,50} x weighting in {uniform, similarity}
  neighbor pool is train70 ONLY (same information budget as the cVAE),
  winner is then evaluated once on the test subset (sub01 by default).

Usage (from the 0715/ folder):
    python3 knn/knn_regression.py --condition 1
    python3 knn/knn_regression.py --condition 2

Env overrides:
    IAM_LLM_FP_SHEET   fingerprint sheet name (default "fingerprint")
    KNN_TEST_PATH      test gt xlsx (default sub01)
"""

import argparse
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# ============================================================
# PATHS & CONFIG
# ============================================================
_HERE = Path(__file__).resolve().parent          # 0715/knn/
_ROOT = _HERE.parent                             # 0715/
_DATA = _ROOT / "data"
_SPLIT = _DATA / "split_outputs_6351"
_CACHE = _ROOT / "_cache"
_CACHE.mkdir(exist_ok=True)

TRAIN_PATH = str(_SPLIT / "AR6_6351_train70_ground_truth.xlsx")
VAL_PATH   = str(_SPLIT / "AR6_6351_val10_ground_truth.xlsx")
TEST_PATH  = os.environ.get("KNN_TEST_PATH", str(_SPLIT / "AR6_6351_sub01_test_gt.xlsx"))
FULL_DATA_PATH = str(_DATA / "AR6_cleaned_6351.xlsx")
FINGERPRINT_PATH = str(_DATA / "model_fingerprint.xlsx")
FP_SHEET = os.environ.get("IAM_LLM_FP_SHEET", "fingerprint")

EMBED_MODEL = "text-embedding-3-large"
EMB_CACHE_FILE = _CACHE / "cvae_condition_embeddings.pkl"   # shared with the cVAE baseline

K_GRID = [1, 2, 3, 4, 5]
WEIGHTINGS = ["uniform", "similarity"]
# channel weight: sim = alpha * cos_model + (1 - alpha) * cos_scenario
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]

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
YEARS = [str(y) for y in range(2010, 2110, 10)]


def load_api_key():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    env_file = _ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY") and "=" in line:
                v = line.partition("=")[2].strip().strip('"').strip("'")
                if v:
                    return v
    raise ValueError("No OpenAI API key found (env var or 0715/.env).")


# ============================================================
# DATA LOADING (same conventions as cvae/train_cvae.py)
# ============================================================
def norm_year_cols(df):
    return df.rename(columns={int(y): y for y in YEARS if int(y) in df.columns})


def map_category(val):
    c = str(val).strip().lower()
    if c in ["c1", "c2", "c3", "c4", "c1-c4"]:
        return 0
    if c in ["c5", "c6", "c5-c6"]:
        return 1
    if c in ["c7", "c8", "c7-c8"]:
        return 2
    return -1


def read_model_fingerprints():
    df = pd.read_excel(FINGERPRINT_PATH, sheet_name=FP_SHEET)
    df.columns = [str(c).strip() for c in df.columns]
    fam_col = next(c for c in df.columns if c.lower().replace(" ", "_") == "model_family")
    mit_col = next(c for c in df.columns if "mitigation" in c.lower())
    rsp_col = next(c for c in df.columns if "respond" in c.lower())
    out = {}
    for _, row in df.iterrows():
        out[str(row[fam_col]).strip()] = (
            f"Mitigation Preference: {row[mit_col]}\nResponds: {row[rsp_col]}"
        )
    return out


def load_desc_dict():
    xl = pd.ExcelFile(FULL_DATA_PATH)
    sheet = next(s for s in xl.sheet_names
                 if "scenario" in [str(c).strip() for c in
                                   pd.read_excel(xl, sheet_name=s, nrows=1).columns])
    df = pd.read_excel(xl, sheet_name=sheet, usecols=["scenario", "scenario_description"])
    out = {}
    for s, d in zip(df["scenario"], df["scenario_description"]):
        s, d = str(s).strip(), "" if pd.isna(d) else str(d).strip()
        if s and d and s not in out:
            out[s] = d
    return out


def read_cases(path, fingerprints, desc_dict):
    df = norm_year_cols(pd.read_excel(path))
    df.columns = [str(c).strip() for c in df.columns]
    var_rank = {v: i for i, v in enumerate(VARS_14)}

    cat_col = "category_c" if "category_c" in df.columns else "c_group"
    cases = []
    for (region, scenario, family), g in df.groupby(["region", "scenario", "model_family"], sort=True):
        g = g.copy()
        g["_rank"] = g["variable"].map(var_rank)
        g = g.dropna(subset=["_rank"]).sort_values("_rank")
        if len(g) != 14:
            raise ValueError(f"Case {region}|{scenario}|{family} has {len(g)} vars (expected 14)")
        desc = desc_dict.get(str(scenario).strip(), "")
        fp = fingerprints.get(str(family).strip(), "")
        cases.append({
            "region": str(region),
            "scenario": str(scenario),
            "model_family": str(family),
            "category": map_category(g[cat_col].iloc[0]),
            # separate channels (region is a hard filter, not a similarity channel)
            "model_text": str(family),
            "fp_text": fp if fp else str(family),
            "scen_text": str(scenario),
            "desc_text": desc if desc else str(scenario),
            "data": np.asarray(g[YEARS].astype(float).values),
        })
    return cases


def channel_texts(case, mode):
    """(model channel, scenario channel) texts for the given mode."""
    if mode == 1:
        return case["model_text"], case["scen_text"]
    if mode == 2:
        return case["fp_text"], case["desc_text"]
    raise ValueError(f"Invalid condition mode: {mode}")


def compute_condition_embeddings(texts):
    cache = {}
    if EMB_CACHE_FILE.exists():
        with open(EMB_CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
    missing = sorted(set(t for t in texts if t not in cache))
    if missing:
        from openai import OpenAI
        client = OpenAI(api_key=load_api_key())
        bs = 500
        for i in tqdm(range(0, len(missing), bs), desc="Embedding conditions"):
            batch = missing[i:i + bs]
            resp = client.embeddings.create(input=batch, model=EMBED_MODEL)
            for j, d in enumerate(resp.data):
                cache[batch[j]] = d.embedding
        with open(EMB_CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
    return cache


# ============================================================
# METRICS (same as cVAE / rag_core evaluation)
# ============================================================
def smape_1d(yt, yp, eps=1e-12):
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    return np.mean(np.abs(yp - yt) / np.maximum(denom, eps)) * 100.0


def safe_corr_1d(x, y, method):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return (pearsonr if method == "pearson" else spearmanr)(x, y)[0]


def trajectory_metrics(yt, yp):
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    mask = np.isfinite(yt)
    yt, yp = yt[mask], np.where(np.isfinite(yp[mask]), yp[mask], 0.0)
    if yt.size == 0:
        return dict(n_points=0, mae=np.nan, rmse=np.nan, **{"smape_%": np.nan},
                    pearson=np.nan, spearman=np.nan)
    return {
        "n_points": int(yt.size),
        "mae": float(np.mean(np.abs(yp - yt))),
        "rmse": float(np.sqrt(np.mean((yp - yt) ** 2))),
        "smape_%": float(smape_1d(yt, yp)),
        "pearson": safe_corr_1d(yt, yp, "pearson"),
        "spearman": safe_corr_1d(yt, yp, "spearman"),
    }


# ============================================================
# KNN CORE
# ============================================================
def build_matrices(cases, emb, mode):
    """Return (M_model, M_scenario): per-channel L2-normalized embedding matrices,
    so that M_model[q] @ M_model[t] = cos_model similarity (same for scenario)."""
    def norm(v):
        v = np.asarray(v, dtype=np.float32)
        return v / max(float(np.linalg.norm(v)), 1e-12)

    Mm, Ms = [], []
    for c in cases:
        mt, st = channel_texts(c, mode)
        Mm.append(norm(emb[mt]))
        Ms.append(norm(emb[st]))
    return np.asarray(Mm, dtype=np.float32), np.asarray(Ms, dtype=np.float32)


def knn_predict(query_Ms, query_cases, train_Ms, train_cases, k, weighting, alpha):
    """Predict trajectories [Q, 14, 10] and categories [Q] for all query cases.

    similarity = alpha * cos_model + (1 - alpha) * cos_scenario
    HARD CONSTRAINT: neighbors must come from the SAME region as the query.
    If a region has fewer than k train cases, all of them are used.
    """
    qMm, qMs = query_Ms
    tMm, tMs = train_Ms
    n_query = qMm.shape[0]

    train_data = np.stack([c["data"] for c in train_cases])   # [N, 14, 10]
    train_cat = np.asarray([c["category"] for c in train_cases])

    region2idx = {}
    for i, c in enumerate(train_cases):
        region2idx.setdefault(c["region"], []).append(i)
    region2idx = {r: np.asarray(v) for r, v in region2idx.items()}

    preds = np.empty((n_query, 14, len(YEARS)))
    cats = np.empty(n_query, dtype=int)
    for q in range(n_query):
        cand = region2idx.get(query_cases[q]["region"])
        if cand is None or len(cand) == 0:
            raise ValueError(f"No train cases in region: {query_cases[q]['region']}")
        sims_q = (alpha * (tMm[cand] @ qMm[q])
                  + (1.0 - alpha) * (tMs[cand] @ qMs[q]))
        kk = min(k, len(cand))
        top = np.argpartition(-sims_q, kk - 1)[:kk]
        idx = cand[top]
        s = sims_q[top]
        if weighting == "similarity":
            w = np.maximum(s, 0) + 1e-9
            w = w / w.sum()
        else:
            w = np.full(len(idx), 1.0 / len(idx))
        preds[q] = np.tensordot(w, train_data[idx], axes=1)

        # majority vote; ties broken by summed similarity
        votes = {}
        for j, i in enumerate(idx):
            c = int(train_cat[i])
            n_votes, s_sum = votes.get(c, (0, 0.0))
            votes[c] = (n_votes + 1, s_sum + float(s[j]))
        cats[q] = max(votes.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]
    return preds, cats


def eval_full(preds, cats, cases):
    trues = np.stack([c["data"] for c in cases])
    true_cat = np.asarray([c["category"] for c in cases])

    ts = [trajectory_metrics(trues[i, v], preds[i, v])
          for i in range(trues.shape[0]) for v in range(trues.shape[1])]
    df = pd.DataFrame(ts)

    acc = float(np.mean(cats == true_cat))
    recalls = []
    for cls in [0, 1, 2]:
        n = int(np.sum(true_cat == cls))
        if n:
            recalls.append(int(np.sum((true_cat == cls) & (cats == cls))) / n)
    bacc = float(np.mean(recalls)) if recalls else float("nan")

    return {
        "n_series": int(len(df)),
        "mae": float(df["mae"].mean()),
        "rmse": float(df["rmse"].mean()),
        "smape_%": float(df["smape_%"].mean()),
        "pearson": float(df["pearson"].mean(skipna=True)),
        "spearman": float(df["spearman"].mean(skipna=True)),
        "n_valid_pearson": int(df["pearson"].notna().sum()),
        "n_valid_spearman": int(df["spearman"].notna().sum()),
        "c_group_accuracy": acc,
        "c_group_bacc": bacc,
    }


def val_smape_only(preds, cases):
    trues = np.stack([c["data"] for c in cases])
    vals = [smape_1d(trues[i, v], np.where(np.isfinite(preds[i, v]), preds[i, v], 0.0))
            for i in range(trues.shape[0]) for v in range(trues.shape[1])]
    return float(np.mean(vals))


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", type=int, choices=[1, 2], required=True,
                    help="1: names embedding | 2: fingerprint+description embedding")
    args = ap.parse_args()
    mode = args.condition

    print(f"KNN baseline | condition mode: {mode} | fp sheet: {FP_SHEET} | pool: train70")
    fingerprints = read_model_fingerprints()
    desc_dict = load_desc_dict()

    train_cases = read_cases(TRAIN_PATH, fingerprints, desc_dict)
    val_cases   = read_cases(VAL_PATH, fingerprints, desc_dict)
    test_cases  = read_cases(TEST_PATH, fingerprints, desc_dict)
    print(f"train {len(train_cases)} | val {len(val_cases)} | test {len(test_cases)}")

    texts = [t for c in train_cases + val_cases + test_cases
             for t in channel_texts(c, mode)]
    emb = compute_condition_embeddings(texts)

    train_Ms = build_matrices(train_cases, emb, mode)
    val_Ms   = build_matrices(val_cases, emb, mode)
    test_Ms  = build_matrices(test_cases, emb, mode)

    # ---- hyperparameter selection on val (by sMAPE) ----
    results = []
    for k in K_GRID:
        for weighting in WEIGHTINGS:
            for alpha in ALPHA_GRID:
                preds, _ = knn_predict(val_Ms, val_cases, train_Ms, train_cases,
                                       k, weighting, alpha)
                vs = val_smape_only(preds, val_cases)
                results.append({"k": k, "weighting": weighting,
                                "alpha_model": alpha, "val_smape": vs})
                print(f"  k={k:<3} {weighting:<10} alpha={alpha:<5} val sMAPE {vs:.2f}%")

    results_df = pd.DataFrame(results).sort_values("val_smape")
    best = results_df.iloc[0]
    print(f"\nBest: k={int(best['k'])}, weighting={best['weighting']}, "
          f"alpha_model={best['alpha_model']} (val sMAPE {best['val_smape']:.2f}%)")
    print("alpha_model is the weight of the MODEL channel; "
          "1-alpha is the weight of the SCENARIO channel.")

    # ---- final test evaluation with the winning config ----
    preds, cats = knn_predict(test_Ms, test_cases, train_Ms, train_cases,
                              int(best["k"]), best["weighting"], float(best["alpha_model"]))
    metrics = eval_full(preds, cats, test_cases)
    print("\nTest metrics:")
    print(json.dumps(metrics, indent=2))

    # ---- save ----
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    tag = (f"knn_cond{mode}_k{int(best['k'])}_{best['weighting']}"
           f"_a{best['alpha_model']}_fp-{FP_SHEET}_{stamp}")
    out_dir = _ROOT / "outputs" / "knn" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_excel(out_dir / "val_sweep.xlsx", index=False)
    json.dump(metrics, open(out_dir / "test_metrics.json", "w"), indent=2)
    json.dump({
        "method": "knn_regression", "condition_mode": mode,
        "selection_metric": "val_smape", "neighbor_pool": "train70",
        "neighbor_constraint": "same_region_only",
        "similarity": "(cos_model + cos_scenario) / 2, separate embeddings per channel",
        "k": int(best["k"]), "weighting": best["weighting"],
        "alpha_model": float(best["alpha_model"]),
        "val_smape": float(best["val_smape"]),
        "k_grid": K_GRID, "weightings": WEIGHTINGS, "alpha_grid": ALPHA_GRID,
        "train_path": TRAIN_PATH, "val_path": VAL_PATH, "test_path": TEST_PATH,
        "fingerprint_sheet": FP_SHEET, "embed_model": EMBED_MODEL,
        "run_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, open(out_dir / "run_config.json", "w"), indent=2)

    # per-case predictions for inspection
    rows = []
    for i, c in enumerate(test_cases):
        for v, var in enumerate(VARS_14):
            row = {"region": c["region"], "scenario": c["scenario"],
                   "model_family": c["model_family"], "variable": var,
                   "c_group_pred": int(cats[i]), "c_group_true": c["category"]}
            row.update({y: preds[i, v, j] for j, y in enumerate(YEARS)})
            rows.append(row)
    pd.DataFrame(rows).to_excel(out_dir / "test_predictions.xlsx", index=False)

    print("saved to:", out_dir)


if __name__ == "__main__":
    main()
