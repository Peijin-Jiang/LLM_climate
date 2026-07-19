"""
train_cvae.py
─────────────
Conditional VAE baseline on the NEW AR6_6351 split, adapted from cVAE_iam-main.

Condition modes (each component embedded SEPARATELY, then concatenated,
so model and scenario information live in independent channels):
  1 = [embed(model_family); embed(scenario); embed(region)]              (name-based)
  2 = [embed(model_fingerprint); embed(scenario_description); embed(region)] (content-based)

Data (explicit files, no on-the-fly carving):
  train : data/split_outputs_6351/AR6_6351_train70_ground_truth.xlsx (7/10 of total)
  val   : data/split_outputs_6351/AR6_6351_val10_ground_truth.xlsx   (1/10 of total)
  test  : data/split_outputs_6351/AR6_6351_sub01_test_gt.xlsx (default; env-overridable)

Model selection is by VAL sMAPE everywhere:
  - within a run: keep the epoch checkpoint with the lowest val sMAPE
  - across configs (--sweep): pick the config with the lowest val sMAPE,
    then report test metrics for that config only

Usage (from the 0715/ folder, venv active, torch installed):
    python3 cvae/train_cvae.py --condition 1              # single run, default config
    python3 cvae/train_cvae.py --condition 2 --sweep      # grid search, pick by val sMAPE

Env overrides:
    IAM_LLM_FP_SHEET   fingerprint sheet name (default "fingerprint")
    CVAE_TEST_PATH     test gt xlsx (default sub01)
    CVAE_EPOCHS        default 200
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import itertools
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ============================================================
# PATHS & CONFIG
# ============================================================
_HERE = Path(__file__).resolve().parent          # 0715/cvae/
_ROOT = _HERE.parent                             # 0715/
_DATA = _ROOT / "data"
_SPLIT = _DATA / "split_outputs_6351"
_CACHE = _ROOT / "_cache"
_CACHE.mkdir(exist_ok=True)

TRAIN_PATH = str(_SPLIT / "AR6_6351_train70_ground_truth.xlsx")
VAL_PATH   = str(_SPLIT / "AR6_6351_val10_ground_truth.xlsx")
TEST_PATH  = os.environ.get("CVAE_TEST_PATH", str(_SPLIT / "AR6_6351_sub01_test_gt.xlsx"))
FULL_DATA_PATH = str(_DATA / "AR6_cleaned_6351.xlsx")      # scenario-description source
FINGERPRINT_PATH = str(_DATA / "model_fingerprint.xlsx")
FP_SHEET = os.environ.get("IAM_LLM_FP_SHEET", "fingerprint")

EMBED_MODEL = "text-embedding-3-large"
EMB_CACHE_FILE = _CACHE / "cvae_condition_embeddings.pkl"

SEED = 42
BATCH_SIZE = 32
EPOCHS = int(os.environ.get("CVAE_EPOCHS", "200"))
# Early stopping: stop when val sMAPE has not improved for PATIENCE epochs.
PATIENCE = int(os.environ.get("CVAE_PATIENCE", "20"))

# default single-run config = best config from the previous study's sweep
DEFAULT_CONFIG = {"lr": 2e-3, "hidden": 512, "latent": 32}

# sweep grid = same ranges as the previous study (sweep_results_0607)
SWEEP_GRID = {
    "lr": [5e-4, 1e-3, 2e-3],
    "hidden": [256, 512],
    "latent": [32, 64],
}

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
# DATA LOADING
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
    """scenario -> description from the FULL unsplit dataset (input, not leakage)."""
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
    """Group rows into cases (region, scenario, model_family); canonical variable order."""
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
            # separate channels, embedded independently
            "model_text": str(family),
            "fp_text": fp if fp else str(family),
            "scen_text": str(scenario),
            "desc_text": desc if desc else str(scenario),
            "region_text": str(region),
            "data": g[YEARS].astype(float).values.tolist(),
        })
    return cases


def condition_texts(case, mode):
    """Texts whose embeddings form the condition vector (one channel each)."""
    if mode == 1:   # name-based
        return [case["model_text"], case["scen_text"], case["region_text"]]
    if mode == 2:   # content-based
        return [case["fp_text"], case["desc_text"], case["region_text"]]
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
# MODEL (identical to cVAE_iam-main)
# ============================================================
class cVAEData(Dataset):
    def __init__(self, cases, emb, mode):
        self.cases, self.emb, self.mode = cases, emb, mode

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        item = self.cases[idx]
        vecs = [self.emb[t] for t in condition_texts(item, self.mode)]
        e = torch.tensor(np.concatenate(vecs), dtype=torch.float32)
        x = torch.tensor(item["data"], dtype=torch.float32)
        cat = torch.tensor(item["category"], dtype=torch.float32)
        x = torch.cat([x, torch.ones(1, x.shape[1]) * cat], dim=0)  # extra category row
        return e, x, cat


class Encoder(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + cond_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x, c):
        h = self.relu(self.fc1(torch.cat([x.view(x.size(0), -1), c], dim=1)))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_dim, output_dim, output_shape):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.output_shape = output_shape

    def forward(self, z, c):
        h = self.relu(self.fc1(torch.cat([z, c], dim=1)))
        return self.fc2(h).view(-1, *self.output_shape)


class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim, latent_dim, output_shape):
        super().__init__()
        self.encoder = Encoder(input_dim, cond_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, cond_dim, hidden_dim, input_dim, output_shape)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, c), mu, logvar

    def generate(self, c, latent_dim):
        z = torch.randn(c.size(0), latent_dim).to(c.device)
        return self.decoder(z, c)


def loss_function(recon_x, x, mu, logvar):
    mse = nn.functional.mse_loss(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld


# ============================================================
# METRICS (aligned with rag_core evaluation)
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


def predict(model, loader, device, latent_dim):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for e, x, _ in loader:
            preds.append(model.generate(e.to(device), latent_dim).cpu().numpy())
            trues.append(x.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def val_smape(model, loader, device, latent_dim):
    """Fast objective used for checkpoint + hyperparameter selection."""
    preds, trues = predict(model, loader, device, latent_dim)
    pv, tv = preds[:, :-1, :], trues[:, :-1, :]
    vals = [smape_1d(tv[i, v], np.where(np.isfinite(pv[i, v]), pv[i, v], 0.0))
            for i in range(tv.shape[0]) for v in range(tv.shape[1])]
    return float(np.mean(vals))


def evaluate(model, loader, device, latent_dim):
    preds, trues = predict(model, loader, device, latent_dim)
    pv, tv = preds[:, :-1, :], trues[:, :-1, :]
    pc, tc = preds[:, -1, :], trues[:, -1, :]

    ts = [trajectory_metrics(tv[i, v], pv[i, v])
          for i in range(tv.shape[0]) for v in range(tv.shape[1])]
    df = pd.DataFrame(ts)

    pred_cls, true_cls = np.round(pc[:, 0]), tc[:, 0]
    acc = float(np.mean(pred_cls == true_cls))
    recalls = []
    for cls in [0, 1, 2]:
        n = int(np.sum(true_cls == cls))
        if n:
            recalls.append(int(np.sum((true_cls == cls) & (pred_cls == cls))) / n)
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


# ============================================================
# TRAINING
# ============================================================
def train_one(config, mode, loaders, dims, device, save_dir, epochs):
    """Train one config; checkpoint selection by VAL sMAPE. Returns best val sMAPE."""
    tr_loader, va_loader = loaders
    input_dim, cond_dim, out_shape = dims

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = CVAE(input_dim, cond_dim, config["hidden"], config["latent"], out_shape).to(device)
    opt = optim.Adam(model.parameters(), lr=config["lr"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_smape, best_epoch = float("inf"), -1
    tr_losses, va_smapes = [], []
    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for e, x, _ in tr_loader:
            e, x = e.to(device), x.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(x, e)
            loss = loss_function(recon, x, mu, logvar)
            loss.backward()
            tl += loss.item()
            opt.step()
        tl /= len(tr_loader.dataset)

        vs = val_smape(model, va_loader, device, config["latent"])
        tr_losses.append(tl)
        va_smapes.append(vs)

        if vs < best_smape:
            best_smape, best_epoch = vs, epoch
            torch.save(model.state_dict(), save_dir / "best_cvae_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch}/{epochs} | train loss {tl:.4f} | val sMAPE {vs:.2f}% "
                  f"(best {best_smape:.2f}% @ ep{best_epoch})")

        if epochs_no_improve >= PATIENCE:
            print(f"  early stop at epoch {epoch} "
                  f"(no val sMAPE improvement for {PATIENCE} epochs; best @ ep{best_epoch})")
            break

    json.dump({"train_loss": tr_losses, "val_smape": va_smapes,
               "best_val_smape": best_smape, "best_epoch": best_epoch},
              open(save_dir / "loss_curve.json", "w"), indent=2)
    if HAS_MPL:
        fig, ax1 = plt.subplots()
        ax1.plot(tr_losses, label="train loss", color="tab:blue")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss")
        ax2 = ax1.twinx()
        ax2.plot(va_smapes, label="val sMAPE", color="tab:red")
        ax2.set_ylabel("val sMAPE (%)")
        fig.legend(loc="upper right")
        plt.savefig(save_dir / "loss_curve.png"); plt.close()

    return best_smape, best_epoch, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", type=int, choices=[1, 2], required=True,
                    help="1: [family; scenario; region] embeddings | "
                         "2: [fingerprint; description; region] embeddings")
    ap.add_argument("--sweep", action="store_true",
                    help="grid-search lr/hidden/latent; pick best config by val sMAPE")
    args = ap.parse_args()
    mode = args.condition

    print(f"Condition mode: {mode} | fp sheet: {FP_SHEET} | selection metric: val sMAPE")
    fingerprints = read_model_fingerprints()
    desc_dict = load_desc_dict()

    train_cases = read_cases(TRAIN_PATH, fingerprints, desc_dict)
    val_cases   = read_cases(VAL_PATH, fingerprints, desc_dict)
    test_cases  = read_cases(TEST_PATH, fingerprints, desc_dict)
    print(f"train {len(train_cases)} | val {len(val_cases)} | test {len(test_cases)}")

    texts = [t for c in train_cases + val_cases + test_cases
             for t in condition_texts(c, mode)]
    emb = compute_condition_embeddings(texts)

    tr_loader = DataLoader(cVAEData(train_cases, emb, mode), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(cVAEData(val_cases, emb, mode), batch_size=BATCH_SIZE)
    te_loader = DataLoader(cVAEData(test_cases, emb, mode), batch_size=BATCH_SIZE)

    e0, x0, _ = cVAEData(train_cases, emb, mode)[0]
    cond_dim, out_shape = e0.shape[0], x0.shape
    input_dim = int(np.prod(out_shape))
    dims = (input_dim, cond_dim, out_shape)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    root = _ROOT / "outputs" / "cvae"

    configs = ([dict(zip(SWEEP_GRID, v)) for v in itertools.product(*SWEEP_GRID.values())]
               if args.sweep else [DEFAULT_CONFIG])

    results = []
    for cfg in configs:
        tag = f"cond{mode}_lr{cfg['lr']}_hid{cfg['hidden']}_lat{cfg['latent']}_fp-{FP_SHEET}_{stamp}"
        print(f"\n=== {tag} ===")
        best_smape, best_epoch, _ = train_one(
            cfg, mode, (tr_loader, va_loader), dims, device, root / tag, EPOCHS)
        results.append({**cfg, "tag": tag, "val_smape": best_smape, "best_epoch": best_epoch})

    results_df = pd.DataFrame(results).sort_values("val_smape")
    print("\nConfig ranking by val sMAPE:")
    print(results_df.to_string(index=False))

    # winner: evaluate on test with its best (val-sMAPE) checkpoint
    best = results_df.iloc[0]
    best_dir = root / best["tag"]
    model = CVAE(input_dim, cond_dim, int(best["hidden"]), int(best["latent"]), out_shape).to(device)
    model.load_state_dict(torch.load(best_dir / "best_cvae_model.pth", map_location=device))

    print(f"\nBest config: {best['tag']} (val sMAPE {best['val_smape']:.2f}%)")
    print("Evaluating on test...")
    metrics = evaluate(model, te_loader, device, int(best["latent"]))
    print(json.dumps(metrics, indent=2))

    json.dump(metrics, open(best_dir / "test_metrics.json", "w"), indent=2)
    json.dump({
        "condition_mode": mode, "selection_metric": "val_smape",
        "config": {k: (float(best[k]) if k == "lr" else int(best[k]))
                   for k in ["lr", "hidden", "latent"]},
        "val_smape": float(best["val_smape"]), "best_epoch": int(best["best_epoch"]),
        "batch_size": BATCH_SIZE, "epochs": EPOCHS, "seed": SEED,
        "train_path": TRAIN_PATH, "val_path": VAL_PATH, "test_path": TEST_PATH,
        "fingerprint_sheet": FP_SHEET, "embed_model": EMBED_MODEL,
        "sweep": bool(args.sweep),
        "run_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, open(best_dir / "training_hyperparameters.json", "w"), indent=2)

    if args.sweep:
        results_df.to_excel(root / f"sweep_summary_cond{mode}_{stamp}.xlsx", index=False)
    print("saved to:", best_dir)


if __name__ == "__main__":
    main()
