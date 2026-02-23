# Single-cell end-to-end pipeline for:
# "Global vs Local Deep Learning for Multi-City PM2.5 Forecasting"
# Folder contract:
# ./city_data/*.csv
# ./output/{processed_dataset.pt,models/,results/,paper_results.md}

import os
import re
import glob
import math
import json
import random
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---- Lightning compatibility (lightning or pytorch_lightning) ----
try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
    from lightning.pytorch.loggers import WandbLogger, CSVLogger
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
    from pytorch_lightning.loggers import WandbLogger, CSVLogger

# -----------------------------
# Reproducibility + device
# -----------------------------
SEED = 42
pl.seed_everything(SEED, workers=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# -----------------------------
# Paths / constants
# -----------------------------
DATA_DIR = Path("./city_data")
OUT_DIR = Path("./output")
MODELS_DIR = OUT_DIR / "models"
RESULTS_DIR = OUT_DIR / "results"

for d in [OUT_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

REQUIRED_COLS = {
    "city", "station", "date.utc", "parameter", "value",
    "coordinates.latitude", "coordinates.longitude"
}
POLLUTANTS = ["pm25", "pm10", "o3", "no2"]
LOOKBACK = 24
HORIZON = 6
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
NUM_WORKERS = 0

# -----------------------------
# Utility
# -----------------------------
def sanitize_city_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(name)).strip("_")

def make_logger(run_name: str):
    # WandB required; fallback to CSV if not available/configured
    try:
        os.environ.setdefault("WANDB_MODE", "offline")
        return WandbLogger(
            project="global-vs-local-pm25",
            name=run_name,
            save_dir=str(OUT_DIR / "wandb"),
            offline=True,
            log_model=False
        )
    except Exception as e:
        warnings.warn(f"WandB logger unavailable ({e}). Falling back to CSV logger.")
        return CSVLogger(save_dir=str(OUT_DIR), name=run_name)

def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / (ss_tot + 1e-12))

def metrics_dict(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = safe_r2(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}

class HistoryCallback(Callback):
    def __init__(self):
        super().__init__()
        self.history = defaultdict(list)

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        for k in ["train_loss", "train_mae", "val_loss", "val_mae"]:
            if k in m and m[k] is not None:
                self.history[k].append(float(m[k].detach().cpu().item()))

class AirDataset(Dataset):
    def __init__(self, X, y, city_idx, y_raw, city_name):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        self.city_idx = torch.tensor(city_idx, dtype=torch.long)
        self.y_raw = torch.tensor(y_raw, dtype=torch.float32).unsqueeze(-1)
        self.city_name = np.array(city_name)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.city_idx[idx], self.y_raw[idx], self.city_name[idx]

def collate_train(batch):
    X = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    city_idx = torch.stack([b[2] for b in batch], dim=0)
    return X, y, city_idx

def collate_eval(batch):
    X = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    city_idx = torch.stack([b[2] for b in batch], dim=0)
    y_raw = torch.stack([b[3] for b in batch], dim=0)
    city_name = [b[4] for b in batch]
    return X, y, city_idx, y_raw, city_name

class GlobalCNNLSTM(pl.LightningModule):
    def __init__(self, input_dim=16, hidden=64, city_vocab=32, city_emb_dim=8, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.city_emb = nn.Embedding(city_vocab, city_emb_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + city_emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, x, city_idx):
        # x: [B, T, F]
        x = x.transpose(1, 2)          # [B, F, T]
        x = self.relu(self.conv(x))    # [B, 32, T]
        x = x.transpose(1, 2)          # [B, T, 32]
        out, _ = self.lstm(x)          # [B, T, 2H]
        seq_feat = out[:, -1, :]       # last step
        cemb = self.city_emb(city_idx) # [B, E]
        z = torch.cat([seq_feat, cemb], dim=1)
        return self.head(z)

    def _loss(self, pred, y):
        mae = self.l1(pred, y)
        mse = self.mse(pred, y)
        return mae + mse, mae

    def training_step(self, batch, _):
        x, y, city_idx = batch
        pred = self(x, city_idx)
        loss, mae = self._loss(pred, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_mae", mae, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        x, y, city_idx = batch
        pred = self(x, city_idx)
        loss, mae = self._loss(pred, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_mae", mae, prog_bar=False, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

class LocalLSTM(pl.LightningModule):
    def __init__(self, input_dim=16, hidden=64, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

    def _loss(self, pred, y):
        mae = self.l1(pred, y)
        mse = self.mse(pred, y)
        return mae + mse, mae

    def training_step(self, batch, _):
        x, y, _city_idx = batch
        pred = self(x)
        loss, mae = self._loss(pred, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_mae", mae, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        x, y, _city_idx = batch
        pred = self(x)
        loss, mae = self._loss(pred, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_mae", mae, prog_bar=False, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def predict_module(module, loader, city_stats):
    module.eval()
    module.to(DEVICE)

    y_true_raw_all, y_pred_raw_all, city_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            X, y_z, city_idx, y_raw, city_name = batch
            X = X.to(DEVICE)
            city_idx_t = city_idx.to(DEVICE)

            if isinstance(module, GlobalCNNLSTM):
                pred_z = module(X, city_idx_t).cpu().numpy().reshape(-1)
            else:
                pred_z = module(X).cpu().numpy().reshape(-1)

            y_raw_np = y_raw.numpy().reshape(-1)
            city_idx_np = city_idx.numpy().reshape(-1)
            city_name = list(city_name)

            # De-normalize prediction per sample using city-level pm25 stats
            pred_raw = []
            for pz, ci in zip(pred_z, city_idx_np):
                cname = city_id_to_name[int(ci)]
                mu, sigma = city_stats[cname]["pm25_mean"], city_stats[cname]["pm25_std"]
                pred_raw.append(pz * sigma + mu)

            y_true_raw_all.extend(y_raw_np.tolist())
            y_pred_raw_all.extend(pred_raw)
            city_all.extend(city_name)

    return np.array(y_true_raw_all), np.array(y_pred_raw_all), np.array(city_all)

# -----------------------------
# 1) Data loading & preprocessing
# -----------------------------
csv_files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
if len(csv_files) == 0:
    raise FileNotFoundError("No CSV files found in ./city_data/*.csv")

dfs = []
bad_files = []
for fp in csv_files:
    try:
        df = pd.read_csv(fp)
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if "city" not in df.columns or df["city"].isna().all():
            inferred_city = Path(fp).stem
            df["city"] = inferred_city

        df = df[list(REQUIRED_COLS)].copy()
        df["date.utc"] = pd.to_datetime(df["date.utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["date.utc"])
        df["parameter"] = df["parameter"].str.lower().str.strip()
        df = df[df["parameter"].isin(POLLUTANTS)].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        dfs.append(df)
    except Exception as e:
        bad_files.append((fp, str(e)))

if len(dfs) == 0:
    raise RuntimeError(f"All files failed to load. Details: {bad_files}")

if bad_files:
    print("Warning: some files skipped due to schema/data issues:")
    for bf, err in bad_files:
        print(f" - {bf}: {err}")

raw = pd.concat(dfs, ignore_index=True)
raw["city"] = raw["city"].astype(str).str.strip()
raw["station"] = raw["station"].astype(str).str.strip()

# Keep station coordinates
coords = (
    raw.groupby(["city", "station"], as_index=False)[["coordinates.latitude", "coordinates.longitude"]]
       .mean()
       .rename(columns={"coordinates.latitude": "lat", "coordinates.longitude": "lon"})
)

# Pivot long -> wide hourly pollutant table
wide = (
    raw.pivot_table(
        index=["city", "station", "date.utc"],
        columns="parameter",
        values="value",
        aggfunc="mean"
    )
    .reset_index()
)
wide.columns.name = None

for p in POLLUTANTS:
    if p not in wide.columns:
        wide[p] = np.nan

# Resample hourly per city/station
resampled = []
for (city, station), g in wide.groupby(["city", "station"]):
    g = g.sort_values("date.utc").set_index("date.utc")
    full = g.asfreq("H")
    full["city"] = city
    full["station"] = station
    resampled.append(full.reset_index())

df = pd.concat(resampled, ignore_index=True)
df = df.merge(coords, on=["city", "station"], how="left")

# Forward-fill gaps <3 hours (limit=2)
df = df.sort_values(["city", "station", "date.utc"]).reset_index(drop=True)
for p in POLLUTANTS:
    df[p] = (
        df.groupby(["city", "station"], group_keys=False)[p]
          .apply(lambda s: s.ffill(limit=2))
    )

# Drop stations with >10% missing for PM2.5 after short-gap fill
station_missing = (
    df.groupby(["city", "station"])["pm25"]
      .apply(lambda s: s.isna().mean())
      .reset_index(name="missing_ratio")
)
keep_stations = station_missing[station_missing["missing_ratio"] <= 0.10][["city", "station"]]
df = df.merge(keep_stations, on=["city", "station"], how="inner")

# Save raw pollutant copies for de-normalized metrics
for p in POLLUTANTS:
    df[f"{p}_raw"] = df[p]

# Z-score normalization per pollutant per city
city_stats = {}
for city, gidx in df.groupby("city").groups.items():
    city_stats[city] = {}
    idx = list(gidx)
    for p in POLLUTANTS:
        mu = df.loc[idx, p].mean(skipna=True)
        sigma = df.loc[idx, p].std(skipna=True)
        sigma = float(sigma) if pd.notna(sigma) and sigma > 1e-8 else 1.0
        city_stats[city][f"{p}_mean"] = float(mu) if pd.notna(mu) else 0.0
        city_stats[city][f"{p}_std"] = sigma
        df.loc[idx, f"{p}_z"] = (df.loc[idx, p] - city_stats[city][f"{p}_mean"]) / sigma

# Feature engineering -> exactly 16 features per timestep
# 4 pollutant z + 4 rolling mean + 4 first diff + 2 hour cyc + 2 coords
df["hour"] = df["date.utc"].dt.hour
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

# Fill coords if missing
df["lat"] = df.groupby(["city", "station"])["lat"].transform(lambda s: s.fillna(s.mean()))
df["lon"] = df.groupby(["city", "station"])["lon"].transform(lambda s: s.fillna(s.mean()))
df["lat"] = df["lat"].fillna(df["lat"].mean())
df["lon"] = df["lon"].fillna(df["lon"].mean())

for p in POLLUTANTS:
    z = f"{p}_z"
    df[f"{z}_roll3"] = df.groupby(["city", "station"])[z].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df[f"{z}_diff1"] = df.groupby(["city", "station"])[z].transform(lambda s: s.diff())

feature_cols = [
    "pm25_z", "pm10_z", "o3_z", "no2_z",
    "pm25_z_roll3", "pm10_z_roll3", "o3_z_roll3", "no2_z_roll3",
    "pm25_z_diff1", "pm10_z_diff1", "o3_z_diff1", "no2_z_diff1",
    "hour_sin", "hour_cos", "lat", "lon"
]

# Clean NA introduced by lag/diff
df = df.sort_values(["city", "station", "date.utc"]).reset_index(drop=True)
df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

# City mapping
cities = sorted(df["city"].dropna().unique().tolist())
if len(cities) < 6:
    warnings.warn(f"Expected 6 cities; found {len(cities)}: {cities}")

city_name_to_id = {c: i for i, c in enumerate(cities)}
city_id_to_name = {i: c for c, i in city_name_to_id.items()}

# Create windows X=[B,24,16], y=pm25(t+6)
records = []
for (city, station), g in df.groupby(["city", "station"]):
    g = g.sort_values("date.utc").reset_index(drop=True)
    # Require valid target and features
    for i in range(LOOKBACK - 1, len(g) - HORIZON):
        hist = g.iloc[i - LOOKBACK + 1:i + 1]
        tgt = g.iloc[i + HORIZON]
        if hist[feature_cols].isna().any().any():
            continue
        if pd.isna(tgt["pm25_z"]) or pd.isna(tgt["pm25_raw"]):
            continue
        X = hist[feature_cols].values.astype(np.float32)      # [24,16]
        y_z = float(tgt["pm25_z"])                            # normalized target
        y_raw = float(tgt["pm25_raw"])                        # original target
        t_target = tgt["date.utc"]
        records.append({
            "city": city,
            "station": station,
            "city_idx": city_name_to_id[city],
            "target_time": t_target,
            "X": X,
            "y_z": y_z,
            "y_raw": y_raw
        })

if len(records) == 0:
    raise RuntimeError("No training windows created. Check missingness/coverage in source files.")

win_df = pd.DataFrame(records)

# Temporal split per city for local models (70/15/15)
win_df["split_local"] = ""
for city, idx in win_df.groupby("city").groups.items():
    cidx = list(idx)
    tmp = win_df.loc[cidx].sort_values("target_time")
    n = len(tmp)
    tr_end = max(1, int(0.70 * n))
    va_end = max(tr_end + 1, int(0.85 * n))
    tr_ids = tmp.index[:tr_end]
    va_ids = tmp.index[tr_end:va_end]
    te_ids = tmp.index[va_end:]
    win_df.loc[tr_ids, "split_local"] = "train"
    win_df.loc[va_ids, "split_local"] = "val"
    win_df.loc[te_ids, "split_local"] = "test"

# Required split: Train(4 cities), Val(Sydney), Test(Los Angeles zero-shot)
def pick_city(preferred_names, available):
    lowered = {c.lower(): c for c in available}
    for name in preferred_names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    # soft contains match
    for c in available:
        lc = c.lower()
        for name in preferred_names:
            n = name.lower().replace(" ", "")
            if n in lc.replace(" ", ""):
                return c
    return None

val_city = pick_city(["Sydney"], cities)
test_city = pick_city(["Los Angeles", "LA", "LosAngeles"], cities)

if val_city is None or test_city is None or val_city == test_city:
    # fallback deterministic
    fallback = sorted(cities)
    val_city = fallback[-2] if len(fallback) >= 2 else fallback[0]
    test_city = fallback[-1]
    warnings.warn(f"Could not robustly find Sydney/Los Angeles. Using val={val_city}, test={test_city}.")

train_cities = [c for c in cities if c not in {val_city, test_city}]
if len(train_cities) < 4:
    warnings.warn(f"Only {len(train_cities)} train cities available: {train_cities}. Using all available.")
else:
    train_cities = train_cities[:4]

print("Global split:")
print(f"  Train cities (4): {train_cities}")
print(f"  Val city: {val_city}")
print(f"  Test city (zero-shot): {test_city}")

# Build arrays
X_all = np.stack(win_df["X"].values)
y_all = win_df["y_z"].values.astype(np.float32)
y_raw_all = win_df["y_raw"].values.astype(np.float32)
city_idx_all = win_df["city_idx"].values.astype(np.int64)
city_name_all = win_df["city"].values.astype(str)

# Save processed artifact
torch.save(
    {
        "X": X_all,
        "y_z": y_all,
        "y_raw": y_raw_all,
        "city_idx": city_idx_all,
        "city_name": city_name_all,
        "feature_cols": feature_cols,
        "city_name_to_id": city_name_to_id,
        "city_stats": city_stats,
        "lookback": LOOKBACK,
        "horizon": HORIZON
    },
    OUT_DIR / "processed_dataset.pt"
)

# Global train/val/test indices
global_train_idx = win_df.index[(win_df["city"].isin(train_cities)) & (win_df["split_local"] == "train")].tolist()
global_val_idx = win_df.index[(win_df["city"] == val_city) & (win_df["split_local"] == "val")].tolist()
global_test_idx = win_df.index[(win_df["city"] == test_city) & (win_df["split_local"] == "test")].tolist()

if len(global_train_idx) == 0 or len(global_val_idx) == 0 or len(global_test_idx) == 0:
    raise RuntimeError(
        f"Insufficient split windows. train={len(global_train_idx)}, val={len(global_val_idx)}, test={len(global_test_idx)}"
    )

train_ds_g = AirDataset(X_all[global_train_idx], y_all[global_train_idx], city_idx_all[global_train_idx], y_raw_all[global_train_idx], city_name_all[global_train_idx])
val_ds_g = AirDataset(X_all[global_val_idx], y_all[global_val_idx], city_idx_all[global_val_idx], y_raw_all[global_val_idx], city_name_all[global_val_idx])
test_ds_g = AirDataset(X_all[global_test_idx], y_all[global_test_idx], city_idx_all[global_test_idx], y_raw_all[global_test_idx], city_name_all[global_test_idx])

train_loader_g = DataLoader(train_ds_g, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_train)
val_loader_g = DataLoader(val_ds_g, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_train)
test_loader_g = DataLoader(test_ds_g, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_eval)

# -----------------------------
# 2) Train Global CNN-LSTM
# -----------------------------
global_model = GlobalCNNLSTM(
    input_dim=16,
    hidden=64,
    city_vocab=len(cities),
    city_emb_dim=8,
    lr=LR
)

hist_global = HistoryCallback()
ckpt_global = ModelCheckpoint(
    dirpath=str(MODELS_DIR),
    filename="global_cnn_lstm_best",
    monitor="val_loss",
    mode="min",
    save_top_k=1
)
es_global = EarlyStopping(monitor="val_loss", mode="min", patience=8)

trainer_global = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator="auto",
    devices="auto",
    logger=make_logger("global_cnn_lstm"),
    callbacks=[hist_global, ckpt_global, es_global],
    deterministic=True,
    enable_progress_bar=True,
    log_every_n_steps=10
)

trainer_global.fit(global_model, train_loader_g, val_loader_g)

best_global_path = ckpt_global.best_model_path if ckpt_global.best_model_path else None
if best_global_path and os.path.exists(best_global_path):
    global_best = GlobalCNNLSTM.load_from_checkpoint(best_global_path)
else:
    global_best = global_model

# Save exact required model file
torch.save(global_best.state_dict(), MODELS_DIR / "global_cnn_lstm.pt")

# -----------------------------
# 3) Evaluate Global (per city test split + LA zero-shot)
# -----------------------------
global_city_metrics = {}
global_preds_for_scatter = {}

for city in cities:
    city_test_idx = win_df.index[(win_df["city"] == city) & (win_df["split_local"] == "test")].tolist()
    if len(city_test_idx) == 0:
        continue
    ds = AirDataset(X_all[city_test_idx], y_all[city_test_idx], city_idx_all[city_test_idx], y_raw_all[city_test_idx], city_name_all[city_test_idx])
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_eval)
    yt, yp, cn = predict_module(global_best, dl, city_stats)
    global_city_metrics[city] = metrics_dict(yt, yp)
    global_preds_for_scatter[city] = (yt, yp)

# Explicit zero-shot LA evaluation (required)
la_key = test_city
if la_key not in global_city_metrics:
    yt, yp, _ = predict_module(global_best, test_loader_g, city_stats)
    global_city_metrics[la_key] = metrics_dict(yt, yp)
    global_preds_for_scatter[la_key] = (yt, yp)

# -----------------------------
# 4) Train Local LSTM per city
# -----------------------------
local_city_metrics = {}
local_histories = {}

for city in cities:
    c_train = win_df.index[(win_df["city"] == city) & (win_df["split_local"] == "train")].tolist()
    c_val = win_df.index[(win_df["city"] == city) & (win_df["split_local"] == "val")].tolist()
    c_test = win_df.index[(win_df["city"] == city) & (win_df["split_local"] == "test")].tolist()

    if min(len(c_train), len(c_val), len(c_test)) == 0:
        warnings.warn(f"Skipping local model for {city}: insufficient split windows.")
        continue

    train_ds = AirDataset(X_all[c_train], y_all[c_train], city_idx_all[c_train], y_raw_all[c_train], city_name_all[c_train])
    val_ds = AirDataset(X_all[c_val], y_all[c_val], city_idx_all[c_val], y_raw_all[c_val], city_name_all[c_val])
    test_ds = AirDataset(X_all[c_test], y_all[c_test], city_idx_all[c_test], y_raw_all[c_test], city_name_all[c_test])

    tr_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_train)
    va_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_train)
    te_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_eval)

    local_model = LocalLSTM(input_dim=16, hidden=64, lr=LR)
    hist_local = HistoryCallback()

    ckpt_local = ModelCheckpoint(
        dirpath=str(MODELS_DIR),
        filename=f"local_lstm_{sanitize_city_name(city)}_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )
    es_local = EarlyStopping(monitor="val_loss", mode="min", patience=8)

    trainer_local = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices="auto",
        logger=make_logger(f"local_lstm_{sanitize_city_name(city)}"),
        callbacks=[hist_local, ckpt_local, es_local],
        deterministic=True,
        enable_progress_bar=False,
        log_every_n_steps=10
    )

    trainer_local.fit(local_model, tr_dl, va_dl)

    best_local_path = ckpt_local.best_model_path if ckpt_local.best_model_path else None
    if best_local_path and os.path.exists(best_local_path):
        local_best = LocalLSTM.load_from_checkpoint(best_local_path)
    else:
        local_best = local_model

    torch.save(local_best.state_dict(), MODELS_DIR / f"local_lstm_{sanitize_city_name(city)}.pt")

    yt, yp, _ = predict_module(local_best, te_dl, city_stats)
    local_city_metrics[city] = metrics_dict(yt, yp)
    local_histories[city] = dict(hist_local.history)

# -----------------------------
# 5) Comparison table + outputs
# -----------------------------
rows = []
for city in sorted(set(list(global_city_metrics.keys()) + list(local_city_metrics.keys()))):
    g = global_city_metrics.get(city, {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan})
    l = local_city_metrics.get(city, {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan})

    imp_pct = np.nan
    if np.isfinite(l["MAE"]) and l["MAE"] > 1e-8 and np.isfinite(g["MAE"]):
        imp_pct = 100.0 * (l["MAE"] - g["MAE"]) / l["MAE"]  # positive = global better

    rows.append({
        "City": city,
        "Local_LSTM_MAE": l["MAE"],
        "Global_CNNLSTM_MAE": g["MAE"],
        "Improvement_%": imp_pct,
        "Local_LSTM_RMSE": l["RMSE"],
        "Global_CNNLSTM_RMSE": g["RMSE"],
        "Local_LSTM_R2": l["R2"],
        "Global_CNNLSTM_R2": g["R2"],
        "Winner": "Global" if (np.isfinite(imp_pct) and imp_pct > 0) else "Local/NA"
    })

comparison_df = pd.DataFrame(rows).sort_values("City")
comparison_path = RESULTS_DIR / "comparison_table.csv"
comparison_df.to_csv(comparison_path, index=False)

# -----------------------------
# 6) Plots
# -----------------------------
# loss_curves.png with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax1, ax2, ax3, ax4 = axes.flatten()

# Global losses
gh = hist_global.history
if len(gh.get("train_loss", [])) > 0:
    ax1.plot(gh["train_loss"], label="train_loss")
if len(gh.get("val_loss", [])) > 0:
    ax1.plot(gh["val_loss"], label="val_loss")
ax1.set_title("Global CNN-LSTM Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

# Global MAE
if len(gh.get("train_mae", [])) > 0:
    ax2.plot(gh["train_mae"], label="train_mae")
if len(gh.get("val_mae", [])) > 0:
    ax2.plot(gh["val_mae"], label="val_mae")
ax2.set_title("Global CNN-LSTM MAE")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MAE")
ax2.legend()

# Local mean loss across cities
max_e = 0
for city, h in local_histories.items():
    max_e = max(max_e, len(h.get("train_loss", [])), len(h.get("val_loss", [])))

if max_e > 0:
    train_stack, val_stack = [], []
    for city, h in local_histories.items():
        tl = h.get("train_loss", [])
        vl = h.get("val_loss", [])
        if len(tl) > 0:
            arr = np.full(max_e, np.nan); arr[:len(tl)] = tl; train_stack.append(arr)
        if len(vl) > 0:
            arr = np.full(max_e, np.nan); arr[:len(vl)] = vl; val_stack.append(arr)
    if train_stack:
        ax3.plot(np.nanmean(np.vstack(train_stack), axis=0), label="mean_train_loss")
    if val_stack:
        ax3.plot(np.nanmean(np.vstack(val_stack), axis=0), label="mean_val_loss")
ax3.set_title("Local LSTM Mean Loss (Across Cities)")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Loss")
ax3.legend()

# Local mean MAE across cities
if max_e > 0:
    train_stack, val_stack = [], []
    for city, h in local_histories.items():
        tm = h.get("train_mae", [])
        vm = h.get("val_mae", [])
        if len(tm) > 0:
            arr = np.full(max_e, np.nan); arr[:len(tm)] = tm; train_stack.append(arr)
        if len(vm) > 0:
            arr = np.full(max_e, np.nan); arr[:len(vm)] = vm; val_stack.append(arr)
    if train_stack:
        ax4.plot(np.nanmean(np.vstack(train_stack), axis=0), label="mean_train_mae")
    if val_stack:
        ax4.plot(np.nanmean(np.vstack(val_stack), axis=0), label="mean_val_mae")
ax4.set_title("Local LSTM Mean MAE (Across Cities)")
ax4.set_xlabel("Epoch")
ax4.set_ylabel("MAE")
ax4.legend()

fig.tight_layout()
fig.savefig(RESULTS_DIR / "loss_curves.png", dpi=200)
plt.close(fig)

# pred_actual_scatter.png (test set; include LA/global and LA/local if available)
fig, ax = plt.subplots(figsize=(8, 7))
if la_key in global_preds_for_scatter:
    yt, yp = global_preds_for_scatter[la_key]
    ax.scatter(yt, yp, s=12, alpha=0.6, label=f"Global ({la_key})")
if la_key in local_city_metrics:
    # recreate local test predictions for LA if local trained
    c_test = win_df.index[(win_df["city"] == la_key) & (win_df["split_local"] == "test")].tolist()
    if len(c_test) > 0 and (MODELS_DIR / f"local_lstm_{sanitize_city_name(la_key)}.pt").exists():
        ds = AirDataset(X_all[c_test], y_all[c_test], city_idx_all[c_test], y_raw_all[c_test], city_name_all[c_test])
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_eval)
        lm = LocalLSTM(input_dim=16, hidden=64, lr=LR)
        lm.load_state_dict(torch.load(MODELS_DIR / f"local_lstm_{sanitize_city_name(la_key)}.pt", map_location="cpu"))
        yt2, yp2, _ = predict_module(lm, dl, city_stats)
        ax.scatter(yt2, yp2, s=12, alpha=0.5, label=f"Local ({la_key})")

lims = ax.get_xlim()
line_min = min(ax.get_xlim()[0], ax.get_ylim()[0])
line_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([line_min, line_max], [line_min, line_max], "k--", lw=1)
ax.set_title("Predicted vs Actual PM2.5 (Test)")
ax.set_xlabel("Actual PM2.5")
ax.set_ylabel("Predicted PM2.5")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS_DIR / "pred_actual_scatter.png", dpi=200)
plt.close(fig)

# -----------------------------
# 7) Paper-ready markdown
# -----------------------------
def df_to_md(dfin):
    return dfin.to_markdown(index=False)

summary_lines = []
summary_lines.append("# Global vs Local Deep Learning for Multi-City PM2.5 Forecasting")
summary_lines.append("")
summary_lines.append("## Experimental Setup")
summary_lines.append(f"- Seed: {SEED}")
summary_lines.append(f"- Device: {DEVICE}")
summary_lines.append(f"- Pollutants used: {', '.join(POLLUTANTS)} (pm25 target; pm10/o3/no2 features)")
summary_lines.append(f"- Window: {LOOKBACK} hours, Horizon: +{HORIZON} hours")
summary_lines.append(f"- Global split: Train={train_cities}, Val={val_city}, Test={test_city} (zero-shot)")
summary_lines.append("")
summary_lines.append("## Comparison Table (Global vs Local)")
summary_lines.append(df_to_md(comparison_df[[
    "City", "Local_LSTM_MAE", "Global_CNNLSTM_MAE", "Improvement_%", "Winner",
    "Local_LSTM_RMSE", "Global_CNNLSTM_RMSE", "Local_LSTM_R2", "Global_CNNLSTM_R2"
]].round(4)))
summary_lines.append("")
summary_lines.append("## Key Finding")
if test_city in comparison_df["City"].values:
    row = comparison_df[comparison_df["City"] == test_city].iloc[0]
    summary_lines.append(
        f"- Zero-shot city ({test_city}): Local MAE={row['Local_LSTM_MAE']:.3f}, "
        f"Global MAE={row['Global_CNNLSTM_MAE']:.3f}, Improvement={row['Improvement_%']:.2f}%."
    )
else:
    summary_lines.append("- Zero-shot city row unavailable due to data split constraints.")
summary_lines.append("")
summary_lines.append("## Files Generated")
summary_lines.append(f"- `{comparison_path}`")
summary_lines.append(f"- `{RESULTS_DIR / 'loss_curves.png'}`")
summary_lines.append(f"- `{RESULTS_DIR / 'pred_actual_scatter.png'}`")
summary_lines.append(f"- `{MODELS_DIR / 'global_cnn_lstm.pt'}`")
summary_lines.append(f"- `{OUT_DIR / 'processed_dataset.pt'}`")

paper_md = "\n".join(summary_lines)
(RESULTS_DIR / "paper_results.md").write_text(paper_md)
(OUT_DIR / "paper_results.md").write_text(paper_md)  # also save at output root per requested structure

print(f"Saved: {comparison_path}")
print(f"Saved: {RESULTS_DIR / 'loss_curves.png'}")
print(f"Saved: {RESULTS_DIR / 'pred_actual_scatter.png'}")
print(f"Saved: {RESULTS_DIR / 'paper_results.md'}")
print(f"Saved: {OUT_DIR / 'paper_results.md'}")
print(f"Saved: {MODELS_DIR / 'global_cnn_lstm.pt'}")
print(f"Saved: {OUT_DIR / 'processed_dataset.pt'}")
print("PIPELINE COMPLETE - PAPER READY")
