import os
import re
import glob
import random
import warnings
import importlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    pl = importlib.import_module("lightning.pytorch")
    callbacks_mod = importlib.import_module("lightning.pytorch.callbacks")
    loggers_mod = importlib.import_module("lightning.pytorch.loggers")
except Exception:
    pl = importlib.import_module("pytorch_lightning")
    callbacks_mod = importlib.import_module("pytorch_lightning.callbacks")
    loggers_mod = importlib.import_module("pytorch_lightning.loggers")

EarlyStopping = callbacks_mod.EarlyStopping
ModelCheckpoint = callbacks_mod.ModelCheckpoint
Callback = callbacks_mod.Callback
WandbLogger = loggers_mod.WandbLogger
CSVLogger = loggers_mod.CSVLogger


# -----------------------------
# Config
# -----------------------------
SEED = 42
LOOKBACK = 24
HORIZON = 6
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
NUM_WORKERS = 0

POLLUTANTS = ["pm25", "pm10", "o3", "no2"]
REQUIRED_COLS = {
    "city", "station", "date.utc", "parameter", "value",
    "coordinates.latitude", "coordinates.longitude"
}

DATA_DIR = Path("./city_data")
OUT_DIR = Path("./output")
MODELS_DIR = OUT_DIR / "models"
RESULTS_DIR = OUT_DIR / "results"


# -----------------------------
# Helpers
# -----------------------------
def set_seed(seed: int = 42):
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sanitize_city_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(name)).strip("_")


def normalize_city_label(label: str) -> str:
    x = str(label).strip()
    x = re.sub(r"\bAQI\b", "", x, flags=re.IGNORECASE)
    x = x.replace("(", " ").replace(")", " ")
    x = re.sub(r"\s+", " ", x).strip()
    aliases = {
        "Dehi": "Delhi",
        "LosAngeles": "Los Angeles",
    }
    return aliases.get(x, x)


def _pick_col(df_cols, candidates):
    lower_map = {c.lower(): c for c in df_cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def canonicalize_openaq_columns(df: pd.DataFrame, fallback_city: str) -> pd.DataFrame:
    cols = list(df.columns)
    mapping = {
        "city": _pick_col(cols, ["city"]),
        "station": _pick_col(cols, ["station", "location_name", "location", "location_id"]),
        "date.utc": _pick_col(cols, ["date.utc", "datetimeUtc", "datetime_utc", "utc", "date", "datetime"]),
        "parameter": _pick_col(cols, ["parameter", "pollutant"]),
        "value": _pick_col(cols, ["value", "concentration"]),
        "coordinates.latitude": _pick_col(cols, ["coordinates.latitude", "latitude", "lat"]),
        "coordinates.longitude": _pick_col(cols, ["coordinates.longitude", "longitude", "lon", "lng"]),
    }

    required_without_city = [
        "station", "date.utc", "parameter", "value",
        "coordinates.latitude", "coordinates.longitude"
    ]
    for k in required_without_city:
        if mapping[k] is None:
            raise ValueError(f"Missing required source column for '{k}'")

    out = pd.DataFrame({
        "station": df[mapping["station"]].astype(str),
        "date.utc": df[mapping["date.utc"]],
        "parameter": df[mapping["parameter"]],
        "value": df[mapping["value"]],
        "coordinates.latitude": df[mapping["coordinates.latitude"]],
        "coordinates.longitude": df[mapping["coordinates.longitude"]],
    })

    if mapping["city"] is not None:
        out["city"] = df[mapping["city"]].astype(str)
        empty_city = out["city"].isna() | out["city"].astype(str).str.strip().eq("")
        out.loc[empty_city, "city"] = fallback_city
    else:
        out["city"] = fallback_city

    return out[[
        "city", "station", "date.utc", "parameter", "value",
        "coordinates.latitude", "coordinates.longitude"
    ]]


def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def metrics_dict(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = safe_r2(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def pick_city(preferred_names, available):
    lowered = {c.lower(): c for c in available}
    for name in preferred_names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    for c in available:
        lc = c.lower().replace(" ", "")
        for name in preferred_names:
            if name.lower().replace(" ", "") in lc:
                return c
    return None


def make_logger(run_name: str, out_dir: Path):
    # WandB required by spec; offline mode by default so script runs without key.
    try:
        os.environ.setdefault("WANDB_MODE", "offline")
        return WandbLogger(
            project="global-vs-local-pm25",
            name=run_name,
            save_dir=str(out_dir / "wandb"),
            offline=True,
            log_model=False,
        )
    except Exception as e:
        warnings.warn(f"WandB unavailable ({e}); falling back to CSV logger.")
        return CSVLogger(save_dir=str(out_dir), name=run_name)


def ensure_dirs():
    for d in [OUT_DIR, MODELS_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


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
            nn.Linear(64, 1),
        )
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, x, city_idx):
        # x: [B, T, F]
        x = x.transpose(1, 2)      # [B, F, T]
        x = self.relu(self.conv(x))
        x = x.transpose(1, 2)      # [B, T, 32]
        out, _ = self.lstm(x)
        seq_feat = out[:, -1, :]
        cemb = self.city_emb(city_idx)
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


def predict_module(module, loader, city_stats, city_id_to_name, device):
    module.eval().to(device)
    y_true_raw_all, y_pred_raw_all, city_all = [], [], []

    with torch.no_grad():
        for X, y_z, city_idx, y_raw, city_name in loader:
            X = X.to(device)
            city_idx_t = city_idx.to(device)

            if isinstance(module, GlobalCNNLSTM):
                pred_z = module(X, city_idx_t).cpu().numpy().reshape(-1)
            else:
                pred_z = module(X).cpu().numpy().reshape(-1)

            y_raw_np = y_raw.numpy().reshape(-1)
            city_idx_np = city_idx.numpy().reshape(-1)
            city_name = list(city_name)

            pred_raw = []
            for pz, ci in zip(pred_z, city_idx_np):
                cname = city_id_to_name[int(ci)]
                mu = city_stats[cname]["pm25_mean"]
                sigma = city_stats[cname]["pm25_std"]
                pred_raw.append(pz * sigma + mu)

            y_true_raw_all.extend(y_raw_np.tolist())
            y_pred_raw_all.extend(pred_raw)
            city_all.extend(city_name)

    return np.array(y_true_raw_all), np.array(y_pred_raw_all), np.array(city_all)


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    set_seed(SEED)
    ensure_dirs()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Load all city CSVs
    csv_files = []
    if DATA_DIR.exists():
        csv_files.extend(glob.glob(str(DATA_DIR / "*.csv")))
        csv_files.extend(glob.glob(str(DATA_DIR / "*.CSV")))
        csv_files.extend(glob.glob(str(DATA_DIR / "**/*.csv"), recursive=True))
        csv_files.extend(glob.glob(str(DATA_DIR / "**/*.CSV"), recursive=True))
    if not csv_files:
        # Fallback to current workspace layout (city folders in project root).
        csv_files.extend(glob.glob("./**/*.csv", recursive=True))
        csv_files.extend(glob.glob("./**/*.CSV", recursive=True))
        csv_files = [f for f in csv_files if "/output/" not in f.replace("\\", "/")]
    csv_files = sorted(set(csv_files))
    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found. Expected ./city_data/*.csv or nested city folders with CSV files."
        )

    dfs, bad_files = [], []
    for fp in csv_files:
        try:
            raw_df = pd.read_csv(fp)
            inferred_city = normalize_city_label(Path(fp).parent.name)
            df = canonicalize_openaq_columns(raw_df, fallback_city=inferred_city)

            df["date.utc"] = pd.to_datetime(df["date.utc"], utc=True, errors="coerce")
            df = df.dropna(subset=["date.utc"])
            df["parameter"] = df["parameter"].str.lower().str.strip()
            df = df[df["parameter"].isin(POLLUTANTS)].copy()
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"])
            dfs.append(df)
        except Exception as e:
            bad_files.append((fp, str(e)))

    if not dfs:
        raise RuntimeError(f"All files failed to load. Details: {bad_files}")

    if bad_files:
        print("Warning: some files were skipped:")
        for fp, err in bad_files:
            print(f" - {fp}: {err}")

    raw = pd.concat(dfs, ignore_index=True)
    raw["city"] = raw["city"].astype(str).map(normalize_city_label)
    raw["station"] = raw["station"].astype(str).str.strip()

    # station coordinates
    coords = (
        raw.groupby(["city", "station"], as_index=False)[["coordinates.latitude", "coordinates.longitude"]]
        .mean()
        .rename(columns={"coordinates.latitude": "lat", "coordinates.longitude": "lon"})
    )

    # 2) Parse and pivot wide
    wide = (
        raw.pivot_table(
            index=["city", "station", "date.utc"],
            columns="parameter",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )
    wide.columns.name = None
    for p in POLLUTANTS:
        if p not in wide.columns:
            wide[p] = np.nan

    # hourly frequency per city/station
    resampled = []
    for (city, station), g in wide.groupby(["city", "station"]):
        g = g.sort_values("date.utc").set_index("date.utc")
        full = g.asfreq("h")
        full["city"] = city
        full["station"] = station
        resampled.append(full.reset_index())

    df = pd.concat(resampled, ignore_index=True)
    df = df.merge(coords, on=["city", "station"], how="left")
    df = df.sort_values(["city", "station", "date.utc"]).reset_index(drop=True)

    # 3) Forward-fill gaps <3h
    for p in POLLUTANTS:
        df[p] = df.groupby(["city", "station"], group_keys=False)[p].apply(lambda s: s.ffill(limit=2))

    # 4) Drop stations >10% missing pm25
    station_missing = (
        df.groupby(["city", "station"])["pm25"]
        .apply(lambda s: s.isna().mean())
        .reset_index(name="missing_ratio")
    )
    keep = station_missing[station_missing["missing_ratio"] <= 0.10][["city", "station"]]
    df = df.merge(keep, on=["city", "station"], how="inner")

    # keep raw target for denormalized metrics
    for p in POLLUTANTS:
        df[f"{p}_raw"] = df[p]

    # 5) Z-score normalization per pollutant per city
    city_stats = {}
    for city, idx in df.groupby("city").groups.items():
        idx = list(idx)
        city_stats[city] = {}
        for p in POLLUTANTS:
            mu = df.loc[idx, p].mean(skipna=True)
            sigma = df.loc[idx, p].std(skipna=True)
            sigma = float(sigma) if pd.notna(sigma) and sigma > 1e-8 else 1.0
            city_stats[city][f"{p}_mean"] = float(mu) if pd.notna(mu) else 0.0
            city_stats[city][f"{p}_std"] = sigma
            df.loc[idx, f"{p}_z"] = (df.loc[idx, p] - city_stats[city][f"{p}_mean"]) / sigma

    # 6) Feature engineering -> X shape [batch,24,16]
    # 4 z + 4 rolling mean + 4 diff + 2 hour cyc + 2 coords = 16
    df["hour"] = df["date.utc"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    df["lat"] = df.groupby(["city", "station"])["lat"].transform(lambda s: s.fillna(s.mean()))
    df["lon"] = df.groupby(["city", "station"])["lon"].transform(lambda s: s.fillna(s.mean()))
    df["lat"] = df["lat"].fillna(df["lat"].mean())
    df["lon"] = df["lon"].fillna(df["lon"].mean())

    for p in POLLUTANTS:
        z = f"{p}_z"
        df[f"{z}_roll3"] = df.groupby(["city", "station"])[z].transform(
            lambda s: s.rolling(3, min_periods=1).mean()
        )
        df[f"{z}_diff1"] = df.groupby(["city", "station"])[z].transform(lambda s: s.diff())

    feature_cols = [
        "pm25_z", "pm10_z", "o3_z", "no2_z",
        "pm25_z_roll3", "pm10_z_roll3", "o3_z_roll3", "no2_z_roll3",
        "pm25_z_diff1", "pm10_z_diff1", "o3_z_diff1", "no2_z_diff1",
        "hour_sin", "hour_cos", "lat", "lon",
    ]

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # city id mapping
    cities = sorted(df["city"].dropna().unique().tolist())
    if len(cities) < 6:
        warnings.warn(f"Expected ~6 cities, found {len(cities)}: {cities}")

    city_name_to_id = {c: i for i, c in enumerate(cities)}
    city_id_to_name = {i: c for c, i in city_name_to_id.items()}

    # 7) Build windows: y = pm25(t+6)
    records = []
    for (city, station), g in df.groupby(["city", "station"]):
        g = g.sort_values("date.utc").reset_index(drop=True)
        for i in range(LOOKBACK - 1, len(g) - HORIZON):
            hist = g.iloc[i - LOOKBACK + 1 : i + 1]
            tgt = g.iloc[i + HORIZON]

            if hist[feature_cols].isna().any().any():
                continue
            if pd.isna(tgt["pm25_z"]) or pd.isna(tgt["pm25_raw"]):
                continue

            records.append(
                {
                    "city": city,
                    "station": station,
                    "city_idx": city_name_to_id[city],
                    "target_time": tgt["date.utc"],
                    "X": hist[feature_cols].values.astype(np.float32),
                    "y_z": float(tgt["pm25_z"]),
                    "y_raw": float(tgt["pm25_raw"]),
                }
            )

    if not records:
        raise RuntimeError("No windows created. Check missingness and station coverage.")

    win_df = pd.DataFrame(records)

    # local temporal split per city: 70/15/15
    win_df["split_local"] = ""
    for city, idx in win_df.groupby("city").groups.items():
        tmp = win_df.loc[list(idx)].sort_values("target_time")
        n = len(tmp)
        tr_end = max(1, int(0.70 * n))
        va_end = max(tr_end + 1, int(0.85 * n))
        win_df.loc[tmp.index[:tr_end], "split_local"] = "train"
        win_df.loc[tmp.index[tr_end:va_end], "split_local"] = "val"
        win_df.loc[tmp.index[va_end:], "split_local"] = "test"

    # global split: Train(4 cities), Val(Sydney), Test(Los Angeles zero-shot)
    # Make choice robust to sparse cities by requiring non-empty split counts.
    split_counts = (
        win_df.groupby(["city", "split_local"])
        .size()
        .unstack(fill_value=0)
    )
    for col in ["train", "val", "test"]:
        if col not in split_counts.columns:
            split_counts[col] = 0
    split_counts = split_counts.reset_index()

    eligible_val = split_counts[split_counts["val"] > 0]["city"].tolist()
    eligible_test = split_counts[split_counts["test"] > 0]["city"].tolist()
    eligible_train = split_counts[split_counts["train"] > 0]["city"].tolist()

    preferred_val = pick_city(["Sydney", "Civic", "Canberra"], eligible_val)
    preferred_test = pick_city(["Los Angeles", "LA", "LosAngeles"], eligible_test)

    test_city = preferred_test
    if test_city is None and eligible_test:
        # largest test split fallback
        test_city = split_counts.sort_values("test", ascending=False)["city"].iloc[0]

    val_city = preferred_val
    if (val_city is None or val_city == test_city) and eligible_val:
        val_candidates = split_counts[
            (split_counts["city"] != test_city) & (split_counts["val"] > 0)
        ].sort_values("val", ascending=False)
        if len(val_candidates) > 0:
            val_city = val_candidates["city"].iloc[0]

    if val_city is None or test_city is None:
        raise RuntimeError(
            f"Could not select valid val/test cities. Eligible val={eligible_val}, eligible test={eligible_test}"
        )

    train_pool = [c for c in eligible_train if c not in {val_city, test_city}]
    train_cities = train_pool[:4] if len(train_pool) >= 4 else train_pool
    if len(train_cities) < 1:
        raise RuntimeError(f"No eligible train cities after val/test selection. train_pool={train_pool}")
    if len(train_cities) < 4:
        warnings.warn(f"Only {len(train_cities)} train cities available; using all: {train_cities}")

    print("Global split:")
    print(f"  Train cities: {train_cities}")
    print(f"  Val city: {val_city}")
    print(f"  Test city (zero-shot): {test_city}")
    print("  Split counts by city:")
    print(split_counts.sort_values("city").to_string(index=False))

    X_all = np.stack(win_df["X"].values)
    y_all = win_df["y_z"].values.astype(np.float32)
    y_raw_all = win_df["y_raw"].values.astype(np.float32)
    city_idx_all = win_df["city_idx"].values.astype(np.int64)
    city_name_all = win_df["city"].values.astype(str)

    # save processed dataset
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
            "horizon": HORIZON,
        },
        OUT_DIR / "processed_dataset.pt",
    )

    # global split indices
    global_train_idx = win_df.index[
        (win_df["city"].isin(train_cities)) & (win_df["split_local"] == "train")
    ].tolist()
    global_val_idx = win_df.index[
        (win_df["city"] == val_city) & (win_df["split_local"] == "val")
    ].tolist()
    global_test_idx = win_df.index[
        (win_df["city"] == test_city) & (win_df["split_local"] == "test")
    ].tolist()

    if min(len(global_train_idx), len(global_val_idx), len(global_test_idx)) == 0:
        raise RuntimeError(
            f"Insufficient global split windows: train={len(global_train_idx)}, val={len(global_val_idx)}, test={len(global_test_idx)}"
        )

    train_ds_g = AirDataset(
        X_all[global_train_idx], y_all[global_train_idx], city_idx_all[global_train_idx], y_raw_all[global_train_idx], city_name_all[global_train_idx]
    )
    val_ds_g = AirDataset(
        X_all[global_val_idx], y_all[global_val_idx], city_idx_all[global_val_idx], y_raw_all[global_val_idx], city_name_all[global_val_idx]
    )
    test_ds_g = AirDataset(
        X_all[global_test_idx], y_all[global_test_idx], city_idx_all[global_test_idx], y_raw_all[global_test_idx], city_name_all[global_test_idx]
    )

    train_loader_g = DataLoader(train_ds_g, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_train)
    val_loader_g = DataLoader(val_ds_g, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_train)
    test_loader_g = DataLoader(test_ds_g, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_eval)

    # 8) Global model training
    global_model = GlobalCNNLSTM(
        input_dim=16,
        hidden=64,
        city_vocab=len(cities),
        city_emb_dim=8,
        lr=LR,
    )

    hist_global = HistoryCallback()
    ckpt_global = ModelCheckpoint(
        dirpath=str(MODELS_DIR),
        filename="global_cnn_lstm_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    es_global = EarlyStopping(monitor="val_loss", mode="min", patience=8)

    trainer_global = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices="auto",
        logger=make_logger("global_cnn_lstm", OUT_DIR),
        callbacks=[hist_global, ckpt_global, es_global],
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    trainer_global.fit(global_model, train_loader_g, val_loader_g)

    best_global_path = ckpt_global.best_model_path if ckpt_global.best_model_path else None
    if best_global_path and os.path.exists(best_global_path):
        global_best = GlobalCNNLSTM.load_from_checkpoint(best_global_path)
    else:
        global_best = global_model

    torch.save(global_best.state_dict(), MODELS_DIR / "global_cnn_lstm.pt")

    # 9) Global evaluation per city test splits
    global_city_metrics = {}
    global_preds_for_scatter = {}

    for city in cities:
        city_test_idx = win_df.index[(win_df["city"] == city) & (win_df["split_local"] == "test")].tolist()
        if not city_test_idx:
            continue
        ds = AirDataset(
            X_all[city_test_idx], y_all[city_test_idx], city_idx_all[city_test_idx], y_raw_all[city_test_idx], city_name_all[city_test_idx]
        )
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_eval)
        yt, yp, _ = predict_module(global_best, dl, city_stats, city_id_to_name, device)
        global_city_metrics[city] = metrics_dict(yt, yp)
        global_preds_for_scatter[city] = (yt, yp)

    # explicit LA zero-shot evaluation
    if test_city not in global_city_metrics:
        yt, yp, _ = predict_module(global_best, test_loader_g, city_stats, city_id_to_name, device)
        global_city_metrics[test_city] = metrics_dict(yt, yp)
        global_preds_for_scatter[test_city] = (yt, yp)

    # 10) Local models (per city)
    local_city_metrics = {}
    local_histories = {}

    for city in cities:
        c_train = win_df.index[(win_df["city"] == city) & (win_df["split_local"] == "train")].tolist()
        c_val = win_df.index[(win_df["city"] == city) & (win_df["split_local"] == "val")].tolist()
        c_test = win_df.index[(win_df["city"] == city) & (win_df["split_local"] == "test")].tolist()

        if min(len(c_train), len(c_val), len(c_test)) == 0:
            warnings.warn(f"Skipping local {city}: insufficient windows")
            continue

        tr_ds = AirDataset(X_all[c_train], y_all[c_train], city_idx_all[c_train], y_raw_all[c_train], city_name_all[c_train])
        va_ds = AirDataset(X_all[c_val], y_all[c_val], city_idx_all[c_val], y_raw_all[c_val], city_name_all[c_val])
        te_ds = AirDataset(X_all[c_test], y_all[c_test], city_idx_all[c_test], y_raw_all[c_test], city_name_all[c_test])

        tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_train)
        va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_train)
        te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_eval)

        local_model = LocalLSTM(input_dim=16, hidden=64, lr=LR)
        hist_local = HistoryCallback()

        ckpt_local = ModelCheckpoint(
            dirpath=str(MODELS_DIR),
            filename=f"local_lstm_{sanitize_city_name(city)}_best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        es_local = EarlyStopping(monitor="val_loss", mode="min", patience=8)

        trainer_local = pl.Trainer(
            max_epochs=EPOCHS,
            accelerator="auto",
            devices="auto",
            logger=make_logger(f"local_lstm_{sanitize_city_name(city)}", OUT_DIR),
            callbacks=[hist_local, ckpt_local, es_local],
            deterministic=True,
            enable_progress_bar=False,
            log_every_n_steps=10,
        )
        trainer_local.fit(local_model, tr_dl, va_dl)

        best_local_path = ckpt_local.best_model_path if ckpt_local.best_model_path else None
        if best_local_path and os.path.exists(best_local_path):
            local_best = LocalLSTM.load_from_checkpoint(best_local_path)
        else:
            local_best = local_model

        torch.save(local_best.state_dict(), MODELS_DIR / f"local_lstm_{sanitize_city_name(city)}.pt")

        yt, yp, _ = predict_module(local_best, te_dl, city_stats, city_id_to_name, device)
        local_city_metrics[city] = metrics_dict(yt, yp)
        local_histories[city] = dict(hist_local.history)

    # 11) Comparison table
    rows = []
    for city in sorted(set(list(global_city_metrics.keys()) + list(local_city_metrics.keys()))):
        g = global_city_metrics.get(city, {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan})
        l = local_city_metrics.get(city, {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan})

        imp_pct = np.nan
        if np.isfinite(l["MAE"]) and l["MAE"] > 1e-8 and np.isfinite(g["MAE"]):
            imp_pct = 100.0 * (l["MAE"] - g["MAE"]) / l["MAE"]

        rows.append(
            {
                "City": city,
                "Local LSTM MAE": l["MAE"],
                "Global CNN-LSTM MAE": g["MAE"],
                "Improvement": imp_pct,
                "Local LSTM RMSE": l["RMSE"],
                "Global CNN-LSTM RMSE": g["RMSE"],
                "Local LSTM R2": l["R2"],
                "Global CNN-LSTM R2": g["R2"],
                "Winner": "⭐ WIN" if (np.isfinite(imp_pct) and imp_pct > 0) else "Local edge",
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values("City")
    comparison_df.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)

    # 12) Plots
    # loss_curves.png (4 plots)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    gh = hist_global.history
    if gh.get("train_loss"):
        ax1.plot(gh["train_loss"], label="train_loss")
    if gh.get("val_loss"):
        ax1.plot(gh["val_loss"], label="val_loss")
    ax1.set_title("Global CNN-LSTM Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    if gh.get("train_mae"):
        ax2.plot(gh["train_mae"], label="train_mae")
    if gh.get("val_mae"):
        ax2.plot(gh["val_mae"], label="val_mae")
    ax2.set_title("Global CNN-LSTM MAE")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE")
    ax2.legend()

    max_e = 0
    for _, h in local_histories.items():
        max_e = max(max_e, len(h.get("train_loss", [])), len(h.get("val_loss", [])))

    if max_e > 0:
        tr_stack, va_stack = [], []
        for _, h in local_histories.items():
            tl = h.get("train_loss", [])
            vl = h.get("val_loss", [])
            if tl:
                a = np.full(max_e, np.nan)
                a[: len(tl)] = tl
                tr_stack.append(a)
            if vl:
                a = np.full(max_e, np.nan)
                a[: len(vl)] = vl
                va_stack.append(a)
        if tr_stack:
            ax3.plot(np.nanmean(np.vstack(tr_stack), axis=0), label="mean_train_loss")
        if va_stack:
            ax3.plot(np.nanmean(np.vstack(va_stack), axis=0), label="mean_val_loss")
    ax3.set_title("Local LSTM Mean Loss (Across Cities)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.legend()

    if max_e > 0:
        tr_stack, va_stack = [], []
        for _, h in local_histories.items():
            tm = h.get("train_mae", [])
            vm = h.get("val_mae", [])
            if tm:
                a = np.full(max_e, np.nan)
                a[: len(tm)] = tm
                tr_stack.append(a)
            if vm:
                a = np.full(max_e, np.nan)
                a[: len(vm)] = vm
                va_stack.append(a)
        if tr_stack:
            ax4.plot(np.nanmean(np.vstack(tr_stack), axis=0), label="mean_train_mae")
        if va_stack:
            ax4.plot(np.nanmean(np.vstack(va_stack), axis=0), label="mean_val_mae")
    ax4.set_title("Local LSTM Mean MAE (Across Cities)")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("MAE")
    ax4.legend()

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "loss_curves.png", dpi=200)
    plt.close(fig)

    # pred_actual_scatter.png (zero-shot city + local if available)
    fig, ax = plt.subplots(figsize=(8, 7))
    if test_city in global_preds_for_scatter:
        yt, yp = global_preds_for_scatter[test_city]
        ax.scatter(yt, yp, s=12, alpha=0.6, label=f"Global ({test_city})")

    local_model_path = MODELS_DIR / f"local_lstm_{sanitize_city_name(test_city)}.pt"
    c_test = win_df.index[(win_df["city"] == test_city) & (win_df["split_local"] == "test")].tolist()
    if c_test and local_model_path.exists():
        ds = AirDataset(X_all[c_test], y_all[c_test], city_idx_all[c_test], y_raw_all[c_test], city_name_all[c_test])
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_eval)
        lm = LocalLSTM(input_dim=16, hidden=64, lr=LR)
        lm.load_state_dict(torch.load(local_model_path, map_location="cpu"))
        yt2, yp2, _ = predict_module(lm, dl, city_stats, city_id_to_name, device)
        ax.scatter(yt2, yp2, s=12, alpha=0.5, label=f"Local ({test_city})")

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

    # 13) paper_results.md
    md_lines = []
    md_lines.append("# Global vs Local Deep Learning for Multi-City PM2.5 Forecasting")
    md_lines.append("")
    md_lines.append("## Experimental Setup")
    md_lines.append(f"- Seed: {SEED}")
    md_lines.append(f"- Device: {device}")
    md_lines.append(f"- Pollutants: {', '.join(POLLUTANTS)} (pm25 target)")
    md_lines.append(f"- Window/Horizon: {LOOKBACK}h / +{HORIZON}h")
    md_lines.append(f"- Global split: Train={train_cities}, Val={val_city}, Test={test_city} (zero-shot)")
    md_lines.append("")
    md_lines.append("## Comparison Table")
    md_lines.append(comparison_df.round(4).to_markdown(index=False))
    md_lines.append("")

    if test_city in comparison_df["City"].values:
        row = comparison_df[comparison_df["City"] == test_city].iloc[0]
        md_lines.append("## Zero-shot Finding")
        md_lines.append(
            f"- {test_city}: Local MAE={row['Local LSTM MAE']:.3f}, Global MAE={row['Global CNN-LSTM MAE']:.3f}, Improvement={row['Improvement']:.2f}%"
        )
        md_lines.append("")

    md_lines.append("## Generated Files")
    md_lines.append("- output/processed_dataset.pt")
    md_lines.append("- output/models/global_cnn_lstm.pt")
    md_lines.append("- output/results/comparison_table.csv")
    md_lines.append("- output/results/loss_curves.png")
    md_lines.append("- output/results/pred_actual_scatter.png")
    md_lines.append("- output/results/paper_results.md")

    paper_md = "\n".join(md_lines)
    (RESULTS_DIR / "paper_results.md").write_text(paper_md)
    (OUT_DIR / "paper_results.md").write_text(paper_md)

    print(f"Saved: {RESULTS_DIR / 'comparison_table.csv'}")
    print(f"Saved: {RESULTS_DIR / 'loss_curves.png'}")
    print(f"Saved: {RESULTS_DIR / 'pred_actual_scatter.png'}")
    print(f"Saved: {RESULTS_DIR / 'paper_results.md'}")
    print(f"Saved: {OUT_DIR / 'paper_results.md'}")
    print(f"Saved: {MODELS_DIR / 'global_cnn_lstm.pt'}")
    print(f"Saved: {OUT_DIR / 'processed_dataset.pt'}")
    print("PIPELINE COMPLETE - PAPER READY")


if __name__ == "__main__":
    main()
