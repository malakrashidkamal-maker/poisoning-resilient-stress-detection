# -*- coding: utf-8 -*-
"""

@author: Rashid
"""

# -*- coding: utf-8 -*-
"""
#
NOTE:
This script requires DEAP &WESAD dataset.
Download from:
http://www.eecs.qmul.ac.uk/mmv/datasets/deap/
https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection 

Place files (s01.dat ... s32.dat) in --data_root folder.

python deap_pipeline.py --data_root /path/to/deap --outdir outputs

"""

# deap_pipeline.py
import os, json, pickle, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import decimate
from ctgan import CTGAN

# ========= PyTorch (GPU) =========
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

# =======================
# CONFIG
# =======================
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--outdir", type=str, default="outputs")
args = parser.parse_args()

DEAP_ROOT = args.data_root
OUTDIR = args.outdir

os.makedirs(OUTDIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = torch.cuda.is_available()
print(f"[GPU] Using device: {DEVICE} | AMP: {USE_AMP}")
CTGAN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Windowing / sampling
SR_RAW = 128   # DEAP
SR_TGT = 4
DECIM = SR_RAW // SR_TGT       # 32
WINDOW_SIZE_S = 2
OVERLAP = 0.80

# Label binarization: arousal >= 6 -> 1 (activated), else 0
AROUSAL_THRESH = 6

# Peripheral channel indices in DEAP after 32 EEG channels:
# data per trial = [40 channels, samples], 0..31 EEG, 32.. peripherals
PERIPH_IDX = {
    "GSR": 32,    # EDA/GSR
    "RESP": 33,   # Respiration
    "BVP": 34,    # Plethysmography
    "TEMP": 35    # Temperature
}

# Poison sweep / attacks / defenses
POISON_GRID = [0.10, 0.20, 0.30, 0.40]
LABEL_FLIP_FRAC = 0.70
NOISE_STD = 0.05
USE_GAN_ATTACK = False
GAN_ATTACK_RATIO = 0.50
OUTLIER_CONTAM = 0.05
CTGAN_EPOCHS_LIGHT = 300
CTGAN_EPOCHS_HEAVY = 500

RUN_LOSO = False
RUN_SHAP = False

# =======================
# DEAP LOADER + FEATURES
# =======================
def load_deap_subject(path):
    """Return (data, labels) from a DEAP python-processed .dat file."""
    with open(path, 'rb') as f:
        obj = pickle.load(f, encoding='latin1')
    data = obj['data']      # shape: (40 trials, 40 channels, ~8064 samples)
    labels = obj['labels']  # shape: (40, 4) [valence, arousal, dominance, liking]
    return data, labels

def downsample_to_4hz(x):
    return decimate(x.astype(np.float64), DECIM, ftype='iir', zero_phase=True)

def deap_trial_to_windows(trial_arr, arousal_label):
    """
    trial_arr: [channels, samples] at 128 Hz.
    Outputs features per 2s window @ 4 Hz: mean/std per peripheral channel.
    """
    # pick and downsample peripheral channels
    series = []
    for name, idx in PERIPH_IDX.items():
        if idx < trial_arr.shape[0]:
            series.append(downsample_to_4hz(trial_arr[idx]))
    if not series:
        return None, None

    L = min(len(s) for s in series)
    series = [s[:L] for s in series]

    win_len = int(WINDOW_SIZE_S * SR_TGT)
    step = max(1, int(win_len * (1 - OVERLAP)))

    feats, ys = [], []
    for start in range(0, L - win_len, step):
        end = start + win_len
        fv = []
        for s in series:
            seg = s[start:end]
            fv.extend([np.mean(seg), np.std(seg)])
        feats.append(fv)
        ys.append(1 if arousal_label >= AROUSAL_THRESH else 0)

    if not feats:
        return None, None
    return np.array(feats, dtype=np.float64), np.array(ys, dtype=int)

def build_deap_dataset(deap_root=DEAP_ROOT, subject_ids=None):
    """
    Build (X, y, groups) for DEAP from python-processed .dat files.
    groups = subject id (1..32).
    """
    if subject_ids is None:
        subject_ids = sorted([
            int(fn[1:3]) for fn in os.listdir(deap_root)
            if fn.lower().startswith('s') and fn.lower().endswith('.dat') and fn[1:3].isdigit()
        ])

    all_X, all_y, all_g = [], [], []
    for sid in subject_ids:
        fpath = os.path.join(deap_root, f"s{sid:02d}.dat")
        if not os.path.isfile(fpath):
            print(f"Missing {fpath}, skip.")
            continue
        data, labels = load_deap_subject(fpath)
        n_trials = data.shape[0]
        subj_X, subj_y = [], []
        for t in range(n_trials):
            trial = data[t]          # [channels, samples]
            arousal = float(labels[t, 1])
            Xw, yw = deap_trial_to_windows(trial, arousal)
            if Xw is None:
                continue
            subj_X.append(Xw); subj_y.append(yw)

        if not subj_X:
            print(f"S{sid:02d}: no valid windows.")
            continue

        Xs = np.vstack(subj_X)
        ys = np.hstack(subj_y)
        gs = np.full(len(ys), sid, dtype=int)

        print(f"DEAP S{sid:02d}: {len(ys)} windows ({ys.sum()} pos, {len(ys)-ys.sum()} neg)")
        all_X.append(Xs); all_y.append(ys); all_g.append(gs)

    if not all_X:
        raise RuntimeError("No DEAP windows extracted. Check PERIPH_IDX or files.")

    X = np.vstack(all_X)
    y = np.hstack(all_y)
    groups = np.hstack(all_g)
    print(f"DEAP dataset: X={X.shape}, pos={y.sum()}, neg={len(y)-y.sum()}")
    return X, y, groups

# =======================
# SPLIT / BALANCE
# =======================
def grouped_split(X, y, groups, test_size=0.2, rs=RANDOM_STATE):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
    train_idx, test_idx = next(gss.split(X, y, groups))
    return train_idx, test_idx

def fit_scale_split(X, y, groups):
    train_idx, test_idx = grouped_split(X, y, groups)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test  = scaler.transform(X[test_idx])
    return X_train, y[train_idx], X_test, y[test_idx], scaler, train_idx, test_idx

def undersample_train(X_train, y_train):
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    return rus.fit_resample(X_train, y_train)

def save_balance_plot(y_before, y_after, title, path):
    fig, axes = plt.subplots(1,2, figsize=(6,3), dpi=300)
    sns.countplot(x=y_before.astype(int), ax=axes[0])
    axes[0].set_title("Train before balance"); axes[0].set_xlabel("Class"); axes[0].set_ylabel("Count")
    sns.countplot(x=y_after.astype(int),  ax=axes[1])
    axes[1].set_title("Train after balance"); axes[1].set_xlabel("Class"); axes[1].set_ylabel("Count")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)

# =======================
# Model: FG-DualNet
# =======================
class FG_DualNetTorch(nn.Module):
    def __init__(self, input_dim, p_drop1=0.4, p_drop2=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.feature_gate = nn.Linear(input_dim, input_dim, bias=True)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        self.drop1 = nn.Dropout(p_drop1)
        self.drop2 = nn.Dropout(p_drop2)

    def forward(self, x):
        gate = torch.sigmoid(self.feature_gate(x))    # (B, D)
        x_g = x * gate
        x = F.relu(self.fc1(x_g))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        logits = self.out(x).squeeze(-1)
        return logits, gate

def train_FG_dualnet_torch(X_train, y_train, input_dim, epochs=12, batch_size=256, val_split=0.2, lr=1e-3):
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    ds = TensorDataset(X_t, y_t)
    n_val = int(len(ds)*val_split)
    n_tr = len(ds) - n_val
    ds_tr, ds_val = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(RANDOM_STATE))
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    model = FG_DualNetTorch(input_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=USE_AMP)
    bce = nn.BCEWithLogitsLoss()

    t0 = time.time()
    model.train()
    for ep in range(epochs):
        for xb, yb in dl_tr:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                logits, _ = model(xb)
                loss = bce(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        # (optional) quick validation pass
        if n_val > 0:
            model.eval()
            with torch.no_grad(), autocast(enabled=USE_AMP):
                _ = sum(bce(model(xb.to(DEVICE))[0], yb.to(DEVICE)).item()*len(xb) for xb, yb in dl_val)
            model.train()
    train_time = time.time() - t0
    return model, train_time

@torch.no_grad()
def predict_proba_FG(model, X):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    out = []
    bs = 4096
    for i in range(0, len(X_t), bs):
        xb = X_t[i:i+bs]
        with autocast(enabled=USE_AMP):
            logits, _ = model(xb)
            p = torch.sigmoid(logits)
        out.append(p.detach().cpu().numpy())
    return np.concatenate(out, axis=0)

@torch.no_grad()
def get_feature_gates_FG(model, X):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    out = []
    bs = 4096
    for i in range(0, len(X_t), bs):
        xb = X_t[i:i+bs]
        with autocast(enabled=USE_AMP):
            _, gates = model(xb)
        out.append(gates.detach().cpu().numpy())
    return np.concatenate(out, axis=0)

# =======================
# CTGAN helpers (GPU aware)
# =======================
def _validate_df_for_gan(X_array, min_rows=60, name="(unknown)"):
    if not isinstance(X_array, np.ndarray):
        X_array = np.asarray(X_array)
    if X_array.size == 0:
        print(f"[CTGAN:{name}] Empty array."); return None
    good = np.isfinite(X_array).all(axis=1)
    if not good.all():
        X_array = X_array[good]
    if X_array.shape[0] < min_rows:
        print(f"[CTGAN:{name}] Too few rows after cleaning: {X_array.shape[0]} (need >= {min_rows}).")
        return None
    X_array = X_array.astype(np.float64, copy=False)
    return pd.DataFrame(X_array, columns=[f"f{i+1}" for i in range(X_array.shape[1])])

def _ctgan_init(epochs, batch_size):
    try:
        return CTGAN(
            epochs=epochs, batch_size=batch_size,
            generator_dim=(128,128), discriminator_dim=(128,128),
            verbose=False, device=CTGAN_DEVICE
        )
    except TypeError:
        print("[CTGAN] 'device' kwarg unsupported; using default backend.")
        return CTGAN(
            epochs=epochs, batch_size=batch_size,
            generator_dim=(128,128), discriminator_dim=(128,128),
            verbose=False
        )

# --- Attacks / defenses ---
def poison_training_data_label_noise(X_train, y_train, frac, flip_frac=0.7, noise_std=0.5):
    rng = np.random.RandomState(RANDOM_STATE)
    Xp, yp = X_train.copy(), y_train.copy()
    n = len(y_train); k = int(frac * n)
    if k <= 0: return Xp, yp
    idx = rng.choice(np.arange(n), size=k, replace=False)
    k_flip = int(flip_frac * k); idx_flip = idx[:k_flip]
    yp[idx_flip] = 1 - yp[idx_flip]
    Xp[idx] = Xp[idx] + rng.normal(0.0, noise_std, size=Xp[idx].shape)
    return Xp, yp

def gan_attack_injection(X_train, y_train, attack_ratio=0.5):
    mask = (y_train == 1); X_pos = X_train[mask]
    df_pos = _validate_df_for_gan(X_pos, min_rows=60, name="attack")
    if df_pos is None:
        print("[CTGAN:attack] Validation failed; skipping GAN attack.")
        return X_train, y_train
    try:
        ctgan_att = _ctgan_init(epochs=200, batch_size=256)
        ctgan_att.fit(df_pos, discrete_columns=[])
        gen_n = max(1, int(attack_ratio * X_pos.shape[0]))
        synth = ctgan_att.sample(gen_n).values.astype(np.float64)
        X_att = np.vstack([X_train, synth]); y_att = np.hstack([y_train, np.ones(gen_n, dtype=int)])
        return X_att, y_att
    except AssertionError:
        print("[CTGAN:attack] AssertionError; skip injection.")
        return X_train, y_train

def filter_outliers(X, y, contamination=0.05):
    iso = IsolationForest(contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1)
    m = iso.fit_predict(X)  # 1 inlier, -1 outlier
    keep = (m == 1)
    return X[keep], y[keep]

def ctgan_augment_class(X_train, y_train, target_class=1, epochs=300, batch_size=256, gen_samples=None):
    n_pos = int((y_train == 1).sum()); n_neg = int((y_train == 0).sum())
    if gen_samples is None:
        gen_samples = max(0, (n_neg - n_pos) if target_class == 1 else (n_pos - n_neg))
    if gen_samples == 0: return X_train, y_train
    X_tar = X_train[y_train == target_class]
    df_tar = _validate_df_for_gan(X_tar, min_rows=60, name="defense")
    if df_tar is None:
        print("[CTGAN:defense] Falling back to SMOTE.")
        try:
            return SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
        except Exception:
            return X_train, y_train
    try:
        ctgan = _ctgan_init(epochs=epochs, batch_size=batch_size)
        ctgan.fit(df_tar, discrete_columns=[])
        synth = ctgan.sample(gen_samples).values.astype(np.float64)
        X_aug = np.vstack([X_train, synth]); y_aug = np.hstack([y_train, np.full(gen_samples, target_class, dtype=int)])
        return X_aug, y_aug
    except AssertionError:
        print("[CTGAN:defense] AssertionError; SMOTE fallback.")
        try:
            return SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
        except Exception:
            return X_train, y_train

# =======================
# PLOTS (300 dpi)
# =======================
def save_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,3), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False, ax=ax)
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)

def save_roc_curve(y_true, y_proba, title, path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc_val = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4,3), dpi=300)
    ax.plot(fpr, tpr, label=f"AUC={roc_auc_val:.3f}")
    ax.plot([0,1], [0,1], linestyle="--")
    ax.set_title(title); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    return roc_auc_val

def save_pr_curve(y_true, y_proba, title, path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(4,3), dpi=300)
    ax.plot(recall, precision, label=f"AP={ap:.3f}")
    ax.set_title(title); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    return ap

# =======================
# JOURNAL (MultiIndex, transactions-style)
# =======================
JOURNAL_ROWS = []
JOURNAL_COLS = pd.MultiIndex.from_tuples(
    [
        ("Run", "Date"),
        ("Run", "Experiment_ID"),
        ("Data", "Dataset"),
        ("Data", "Subjects"),
        ("Split", "Train_Size"),
        ("Split", "Test_Size"),
        ("Split", "Scaler"),
        ("Poison", "Poison_Fraction"),
        ("Poison", "Label_Flip_Frac"),
        ("Poison", "Noise_STD"),
        ("Defense", "Outlier_Filter"),
        ("Defense", "Contamination"),
        ("Defense", "Augmentor"),
        ("Defense", "Augmentor_Epochs"),
        ("Balance", "Method"),
        ("Model", "Name"),
        ("Model", "Key_Params"),
        ("Training", "Epochs/Estimators"),
        ("Training", "Batch/Depth"),
        ("Training", "Random_State"),
        ("Timing", "Train_Time_s"),
        ("Timing", "Infer_Time_ms_per_sample"),
        ("Metrics", "Accuracy"),
        ("Metrics", "F1_Macro"),
        ("Metrics", "F1_0"),
        ("Metrics", "F1_1"),
        ("Metrics", "ROC_AUC"),
        ("Metrics", "PR_AUC"),
        ("Artifacts", "CM_Image"),
        ("Artifacts", "ROC_Image"),
        ("Artifacts", "PR_Image"),
        ("Artifacts", "Report_CSV"),
        ("Notes", "Freeform"),
    ],
    names=["Section", "Field"]
)

def _append_journal_row(stage, model_name, frac, params, sizes, metrics, timings, artifacts, defense_info, balance_method, subjects_str, augmentor_epochs):
    JOURNAL_ROWS.append([
        pd.Timestamp.utcnow().date().isoformat(),
        f"{stage}-{model_name}-{int(frac*100):02d}",
        "DEAP",
        subjects_str,
        sizes.get("train", np.nan),
        sizes.get("test", np.nan),
        "StandardScaler",
        frac,
        LABEL_FLIP_FRAC,
        NOISE_STD,
        defense_info.get("outlier", "None"),
        defense_info.get("contam", np.nan),
        defense_info.get("augmentor", "None"),
        augmentor_epochs if defense_info.get("augmentor","None").startswith("CTGAN") else np.nan,
        balance_method,
        model_name,
        params,
        timings.get("epochs_or_estimators", np.nan),
        timings.get("batch_or_depth", "-"),
        RANDOM_STATE,
        round(timings.get("train_s", np.nan), 6) if timings.get("train_s") is not None else np.nan,
        round(timings.get("infer_ms_per_sample", np.nan), 6) if timings.get("infer_ms_per_sample") is not None else np.nan,
        metrics.get("acc", np.nan),
        metrics.get("f1_macro", np.nan),
        metrics.get("f1_0", np.nan),
        metrics.get("f1_1", np.nan),
        metrics.get("roc_auc", np.nan),
        metrics.get("pr_auc", np.nan),
        artifacts.get("cm", ""),
        artifacts.get("roc", ""),
        artifacts.get("pr", ""),
        artifacts.get("report_csv", ""),
        ""
    ])

def _save_journal(outdir):
    df = pd.DataFrame(JOURNAL_ROWS, columns=JOURNAL_COLS)
    df.to_csv(os.path.join(outdir, "experiment_transactions_journal_deap.csv"))
    # Excel + MultiIndex: keep index=True (default)
    df.to_excel(os.path.join(outdir, "experiment_transactions_journal_deap.xlsx"))
    return df

# =======================
# REPORTING WRAPPER
# =======================
def report_and_save(stage, model_name, y_true, y_pred, y_proba=None):
    rep = classification_report(y_true, y_pred, output_dict=True)
    print(f"\n[{stage}] {model_name}\n", classification_report(y_true, y_pred))
    df_rep = pd.DataFrame(rep).transpose()
    df_rep.to_csv(os.path.join(OUTDIR, f"report_{stage}_{model_name}.csv"))
    with open(os.path.join(OUTDIR, f"report_{stage}_{model_name}.json"), "w") as f:
        json.dump(rep, f, indent=2)

    # Confusion matrix
    save_confusion_matrix(y_true, y_pred, f"{stage} - {model_name}",
                          os.path.join(OUTDIR, f"cm_{stage}_{model_name}.png"))

    roc_auc_val, pr_auc_val = np.nan, np.nan
    if y_proba is not None:
        roc_auc_val = save_roc_curve(y_true, y_proba, f"{stage} - {model_name} ROC",
                                     os.path.join(OUTDIR, f"roc_{stage}_{model_name}.png"))
        pr_auc_val = save_pr_curve(y_true, y_proba, f"{stage} - {model_name} PR",
                                   os.path.join(OUTDIR, f"pr_{stage}_{model_name}.png"))
    return rep, roc_auc_val, pr_auc_val

def _infer_time_ms_per_sample(predict_fn, X):
    t0 = time.time()
    _ = predict_fn(X)
    t1 = time.time()
    return (t1 - t0) * 1000.0 / max(1, len(X))

# =======================
# BASELINES (with journaling)
# =======================
def run_baselines(stage, X_train, y_train, X_test, y_test, frac, subjects_str, balance_method, augmentor_epochs):
    results = {}
    sizes = {"train": int(len(y_train)), "test": int(len(y_test))}
    defense_info = {"outlier": "None", "contam": np.nan, "augmentor": "None"}

    # LR
    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    train_s = time.time() - t0
    ypr = lr.predict(X_test); yproba = lr.predict_proba(X_test)[:,1]
    rep, roc_auc_val, pr_auc_val = report_and_save(stage, "LR", y_test, ypr, y_proba=yproba)
    results['LR'] = rep
    _append_journal_row(stage, "LR", frac, "max_iter=1000", sizes,
        metrics={"acc": rep["accuracy"], "f1_macro": rep["macro avg"]["f1-score"], "f1_0": rep["0"]["f1-score"], "f1_1": rep["1"]["f1-score"], "roc_auc": roc_auc_val, "pr_auc": pr_auc_val},
        timings={"train_s": train_s, "infer_ms_per_sample": _infer_time_ms_per_sample(lr.predict, X_test), "epochs_or_estimators": 1, "batch_or_depth": "-"},
        artifacts={"cm": os.path.join(OUTDIR, f"cm_{stage}_LR.png"), "roc": os.path.join(OUTDIR, f"roc_{stage}_LR.png"), "pr": os.path.join(OUTDIR, f"pr_{stage}_LR.png"), "report_csv": os.path.join(OUTDIR, f"report_{stage}_LR.csv")},
        defense_info=defense_info, balance_method=balance_method, subjects_str=subjects_str, augmentor_epochs=augmentor_epochs
    )

    # RF
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    train_s = time.time() - t0
    ypr = rf.predict(X_test); yproba = rf.predict_proba(X_test)[:,1]
    rep, roc_auc_val, pr_auc_val = report_and_save(stage, "RF", y_test, ypr, y_proba=yproba)
    results['RF'] = rep
    _append_journal_row(stage, "RF", frac, "n_estimators=400", sizes,
        metrics={"acc": rep["accuracy"], "f1_macro": rep["macro avg"]["f1-score"], "f1_0": rep["0"]["f1-score"], "f1_1": rep["1"]["f1-score"], "roc_auc": roc_auc_val, "pr_auc": pr_auc_val},
        timings={"train_s": train_s, "infer_ms_per_sample": _infer_time_ms_per_sample(rf.predict, X_test), "epochs_or_estimators": 400, "batch_or_depth": "-"},
        artifacts={"cm": os.path.join(OUTDIR, f"cm_{stage}_RF.png"), "roc": os.path.join(OUTDIR, f"roc_{stage}_RF.png"), "pr": os.path.join(OUTDIR, f"pr_{stage}_RF.png"), "report_csv": os.path.join(OUTDIR, f"report_{stage}_RF.csv")},
        defense_info=defense_info, balance_method=balance_method, subjects_str=subjects_str, augmentor_epochs=augmentor_epochs
    )

    # SVM
    t0 = time.time()
    svm = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
    svm.fit(X_train, y_train)
    train_s = time.time() - t0
    ypr = svm.predict(X_test); yproba = svm.predict_proba(X_test)[:,1]
    rep, roc_auc_val, pr_auc_val = report_and_save(stage, "SVM", y_test, ypr, y_proba=yproba)
    results['SVM'] = rep
    _append_journal_row(stage, "SVM", frac, "rbf, prob=True", sizes,
        metrics={"acc": rep["accuracy"], "f1_macro": rep["macro avg"]["f1-score"], "f1_0": rep["0"]["f1-score"], "f1_1": rep["1"]["f1-score"], "roc_auc": roc_auc_val, "pr_auc": pr_auc_val},
        timings={"train_s": train_s, "infer_ms_per_sample": _infer_time_ms_per_sample(svm.predict, X_test), "epochs_or_estimators": 1, "batch_or_depth": "-"},
        artifacts={"cm": os.path.join(OUTDIR, f"cm_{stage}_SVM.png"), "roc": os.path.join(OUTDIR, f"roc_{stage}_SVM.png"), "pr": os.path.join(OUTDIR, f"pr_{stage}_SVM.png"), "report_csv": os.path.join(OUTDIR, f"report_{stage}_SVM.csv")},
        defense_info=defense_info, balance_method=balance_method, subjects_str=subjects_str, augmentor_epochs=augmentor_epochs
    )

    # XGB (GPU if available)
    if HAVE_XGB:
        t0 = time.time()
        xgb = XGBClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1,
            tree_method = "gpu_hist" if DEVICE == "cuda" else "hist"
            predictor = "gpu_predictor" if DEVICE == "cuda" else "auto"
        )
        xgb.fit(X_train, y_train)
        train_s = time.time() - t0
        ypr = xgb.predict(X_test); yproba = xgb.predict_proba(X_test)[:,1]
        rep, roc_auc_val, pr_auc_val = report_and_save(stage, "XGB", y_test, ypr, y_proba=yproba)
        results['XGB'] = rep
        _append_journal_row(stage, "XGB", frac, "gpu_hist, lr=0.05, depth=6", sizes,
            metrics={"acc": rep["accuracy"], "f1_macro": rep["macro avg"]["f1-score"], "f1_0": rep["0"]["f1-score"], "f1_1": rep["1"]["f1-score"], "roc_auc": roc_auc_val, "pr_auc": pr_auc_val},
            timings={"train_s": train_s, "infer_ms_per_sample": _infer_time_ms_per_sample(xgb.predict, X_test), "epochs_or_estimators": 600, "batch_or_depth": "max_depth=6"},
            artifacts={"cm": os.path.join(OUTDIR, f"cm_{stage}_XGB.png"), "roc": os.path.join(OUTDIR, f"roc_{stage}_XGB.png"), "pr": os.path.join(OUTDIR, f"pr_{stage}_XGB.png"), "report_csv": os.path.join(OUTDIR, f"report_{stage}_XGB.csv")},
            defense_info=defense_info, balance_method=balance_method, subjects_str=subjects_str, augmentor_epochs=augmentor_epochs
        )

    return results

# =======================
# FG-DualNet 
# =======================
def run_FG_dualnet(stage, X_train, y_train, X_test, y_test, input_dim, epochs=12, batch=256,
                   frac=0.0, subjects_str="", balance_method="Undersample", augmentor_epochs=np.nan, save_gates=True):
    model, train_time = train_FG_dualnet_torch(X_train, y_train, input_dim, epochs=epochs, batch_size=batch)
    # Predict
    t0 = time.time()
    yproba = predict_proba_FG(model, X_test)
    infer_ms = (time.time() - t0) * 1000.0 / max(1, len(X_test))
    ypred = (yproba > 0.5).astype(int)

    rep, roc_auc_val, pr_auc_val = report_and_save(stage, "FG-DualNet", y_test, ypred, y_proba=yproba)

    if save_gates:
        gates = get_feature_gates_FG(model, X_test)
        np.save(os.path.join(OUTDIR, f"gates_{stage}.npy"), gates)
        fig, ax = plt.subplots(figsize=(6,3), dpi=300)
        ax.bar([f"f{i+1}" for i in range(input_dim)], gates.mean(axis=0))
        ax.set_title(f"{stage} - Feature Attention (mean)")
        ax.set_ylim(0,1); plt.xticks(rotation=30, ha='right')
        fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, f"gate_importance_{stage}.png"), dpi=300); plt.close(fig)

    _append_journal_row(
        stage, "FG-DualNet", frac,
        "adam, gate",
        {"train": int(len(y_train)), "test": int(len(y_test))},
        metrics={"acc": rep["accuracy"], "f1_macro": rep["macro avg"]["f1-score"], "f1_0": rep["0"]["f1-score"], "f1_1": rep["1"]["f1-score"], "roc_auc": roc_auc_val, "pr_auc": pr_auc_val},
        timings={"train_s": train_time, "infer_ms_per_sample": infer_ms, "epochs_or_estimators": epochs, "batch_or_depth": f"batch={batch}"},
        artifacts={"cm": os.path.join(OUTDIR, f"cm_{stage}_FG-DualNet.png"), "roc": os.path.join(OUTDIR, f"roc_{stage}_FG-DualNet.png"), "pr": os.path.join(OUTDIR, f"pr_{stage}_FG-DualNet.png"), "report_csv": os.path.join(OUTDIR, f"report_{stage}_FG-DualNet.csv")},
        defense_info={"outlier": "None", "contam": np.nan, "augmentor": "None"},
        balance_method=balance_method, subjects_str=subjects_str, augmentor_epochs=augmentor_epochs
    )
    return rep

# =======================
# SWEEP RUNNER 
# =======================
def run_poison_and_defenses_at_fraction(frac, X_train, y_train, X_test, y_test, input_dim, subjects_str):
    tag = f"f{int(frac*100)}"

    # Poison base
    Xp, yp = poison_training_data_label_noise(X_train, y_train, frac, LABEL_FLIP_FRAC, NOISE_STD)
    if USE_GAN_ATTACK:
        Xp, yp = gan_attack_injection(Xp, yp, attack_ratio=GAN_ATTACK_RATIO)
    Xp_bal, yp_bal = undersample_train(Xp, yp)
    save_balance_plot(yp, yp_bal, f"POISON {tag} (undersample)", os.path.join(OUTDIR, f"balance_POISON_{tag}_undersample.png"))

    # Baselines + FG under poison
    poison_results = run_baselines(f"POISON_UNDER_{tag}", Xp_bal, yp_bal, X_test, y_test,
                                   frac=frac, subjects_str=subjects_str, balance_method="Undersample", augmentor_epochs=np.nan)
    FG_poison_rep = run_FG_dualnet(f"POISON_UNDER_{tag}", Xp_bal, yp_bal, X_test, y_test,
                                   input_dim=input_dim, epochs=12, batch=256,
                                   frac=frac, subjects_str=subjects_str, balance_method="Undersample", augmentor_epochs=np.nan)

    # Defense: IsolationForest -> CTGAN
    Xf, yf = filter_outliers(Xp, yp, contamination=OUTLIER_CONTAM)
    epochs = CTGAN_EPOCHS_LIGHT if frac <= 0.2 else CTGAN_EPOCHS_HEAVY
    X_ct, y_ct = ctgan_augment_class(Xf, yf, target_class=1, epochs=epochs, batch_size=256)
    save_balance_plot(yf, y_ct, f"DEFENSE_CTGAN {tag}", os.path.join(OUTDIR, f"balance_DEFENSE_CTGAN_{tag}.png"))

    def _wrap_run_baselines(stage_name, Xtr, ytr, balance_method_local, augmentor_epochs_local, defense_tag):
        before = len(JOURNAL_ROWS)
        out = run_baselines(stage_name, Xtr, ytr, X_test, y_test,
                            frac=frac, subjects_str=subjects_str,
                            balance_method=balance_method_local, augmentor_epochs=augmentor_epochs_local)
        # Patch defense metadata on newly appended journal rows
        for i in range(before, len(JOURNAL_ROWS)):
            JOURNAL_ROWS[i][JOURNAL_COLS.get_loc(("Defense","Outlier_Filter"))] = "IsolationForest"
            JOURNAL_ROWS[i][JOURNAL_COLS.get_loc(("Defense","Contamination"))] = OUTLIER_CONTAM
            JOURNAL_ROWS[i][JOURNAL_COLS.get_loc(("Defense","Augmentor"))] = defense_tag
            JOURNAL_ROWS[i][JOURNAL_COLS.get_loc(("Defense","Augmentor_Epochs"))] = augmentor_epochs_local if "CTGAN" in defense_tag else np.nan
        return out

    def_ct = _wrap_run_baselines(f"DEFENSE_CTGAN_{tag}", X_ct, y_ct, "CTGAN", epochs, "CTGAN(target=stress)")
    FG_def_ct = run_FG_dualnet(f"DEFENSE_CTGAN_{tag}", X_ct, y_ct, X_test, y_test,
                               input_dim=input_dim, epochs=12, batch=256,
                               frac=frac, subjects_str=subjects_str, balance_method="CTGAN", augmentor_epochs=epochs)
    # Patch last FG row defense info
    JOURNAL_ROWS[-1][JOURNAL_COLS.get_loc(("Defense","Outlier_Filter"))] = "IsolationForest"
    JOURNAL_ROWS[-1][JOURNAL_COLS.get_loc(("Defense","Contamination"))] = OUTLIER_CONTAM
    JOURNAL_ROWS[-1][JOURNAL_COLS.get_loc(("Defense","Augmentor"))] = "CTGAN(target=stress)"
    JOURNAL_ROWS[-1][JOURNAL_COLS.get_loc(("Defense","Augmentor_Epochs"))] = epochs

    # Defense: IsolationForest -> SMOTE
    Xf2, yf2 = filter_outliers(Xp, yp, contamination=OUTLIER_CONTAM)
    X_sm, y_sm = SMOTE(random_state=RANDOM_STATE).fit_resample(Xf2, yf2)
    save_balance_plot(yf2, y_sm, f"DEFENSE_SMOTE {tag}", os.path.join(OUTDIR, f"balance_DEFENSE_SMOTE_{tag}.png"))

    def_sm = _wrap_run_baselines(f"DEFENSE_SMOTE_{tag}", X_sm, y_sm, "SMOTE", np.nan, "SMOTE")
    FG_def_sm = run_FG_dualnet(f"DEFENSE_SMOTE_{tag}", X_sm, y_sm, X_test, y_test,
                               input_dim=input_dim, epochs=12, batch=256,
                               frac=frac, subjects_str=subjects_str, balance_method="SMOTE", augmentor_epochs=np.nan)
    JOURNAL_ROWS[-1][JOURNAL_COLS.get_loc(("Defense","Outlier_Filter"))] = "IsolationForest"
    JOURNAL_ROWS[-1][JOURNAL_COLS.get_loc(("Defense","Contamination"))] = OUTLIER_CONTAM
    JOURNAL_ROWS[-1][JOURNAL_COLS.get_loc(("Defense","Augmentor"))] = "SMOTE"
    JOURNAL_ROWS[-1][JOURNAL_COLS.get_loc(("Defense","Augmentor_Epochs"))] = np.nan

    def macro_f1(rep): return rep["macro avg"]["f1-score"]
    return {
        "POISON_RF": macro_f1(poison_results["RF"]),
        "POISON_FG": macro_f1(FG_poison_rep),
        "DEF_CTGAN_RF": macro_f1(def_ct["RF"]),
        "DEF_CTGAN_FG": macro_f1(FG_def_ct),
        "DEF_SMOTE_RF": macro_f1(def_sm["RF"]),
        "DEF_SMOTE_FG": macro_f1(FG_def_sm),
    }

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    # 1) Build DEAP dataset
    X, y, groups = build_deap_dataset(DEAP_ROOT)
    print("Label distribution:", np.bincount(y))

    # Save raw features (for audit/repro)
    feat_df = pd.DataFrame(X, columns=[f"f{i+1}" for i in range(X.shape[1])])
    feat_df["label"] = y; feat_df["subject"] = groups
    feat_df.to_csv(os.path.join(OUTDIR, "deap_features_subjects.csv"), index=False)

    subjects_str = ",".join([f"S{int(s)}" for s in sorted(set(groups))])

    # 2) Grouped split + scale
    X_train, y_train, X_test, y_test, scaler, tr_idx, te_idx = fit_scale_split(X, y, groups)

    # 3) CLEAN baseline (undersample)
    X_train_bal, y_train_bal = undersample_train(X_train, y_train)
    save_balance_plot(y_train, y_train_bal, "CLEAN train balance (undersample)",
                      os.path.join(OUTDIR, "balance_CLEAN_undersample.png"))

    _ = run_baselines("CLEAN_UNDER", X_train_bal, y_train_bal, X_test, y_test,
                      frac=0.0, subjects_str=subjects_str,
                      balance_method="Undersample", augmentor_epochs=np.nan)
    _ = run_FG_dualnet("CLEAN_UNDER", X_train_bal, y_train_bal, X_test, y_test,
                       input_dim=X.shape[1], epochs=12, batch=256,
                       frac=0.0, subjects_str=subjects_str,
                       balance_method="Undersample", augmentor_epochs=np.nan)

    # 4) Poison/Defense sweep
    poison_curve = []
    for frac in POISON_GRID:
        print(f"\n=== DEAP: POISON/DEFENSE at fraction={frac:.2f} (GAN attack={USE_GAN_ATTACK}) ===")
        res = run_poison_and_defenses_at_fraction(frac, X_train, y_train, X_test, y_test, input_dim=X.shape[1], subjects_str=subjects_str)
        res["frac"] = frac; poison_curve.append(res)
    curve_df = pd.DataFrame(poison_curve)
    curve_df.to_csv(os.path.join(OUTDIR, "poison_defense_curve_deap.csv"), index=False)

    # 5) Macro-F1 vs poison fraction plots
    plt.figure(figsize=(7,4), dpi=300)
    for col in ["POISON_RF", "DEF_CTGAN_RF", "DEF_SMOTE_RF"]:
        plt.plot(curve_df["frac"], curve_df[col], marker="o", label=col)
    plt.xlabel("Poison fraction"); plt.ylabel("Macro F1"); plt.ylim(0,1.0)
    plt.title(f"DEAP RF: Macro-F1 vs Poison Fraction (GAN attack={USE_GAN_ATTACK})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "curve_rf_poison_defense_deap.png"), dpi=300); plt.close()

    plt.figure(figsize=(7,4), dpi=300)
    for col in ["POISON_FG", "DEF_CTGAN_FG", "DEF_SMOTE_FG"]:
        plt.plot(curve_df["frac"], curve_df[col], marker="o", label=col)
    plt.xlabel("Poison fraction"); plt.ylabel("Macro F1"); plt.ylim(0,1.0)
    plt.title(f"DEAP FG-DualNet: Macro-F1 vs Poison Fraction (GAN attack={USE_GAN_ATTACK})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "curve_FG_poison_defense_deap.png"), dpi=300); plt.close()

    # 6) Optional LOSO + SHAP
    if RUN_LOSO:
        rows = []
        logo = LeaveOneGroupOut()
        for tr, te in logo.split(X, y, groups):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr]); Xte = scaler.transform(X[te])
            ytr, yte = y[tr], y[te]
            Xtr_bal, ytr_bal = undersample_train(Xtr, ytr)
            lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(Xtr_bal, ytr_bal)
            yp = lr.predict(Xte)
            rep = classification_report(yte, yp, output_dict=True)
            rows.append({"fold_subj": int(np.unique(groups[te])[0]),
                         "acc": rep["accuracy"], "f1_macro": rep["macro avg"]["f1-score"]})
        pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "loso_lr_deap.csv"), index=False)

    # Save transactions journal
    journal_df = _save_journal(OUTDIR)
