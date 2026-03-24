# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Credit Limit Dynamic Adjustment — User-Level Survival Analysis v4       ║
# ║  Based on research memo 20260324                                          ║
# ║  Run each section sequentially in a Jupyter notebook.                     ║
# ║  No files are saved; all output goes to display() / print() / plt.show() ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION A: CONFIGURATION & IMPORTS                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, log_loss
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from IPython.display import display
import xgboost as xgb
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ── Plot defaults ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.dpi": 150,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.figsize": (10, 5),
    "legend.fontsize": 8,
})
sns.set_style("whitegrid")

# ── Master Configuration ──────────────────────────────────────────────────
CFG = dict(
    # ── Data paths ────────────────────────────────────────────────────────
    data_path          = "working_data/2wcase_updated.csv",
    data_format        = "csv",
    model_path_v4      = "working_data/model/v4-短期.model",   # 短期 = short-term
    feature_path_v4    = "working_data/model/v4-短期.xlsx",    # feature list for V4
    model_path_v5      = "working_data/model/v5-短期.model",   # 短期 = short-term
    feature_path_v5    = "working_data/model/v5-短期.xlsx",    # feature list for V5

    # ── Column names ──────────────────────────────────────────────────────
    cols = dict(
        user_id    = "user_id",
        event_date = "first_done_date",
        bf_credit  = "first_bf_credit",
        af_credit  = "first_af_credit",
        model_type = "model_type",
        label1     = "mob1_label7",
        label2     = "mob2_label7",
    ),

    # ── Overlap window (empirically determined; V4 & V5 both present) ─────
    overlap_start      = pd.Timestamp("2025-03-20"),
    overlap_end        = pd.Timestamp("2025-06-05"),

    # ── Survival / follow-up ──────────────────────────────────────────────
    max_followup_days  = 60,

    # ── Cross-fitting ─────────────────────────────────────────────────────
    n_folds            = 5,
    seed               = 42,

    # ── XGBoost device; set "cpu" when no GPU available ───────────────────
    xgb_device         = "cuda",

    # ── XGBoost hyperparams (shared across all nuisance learners) ─────────
    xgb_params = dict(
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.6,
        min_child_weight = 20,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        verbosity        = 0,
    ),

    # ── Feature-set switch ────────────────────────────────────────────────
    # True  => feat_union (all V4 ∪ V5 features; XGBoost handles NaN natively)
    # False => feat_inter (intersection only, matches old code)
    use_full_features  = True,

    # ── Task-level switches ───────────────────────────────────────────────
    run_dml            = True,   # DML partialling-out in PfP
    run_extra_inc      = True,   # Task 4 extra-inc propensity

    # ── Significance threshold ────────────────────────────────────────────
    alpha              = 0.05,
)

# Required columns — missing → hard error
REQUIRED_COLS = [
    CFG["cols"]["user_id"],
    CFG["cols"]["event_date"],
    CFG["cols"]["bf_credit"],
    CFG["cols"]["af_credit"],
]

print("Configuration loaded.")
print(f"  Overlap window : {CFG['overlap_start'].date()} → {CFG['overlap_end'].date()}")
print(f"  Max follow-up  : {CFG['max_followup_days']} days")
print(f"  Cross-fit folds: {CFG['n_folds']}")
print(f"  Full features  : {CFG['use_full_features']}")

# ── Utility functions ─────────────────────────────────────────────────────
def stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""

def stata_table(model, title="", note=""):
    print(f"\n{'─'*78}")
    if title: print(f"  {title}")
    print(f"{'─'*78}")
    print(f"  {'Variable':.<28s} {'Coef.':>9} {'SE':>9} {'z':>7} {'p':>8}  {'95% CI':>22}")
    print(f"{'─'*78}")
    ci = model.conf_int()
    for v in model.params.index:
        c  = model.params[v]; se = model.bse[v]
        z  = c / se if se > 0 else 0; p = model.pvalues[v]
        lo, hi = ci.loc[v, 0], ci.loc[v, 1]
        print(f"  {v:.<28s} {c:>9.4f} {se:>9.4f} {z:>7.2f} {p:>8.4f}  [{lo:>8.4f}, {hi:>8.4f}] {stars(p)}")
    print(f"{'─'*78}")
    print(f"  N={int(model.nobs):,}  AIC={model.aic:.1f}")
    if note: print(f"  Note: {note}")
    print("  * p<0.10  ** p<0.05  *** p<0.01")

def kaplan_meier(T, E):
    dk = pd.DataFrame({"T": T, "E": E}).dropna()
    ts = sorted(dk["T"].unique()); nr = len(dk); s = 1.0
    out = [{"time": 0, "survival": 1.0}]
    for t in ts:
        d = ((dk["T"] == t) & (dk["E"] == 1)).sum()
        c_ = ((dk["T"] == t) & (dk["E"] == 0)).sum()
        if nr > 0: s *= (1 - d / nr)
        out.append({"time": t, "survival": s}); nr -= (d + c_)
    return pd.DataFrame(out)

def logrank_test(T1, E1, T2, E2):
    d1 = pd.DataFrame({"T": T1, "E": E1, "g": 0}).dropna()
    d2 = pd.DataFrame({"T": T2, "E": E2, "g": 1}).dropna()
    da = pd.concat([d1, d2])
    ts = sorted(da.loc[da["E"] == 1, "T"].unique())
    O1 = E1e = V = 0.0
    for t in ts:
        ar = da[da["T"] >= t]; n = len(ar); n1 = (ar["g"] == 0).sum()
        d   = ((ar["T"] == t) & (ar["E"] == 1)).sum()
        d1v = ((ar["T"] == t) & (ar["E"] == 1) & (ar["g"] == 0)).sum()
        if n < 2: continue
        e1 = n1 * d / n; O1 += d1v; E1e += e1
        V += n1 * (n - n1) * d * (n - d) / (n**2 * (n - 1)) if n > 1 else 0
    if V <= 0: return np.nan, np.nan
    chi2 = (O1 - E1e) ** 2 / V
    return chi2, 1 - stats.chi2.cdf(chi2, 1)

def compute_smd(t, c):
    t = pd.Series(t).dropna(); c = pd.Series(c).dropna()
    if len(t) < 2 or len(c) < 2: return np.nan
    ps = np.sqrt((t.var() + c.var()) / 2)
    return (t.mean() - c.mean()) / ps if ps > 0 else 0

def rd_ci(y1, y0, alpha=0.05):
    y1 = pd.Series(y1).dropna().astype(float)
    y0 = pd.Series(y0).dropna().astype(float)
    p1, p0 = y1.mean(), y0.mean(); n1, n0 = len(y1), len(y0)
    rd = p1 - p0
    se = np.sqrt(p1*(1-p1)/max(n1,1) + p0*(1-p0)/max(n0,1))
    z  = stats.norm.ppf(1 - alpha/2)
    return rd, rd - z*se, rd + z*se

def safe_qcut_rank(s, q, labels=False):
    r     = pd.Series(s).rank(method="first")
    q_eff = min(q, r.nunique())
    if q_eff < 2:
        return pd.Series(np.zeros(len(r), dtype=int), index=r.index)
    return pd.qcut(r, q=q_eff, labels=labels, duplicates="drop")

def xgb_crossfit(df, feat_list, target, cfg, group_col=None,
                 max_depth=4, colsample=0.6, mcw=20, num_rounds=300,
                 objective="binary:logistic", is_binary=True):
    """
    Cross-fitted OOF predictions from XGBoost.
    KEY: feat_list is NOT filtered by notna() — XGBoost handles NaN natively.
    GroupKFold(user_id) avoids user-level leakage when a user appears multiple times.
    """
    fl = [f for f in feat_list if f in df.columns]
    if not fl:
        return np.full(len(df), np.nan), np.nan, np.nan, 0

    X   = df[fl].values.astype(np.float32)
    y   = df[target].values.astype(int if is_binary else float)
    oof = np.full(len(df), np.nan)

    use_group = (
        group_col is not None
        and group_col in df.columns
        and df[group_col].nunique() >= cfg["n_folds"]
    )
    if use_group:
        splitter   = GroupKFold(n_splits=cfg["n_folds"])
        split_iter = splitter.split(X, y, groups=df[group_col].values)
    else:
        splitter   = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True,
                                     random_state=cfg["seed"])
        split_iter = splitter.split(X, y if is_binary else np.zeros(len(y)))

    params = dict(
        objective        = objective,
        eval_metric      = "logloss" if is_binary else "rmse",
        max_depth        = max_depth,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = colsample,
        min_child_weight = mcw,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        device           = cfg.get("xgb_device", "cpu"),
        verbosity        = 0,
    )
    aucs = []
    for tr_i, va_i in split_iter:
        dt  = xgb.DMatrix(X[tr_i], label=y[tr_i], feature_names=fl, missing=np.nan)
        dv  = xgb.DMatrix(X[va_i], label=y[va_i], feature_names=fl, missing=np.nan)
        bst = xgb.train(params, dt, num_boost_round=num_rounds,
                        evals=[(dv, "v")], early_stopping_rounds=30,
                        verbose_eval=False)
        oof[va_i] = bst.predict(dv)
        if is_binary and len(np.unique(y[va_i])) == 2:
            aucs.append(roc_auc_score(y[va_i], oof[va_i]))

    mean_m = np.mean(aucs) if aucs else np.nan
    std_m  = np.std(aucs)  if aucs else np.nan
    return oof, mean_m, std_m, len(fl)

def safe_glm(formula, data, family=None, cluster_col=None):
    """GLM with cluster-robust SE; returns None on failure."""
    if family is None:
        family = sm.families.Binomial()
    try:
        if cluster_col and cluster_col in data.columns:
            m = smf.glm(formula, data=data, family=family).fit(
                cov_type="cluster", cov_kwds={"groups": data[cluster_col]})
        else:
            m = smf.glm(formula, data=data, family=family).fit(cov_type="HC1")
        return m
    except Exception as e:
        print(f"  [GLM failed] {formula[:60]}... — {e}")
        return None

print("Utility functions loaded.")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION B: DATA LOADING & PREPROCESSING                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
C = CFG["cols"]

raw = pd.read_csv(CFG["data_path"], low_memory=False)
missing_req = [c for c in REQUIRED_COLS if c not in raw.columns]
if missing_req:
    raise ValueError(f"Missing required columns: {missing_req}")
print(f"Loaded {len(raw):,} rows, {len(raw.columns)} columns.")

# ── B1: Basic cleaning ────────────────────────────────────────────────────
def basic_clean(df):
    df = df.copy()
    df[C["event_date"]] = pd.to_datetime(df[C["event_date"]], errors="coerce")
    for col in [C["bf_credit"], C["af_credit"]]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in [C["label1"], C["label2"]]:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan

    df = df.dropna(subset=[C["event_date"], C["bf_credit"], C["af_credit"]]).copy()
    df["event_date"] = df[C["event_date"]].dt.normalize()

    df["treat_dir"] = np.select(
        [df[C["af_credit"]] > df[C["bf_credit"]],
         df[C["af_credit"]] == df[C["bf_credit"]],
         df[C["af_credit"]] < df[C["bf_credit"]]],
        ["inc", "same", "dec"], default="unknown",
    )
    df = df[df["treat_dir"].isin(["inc", "same", "dec"])].copy()
    df["is_inc"]    = (df["treat_dir"] == "inc").astype(int)
    df["is_dec"]    = (df["treat_dir"] == "dec").astype(int)
    df["delta_abs"] = df[C["af_credit"]] - df[C["bf_credit"]]
    bf_safe = df[C["bf_credit"]].clip(lower=1.0)
    df["delta_rel"] = df["delta_abs"] / bf_safe
    df["delta_log"] = np.log(df[C["af_credit"]].clip(lower=1.0) / bf_safe)
    df["ratio_cr"]  = df[C["af_credit"]].clip(lower=1.0) / bf_safe

    if C["model_type"] in df.columns:
        mt = df[C["model_type"]].astype(str).str.upper()
        df["model_family"] = np.where(mt.str.contains("V4"), "v4",
                             np.where(mt.str.contains("V5"), "v5", "other"))
    else:
        df["model_family"] = "unknown"

    # Mature label & default
    for col in [C["label1"], C["label2"]]:
        df[f"{col}_mature"] = df[col].notna().astype(int)
    df["label_any_mature"] = (
        df[f"{C['label1']}_mature"] | df[f"{C['label2']}_mature"]
    ).astype(int)
    df["default"] = np.where(
        df["label_any_mature"] == 1,
        np.fmax(df[C["label1"]].fillna(0), df[C["label2"]].fillna(0)),
        np.nan,
    )
    return df

df_2w = basic_clean(raw)
print(f"After cleaning: {len(df_2w):,} rows, {df_2w[C['user_id']].nunique():,} users")

# ── B2: Load XGB models and score all records ─────────────────────────────
feat_df_v4 = pd.read_excel(CFG["feature_path_v4"])
feat_df_v5 = pd.read_excel(CFG["feature_path_v5"])

def get_feat_list(df):
    return df["索引"].dropna().tolist() if "索引" in df.columns else df.iloc[:, 0].dropna().tolist()

CFG["feat_v4"]    = get_feat_list(feat_df_v4)
CFG["feat_v5"]    = get_feat_list(feat_df_v5)
CFG["feat_inter"] = sorted(set(CFG["feat_v4"]) & set(CFG["feat_v5"]))
CFG["feat_union"] = sorted(set(CFG["feat_v4"]) | set(CFG["feat_v5"]))
print(f"Features — V4:{len(CFG['feat_v4'])}, V5:{len(CFG['feat_v5'])}, "
      f"inter:{len(CFG['feat_inter'])}, union:{len(CFG['feat_union'])}")

bv4 = xgb.Booster(); bv4.load_model(CFG["model_path_v4"])
bv5 = xgb.Booster(); bv5.load_model(CFG["model_path_v5"])

def score_with_model(df, booster, flist, out_col):
    X = pd.DataFrame(
        {f: pd.to_numeric(df[f], errors="coerce").values if f in df.columns else np.nan
         for f in flist}, index=df.index,
    )
    df[out_col] = booster.predict(xgb.DMatrix(X.values, feature_names=flist, missing=np.nan))
    return df

df_2w = score_with_model(df_2w, bv4, CFG["feat_v4"], "score_v4")
df_2w = score_with_model(df_2w, bv5, CFG["feat_v5"], "score_v5")
print("Scores computed.")

# ── B3: Overlap window & continuing users ─────────────────────────────────
# Continuing users: had records BEFORE overlap start AND within overlap window
# (Research memo: these are the identification sample; cannot extrapolate to all users)
dm = df_2w.groupby(["event_date", "model_family"]).size().unstack(fill_value=0)
ov = dm.index[(dm.get("v4", pd.Series(dtype=int)) > 0) &
              (dm.get("v5", pd.Series(dtype=int)) > 0)]
if len(ov):
    CFG["overlap_start"] = max(CFG["overlap_start"], ov.min())
    CFG["overlap_end"]   = min(CFG["overlap_end"],   ov.max())

df_sorted = df_2w.sort_values([C["user_id"], "event_date", C["event_date"]])
users_pre  = set(df_sorted[df_sorted["event_date"] <  CFG["overlap_start"]][C["user_id"]])
users_in   = set(df_sorted[(df_sorted["event_date"] >= CFG["overlap_start"]) &
                            (df_sorted["event_date"] <= CFG["overlap_end"])][C["user_id"]])
continuing = users_pre & users_in

df_aw = df_sorted[
    (df_sorted["event_date"] >= CFG["overlap_start"]) &
    (df_sorted["event_date"] <= CFG["overlap_end"])   &
    (df_sorted[C["user_id"]].isin(continuing))
].copy()

# ── B4: Anchor = first event in overlap window per continuing user ─────────
# NOTE (research memo §5.3): the researcher does not endorse this anchor definition.
# It is retained as the starting entry point for user-level survival analysis.
anchor = df_aw.loc[df_aw.groupby(C["user_id"])["event_date"].idxmin()].copy()
anchor = anchor.rename(columns={"event_date": "anchor_date"})

df_all = df_sorted[df_sorted[C["user_id"]].isin(continuing)].copy()

# ── B5: Pre-anchor history features  H_i^0 ────────────────────────────────
# For each user: summary of events BEFORE anchor_date
pre_events = df_all.merge(
    anchor[[C["user_id"], "anchor_date"]], on=C["user_id"], how="inner"
)
pre_events = pre_events[pre_events["event_date"] < pre_events["anchor_date"]].copy()

def hist_summary(grp):
    n_ev   = len(grp)
    n_inc  = grp["is_inc"].sum()
    n_dec  = grp["is_dec"].sum()
    n_same = (grp["treat_dir"] == "same").sum()
    cum_dr = grp["delta_rel"].sum()
    last_d = grp["delta_rel"].iloc[-1] if n_ev > 0 else 0.0
    last_t = grp["treat_dir"].iloc[-1] if n_ev > 0 else "none"
    return pd.Series(dict(
        pre_n_events  = n_ev,
        pre_n_inc     = n_inc,
        pre_n_dec     = n_dec,
        pre_n_same    = n_same,
        pre_cum_dr    = cum_dr,
        pre_last_dr   = last_d,
        pre_last_treat= last_t,
    ))

hist_feat = (
    pre_events.sort_values([C["user_id"], "event_date"])
    .groupby(C["user_id"], group_keys=False)
    .apply(hist_summary, include_groups=False)
    .reset_index()
)

# ── B6: User-level survival table ─────────────────────────────────────────
first_def = (
    df_all[df_all["default"] > 0]
    .sort_values("event_date")
    .groupby(C["user_id"])["event_date"]
    .first().reset_index()
    .rename(columns={"event_date": "default_date"})
)
last_obs = (
    df_all.groupby(C["user_id"])["event_date"]
    .max().reset_index()
    .rename(columns={"event_date": "last_date"})
)

union_in_anchor = [f for f in CFG["feat_union"] if f in anchor.columns]
us = (
    anchor[[C["user_id"], "anchor_date", "model_family", "treat_dir",
            C["bf_credit"], C["af_credit"], "score_v4", "score_v5",
            "is_inc", "is_dec", "delta_abs", "delta_rel", "delta_log"] + union_in_anchor]
    .merge(first_def, on=C["user_id"], how="left")
    .merge(last_obs,  on=C["user_id"], how="left")
    .merge(hist_feat, on=C["user_id"], how="left")
)

# Drop users whose first default was before anchor (data quality guard)
us = us[~(us["default_date"].notna() & (us["default_date"] < us["anchor_date"]))].copy()

us["defaulted"] = (
    us["default_date"].notna() & (us["default_date"] >= us["anchor_date"])
).astype(int)
us["T"] = np.where(
    us["defaulted"] == 1,
    (us["default_date"] - us["anchor_date"]).dt.days,
    (us["last_date"]   - us["anchor_date"]).dt.days,
).clip(1, CFG["max_followup_days"])
us.loc[
    (us["defaulted"] == 1) &
    ((us["default_date"] - us["anchor_date"]).dt.days > CFG["max_followup_days"]),
    "defaulted"] = 0

us["is_v5"]    = (us["model_family"] == "v5").astype(int)
us["gap_raw"]  = us["score_v4"] - us["score_v5"]
us["gap_std"]  = (us["gap_raw"] - us["gap_raw"].mean()) / us["gap_raw"].std().clip(lower=1e-8)
us["score_avg"]= (us["score_v4"] + us["score_v5"]) / 2
us["lc"]       = np.log1p(us[C["bf_credit"]])
us["lc_z"]     = (us["lc"] - us["lc"].mean()) / us["lc"].std().clip(lower=1e-8)

print(f"\nOverlap: {CFG['overlap_start'].date()} → {CFG['overlap_end'].date()}")
print(f"Continuing users: {len(continuing):,}")
print(f"User-level table: {len(us):,}  V4={(us['is_v5']==0).sum()}  V5={(us['is_v5']==1).sum()}")
print(f"Defaults: {us['defaulted'].sum()} ({us['defaulted'].mean():.1%})")
print(f"Treatment: inc={(us['treat_dir']=='inc').sum()}, "
      f"same={(us['treat_dir']=='same').sum()}, dec={(us['treat_dir']=='dec').sum()}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION C: KM CURVES & LOG-RANK TEST                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for mf, c in [("v4", "C0"), ("v5", "C1")]:
    s  = us[us["model_family"] == mf]
    km = kaplan_meier(s["T"], s["defaulted"])
    axes[0].step(km["time"], km["survival"], where="post", lw=2, color=c,
                 label=f"{mf} (n={len(s)}, DR={s['defaulted'].mean():.1%})")
axes[0].set(xlabel="Days from anchor", ylabel="S(t)", title="KM: V4 vs V5"); axes[0].legend()

for td, c in [("inc", "C1"), ("same", "C2"), ("dec", "C3")]:
    s  = us[us["treat_dir"] == td]
    km = kaplan_meier(s["T"], s["defaulted"])
    axes[1].step(km["time"], km["survival"], where="post", lw=2, color=c,
                 label=f"{td} (n={len(s)}, DR={s['defaulted'].mean():.1%})")
axes[1].set(xlabel="Days from anchor", ylabel="S(t)", title="KM: By treatment"); axes[1].legend()

for mf, ls in [("v4", "-"), ("v5", "--")]:
    for td, c in [("inc", "C1"), ("same", "C2"), ("dec", "C3")]:
        s = us[(us["model_family"] == mf) & (us["treat_dir"] == td)]
        if len(s) < 10: continue
        km = kaplan_meier(s["T"], s["defaulted"])
        axes[2].step(km["time"], km["survival"], where="post", lw=1.5,
                     color=c, ls=ls, label=f"{mf}-{td} ({s['defaulted'].mean():.0%})")
axes[2].set(xlabel="Days from anchor", ylabel="S(t)", title="KM: Model × Treatment")
axes[2].legend(fontsize=6, ncol=2)
fig.tight_layout(); plt.show()

# Log-rank
s4 = us[us["is_v5"] == 0]; s5 = us[us["is_v5"] == 1]
chi2_lr, p_lr = logrank_test(s4["T"], s4["defaulted"], s5["T"], s5["defaulted"])
print(f"Log-rank V4 vs V5: chi2={chi2_lr:.3f}, p={p_lr:.4f} {stars(p_lr)}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION D: m(X) — FULL FEATURE SET, GROUPKFOLD BY user_id               ║
# ║  Key fix vs prior version: use feat_union, no NaN-based column filtering  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n" + "="*78)
print("  SECTION D: m(X) CONSTRUCTION")
print("="*78)

# Select feature set
_fset = CFG["feat_union"] if CFG["use_full_features"] else CFG["feat_inter"]
# Include all features that exist in us; XGBoost handles NaN internally
feat_mX_us = [f for f in _fset if f in us.columns]

print(f"  Feature set   : {'feat_union' if CFG['use_full_features'] else 'feat_inter'}")
print(f"  Columns in us  present in feat set: {len(feat_mX_us)}")

risk_oof_us, risk_auc_us, risk_std_us, n_feat_us = xgb_crossfit(
    us, feat_mX_us, "defaulted", CFG,
    group_col=C["user_id"],   # GroupKFold: each user in exactly one fold
    mcw=10, num_rounds=300,
)
us["mX"] = risk_oof_us
us["mX"] = us["mX"].fillna(us["mX"].median())  # safety fill

mX_summary = pd.DataFrame({
    "metric"  : ["n_features_passed", "n_features_in_data", "OOF_AUC_mean", "OOF_AUC_std"],
    "value"   : [len(_fset), n_feat_us, risk_auc_us, risk_std_us],
})
display(mX_summary)
print(f"\n  m(X) OOF AUC = {risk_auc_us:.4f} ± {risk_std_us:.4f}  (n_features={n_feat_us})")

# Diagnostic: m(X) vs defaulted
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].hist(us.loc[us["defaulted"]==0, "mX"], bins=40, alpha=.5, density=True, label="no default")
axes[0].hist(us.loc[us["defaulted"]==1, "mX"], bins=40, alpha=.5, density=True, label="default")
axes[0].set(title="m(X) distribution by outcome", xlabel="m(X)"); axes[0].legend()

fpr, tpr, _ = roc_curve(us["defaulted"], us["mX"])
axes[1].plot(fpr, tpr, lw=2)
axes[1].plot([0,1],[0,1],"k--",alpha=.4)
axes[1].set(title=f"m(X) ROC (AUC={risk_auc_us:.3f})", xlabel="FPR", ylabel="TPR")

axes[2].scatter(us["mX"], us["defaulted"] + np.random.normal(0, .02, len(us)),
                alpha=.1, s=4)
axes[2].set(title="m(X) vs defaulted (jittered)", xlabel="m(X)", ylabel="defaulted")
fig.tight_layout(); plt.show()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION E: COUNTING PROCESS + CUMULATIVE / DOSE EXPOSURE CONSTRUCTION    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n" + "="*78)
print("  SECTION E: COUNTING PROCESS + EXPOSURE CONSTRUCTION")
print("="*78)
# Research memo §5.5 / §7.11: exposure types —
#   (1) cum_*_rel: cumulative credit-change relative to base credit
#   (2) dose_*_rel: integral of excess credit exposure × time (relative to base)
#   (3) lag1 versions of all exposures to avoid contemporaneous feedback
# NOTE (memo §7.10): tv_is_inc is event-level; this is the v1 definition.
#   Cumulative/dose exposures are the first step toward user-path exposure.

anchor_info = anchor[[C["user_id"], "anchor_date", "model_family",
                       C["bf_credit"], "score_v4", "score_v5"]].copy()
ep = df_all.merge(anchor_info, on=C["user_id"], how="inner", suffixes=("", "_anc"))
ep["day"] = (ep["event_date"] - ep["anchor_date"]).dt.days
ep = ep[(ep["day"] >= 0) & (ep["day"] <= CFG["max_followup_days"])].copy()
ep["event_default"] = (ep["default"] > 0).astype(int)
ep = ep.sort_values([C["user_id"], "day", C["event_date"]])

EPS = 0.01
cp_rows = []

for uid, grp in ep.groupby(C["user_id"]):
    grp = grp.sort_values(["day", C["event_date"]]).reset_index(drop=True)
    n   = len(grp)
    if n == 0:
        continue

    # ── Effective times (handle same-day ties) ────────────────────────────
    eff = np.empty(n, dtype=float)
    for idx in range(n):
        d  = grp.iloc[idx]["day"]
        K  = (grp["day"] == d).sum()
        k  = int((grp["day"] == d).iloc[:idx+1].sum()) - 1
        eff[idx] = d + (k + 1) / (K + 1)
    for idx in range(n):
        eff[idx] = (max(eff[idx], EPS) if idx == 0
                    else max(eff[idx], eff[idx-1] + EPS))

    # ── Base credit (before the first anchor-window event) ────────────────
    base_credit = float(grp.iloc[0][C["bf_credit"]])
    base_safe   = max(base_credit, 1.0)

    # Running accumulators
    cum_pos_rel = cum_neg_rel = cum_net_rel = cum_abs_rel = 0.0
    dose_pos_rel = dose_neg_rel = dose_net_rel = dose_abs_rel = 0.0
    current_level = base_credit
    prev_time     = 0.0

    for i in range(n):
        row          = grp.iloc[i]
        current_time = eff[i]
        duration     = current_time - prev_time

        # Excess credit during [prev_time, current_time] (pre-event state)
        excess_rel = (current_level - base_credit) / base_safe

        # ── Dose: integral of excess over time ────────────────────────────
        # These are LAGGED doses (before this event's delta lands)
        dose_pos_rel_lag = dose_pos_rel
        dose_neg_rel_lag = dose_neg_rel
        dose_net_rel_lag = dose_net_rel
        dose_abs_rel_lag = dose_abs_rel

        dose_pos_rel += max(0.0, excess_rel) * duration
        dose_neg_rel += max(0.0, -excess_rel) * duration
        dose_net_rel += excess_rel * duration
        dose_abs_rel += abs(excess_rel) * duration

        # ── Cumulative: pre-event sums (already lagged) ───────────────────
        cum_pos_rel_lag = cum_pos_rel
        cum_neg_rel_lag = cum_neg_rel
        cum_net_rel_lag = cum_net_rel
        cum_abs_rel_lag = cum_abs_rel

        # Apply this event's delta
        delta      = float(row[C["af_credit"]]) - float(row[C["bf_credit"]])
        delta_rel  = delta / base_safe
        cum_pos_rel += max(0.0,  delta_rel)
        cum_neg_rel += min(0.0,  delta_rel)
        cum_net_rel += delta_rel
        cum_abs_rel += abs(delta_rel)

        current_level = float(row[C["af_credit"]])
        prev_time     = current_time

        start = eff[i-1] if i > 0 else 0.0
        ev    = int(row["event_default"] == 1)

        cp_rows.append({
            "user_id"          : uid,
            "interval_idx"     : i,
            "start"            : start,
            "stop"             : current_time,
            "event"            : ev,
            "day"              : int(row["day"]),
            "is_v5"            : int(str(row["model_family_anc"]) == "v5"
                                     if "model_family_anc" in row.index
                                     else row.get("model_family", "v4") == "v5"),
            # Event-level treatment indicators (contemporaneous)
            "tv_is_inc"        : int(row["is_inc"]),
            "tv_is_dec"        : int(row["is_dec"]),
            "tv_is_same"       : int(row["treat_dir"] == "same"),
            # Scores (contemporaneous)
            "tv_score_v4"      : float(row["score_v4"]),
            "tv_score_v5"      : float(row["score_v5"]),
            # Credit
            "base_credit"      : base_credit,
            "tv_delta_rel"     : delta_rel,
            # ── Cumulative exposures (AFTER this event, i.e. contemporaneous) ──
            "cum_pos_rel"      : cum_pos_rel,
            "cum_neg_rel"      : cum_neg_rel,
            "cum_net_rel"      : cum_net_rel,
            "cum_abs_rel"      : cum_abs_rel,
            # ── Cumulative exposures LAG1 (BEFORE this event's delta) ─────
            "cum_pos_rel_lag1" : cum_pos_rel_lag,
            "cum_neg_rel_lag1" : cum_neg_rel_lag,
            "cum_net_rel_lag1" : cum_net_rel_lag,
            "cum_abs_rel_lag1" : cum_abs_rel_lag,
            # ── Dose exposures (up to current_time, includes this interval) ─
            "dose_pos_rel"     : dose_pos_rel,
            "dose_neg_rel"     : dose_neg_rel,
            "dose_net_rel"     : dose_net_rel,
            "dose_abs_rel"     : dose_abs_rel,
            # ── Dose exposures LAG1 (before this interval's contribution) ──
            "dose_pos_rel_lag1": dose_pos_rel_lag,
            "dose_neg_rel_lag1": dose_neg_rel_lag,
            "dose_net_rel_lag1": dose_net_rel_lag,
            "dose_abs_rel_lag1": dose_abs_rel_lag,
        })
        if ev == 1:
            break   # user has defaulted; no further intervals

cp = pd.DataFrame(cp_rows)
cp["t_mid"]    = (cp["start"] + cp["stop"]) / 2
cp["t_mid_sq"] = cp["t_mid"] ** 2

# ── Lag1 score & treatment via shift ──────────────────────────────────────
# (additional lag1 for score and event indicators via pandas shift)
for col in ["tv_score_v4", "tv_score_v5", "tv_is_inc", "tv_is_dec", "tv_is_same"]:
    cp[f"{col}_lag1"] = cp.groupby("user_id")[col].shift(1)

cp["tv_own_score"]      = np.where(cp["is_v5"] == 1, cp["tv_score_v5"], cp["tv_score_v4"])
cp["tv_own_score_lag1"] = cp.groupby("user_id")["tv_own_score"].shift(1)

# ── Merge covariate features for m(X) at event level ─────────────────────
feat_in_ep  = [f for f in feat_mX_us if f in ep.columns]
ep_feat_df  = ep[["user_id"] + feat_in_ep].copy()
ep_feat_df["_rank"] = ep_feat_df.groupby("user_id").cumcount()
cp["_rank"]  = cp.groupby("user_id").cumcount()
cp = cp.merge(ep_feat_df, on=["user_id", "_rank"], how="left", suffixes=("", "_ep"))
feat_in_cp  = [f for f in feat_in_ep if f in cp.columns]

# ── m(X) at event level (GroupKFold by user_id) ───────────────────────────
risk_oof_cp, risk_auc_cp, risk_std_cp, n_feat_cp = xgb_crossfit(
    cp, feat_in_cp, "event", CFG,
    group_col="user_id", mcw=30, num_rounds=300,
)
cp["mX"] = risk_oof_cp

# Fill remaining NaN in key covariates with median (statsmodels GLM needs no NaN)
for col in ["tv_score_v4", "tv_score_v5", "tv_own_score", "mX",
            "tv_score_v4_lag1", "tv_score_v5_lag1", "tv_own_score_lag1"]:
    if col in cp.columns:
        cp[col] = cp[col].fillna(cp[col].median())

print(f"\nCounting process: {len(cp):,} intervals, "
      f"{cp['user_id'].nunique():,} users, {cp['event'].sum()} defaults")
print(f"m(X) event-level OOF AUC: {risk_auc_cp:.4f} ± {risk_std_cp:.4f}")
print(f"Features attached to CP: {n_feat_cp}")

# Integrity check
viol = (cp["stop"] <= cp["start"]).sum()
print(f"Stop <= Start violations: {viol}")

# ── Descriptive: exposure distributions ───────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes[0],
        ["cum_pos_rel_lag1", "cum_neg_rel_lag1", "cum_net_rel_lag1", "cum_abs_rel_lag1"]):
    ax.hist(cp[col].dropna(), bins=50, alpha=.7)
    ax.set(title=col, xlabel="value")
for ax, col in zip(axes[1],
        ["dose_pos_rel_lag1", "dose_neg_rel_lag1", "dose_net_rel_lag1", "dose_abs_rel_lag1"]):
    ax.hist(cp[col].dropna(), bins=50, alpha=.7)
    ax.set(title=col, xlabel="value")
fig.suptitle("Lag-1 Cumulative & Dose Exposure Distributions", fontsize=12)
fig.tight_layout(); plt.show()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TASK 1A: USER-LEVEL PfP                                                  ║
# ║  Model: Pr(Y=1) = g(α + β·S + γ·m(X) + δ·lc_z)                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n" + "="*78)
print("  TASK 1A: USER-LEVEL PfP")
print("="*78)

# Evidence bucket initialised here; filled throughout tasks
EV = {}

pfp_specs = [
    ("pfp_1", "defaulted ~ mX + lc_z"),
    ("pfp_2", "defaulted ~ mX + lc_z + score_v4"),
    ("pfp_3", "defaulted ~ mX + lc_z + score_v5"),
    ("pfp_4", "defaulted ~ mX + lc_z + score_v4 + score_v5"),
    ("pfp_5", "defaulted ~ mX + lc_z + score_v5 * is_v5"),
]

pfp_models = {}
for name, fml in pfp_specs:
    m = safe_glm(fml, us, cluster_col=None)
    if m:
        pfp_models[name] = m
        stata_table(m, f"[{name}] {fml}")

# Evidence: does V5 score have residual signal beyond m(X)?
if "pfp_3" in pfp_models:
    m3 = pfp_models["pfp_3"]
    v5_coef = m3.params.get("score_v5", np.nan)
    v5_p    = m3.pvalues.get("score_v5", np.nan)
    EV["pfp_v5_residual"] = dict(
        hypothesis = "V5 score has residual signal beyond m(X)",
        tested     = True,
        supported  = (v5_p < CFG["alpha"]),
        detail     = f"score_v5 coef={v5_coef:.4f} p={v5_p:.4f} {stars(v5_p)}",
    )
    print(f"\n  PfP evidence: {EV['pfp_v5_residual']['detail']}")

if "pfp_2" in pfp_models:
    m2 = pfp_models["pfp_2"]
    v4_coef = m2.params.get("score_v4", np.nan)
    v4_p    = m2.pvalues.get("score_v4", np.nan)
    EV["pfp_v4_residual"] = dict(
        hypothesis = "V4 score has residual signal beyond m(X)",
        tested     = True,
        supported  = (v4_p < CFG["alpha"]),
        detail     = f"score_v4 coef={v4_coef:.4f} p={v4_p:.4f} {stars(v4_p)}",
    )

# ── Coefficient comparison plot ───────────────────────────────────────────
rows_pfp = []
for nm, m in pfp_models.items():
    for v in ["score_v4", "score_v5", "is_v5"]:
        if v in m.params.index:
            rows_pfp.append({"spec": nm, "var": v,
                             "coef": m.params[v], "p": m.pvalues[v],
                             "ci_lo": m.conf_int().loc[v, 0],
                             "ci_hi": m.conf_int().loc[v, 1]})
df_pfp = pd.DataFrame(rows_pfp)
if not df_pfp.empty:
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (var, grp) in enumerate(df_pfp.groupby("var")):
        x_pos = np.arange(len(grp)) + i * (len(grp) + 1)
        ax.bar(x_pos, grp["coef"], yerr=[grp["coef"] - grp["ci_lo"],
                                          grp["ci_hi"] - grp["coef"]],
               capsize=4, alpha=.7, label=var)
    ax.axhline(0, c="k", lw=.8)
    ax.set(title="User-level PfP: score coefficients by spec", ylabel="Coef")
    ax.legend()
    fig.tight_layout(); plt.show()

# ── DML partialling-out ────────────────────────────────────────────────────
if CFG["run_dml"]:
    print("\n  ── DML partialling-out ─────────────────────────────────────────────")
    # Y residual = defaulted - m(X)   [already have mX as OOF]
    Y_res = us["defaulted"].values - us["mX"].values

    dml_results = {}
    for score_col in ["score_v4", "score_v5", "lc_z"]:
        # Cross-fit E[score | X] using regression
        oof_s, _, _, _ = xgb_crossfit(
            us, feat_mX_us, score_col, CFG,
            group_col=C["user_id"],
            objective="reg:squarederror", is_binary=False,
            mcw=10, num_rounds=200,
        )
        S_res = us[score_col].values - oof_s
        mask  = np.isfinite(Y_res) & np.isfinite(S_res)
        X_dml = S_res[mask].reshape(-1, 1)
        y_dml = Y_res[mask]
        lr    = LinearRegression().fit(X_dml, y_dml)
        theta = lr.coef_[0]
        # Bootstrap SE
        n_boot = 200
        boot_c = [LinearRegression().fit(
                      X_dml[idx], y_dml[idx]
                  ).coef_[0]
                  for idx in (np.random.default_rng(42+k).choice(
                      len(y_dml), len(y_dml)) for k in range(n_boot))]
        se_boot = np.std(boot_c)
        z_val   = theta / se_boot if se_boot > 0 else 0
        p_val   = 2 * (1 - stats.norm.cdf(abs(z_val)))
        dml_results[score_col] = dict(theta=theta, se=se_boot, z=z_val, p=p_val)
        print(f"  DML [{score_col}]: θ={theta:+.4f}  SE={se_boot:.4f}  "
              f"z={z_val:.2f}  p={p_val:.4f} {stars(p_val)}")

    EV["dml_v5_score"] = dict(
        hypothesis = "V5 score has DML partialling-out effect on default",
        tested     = True,
        supported  = (dml_results.get("score_v5", {}).get("p", 1) < CFG["alpha"]),
        detail     = (f"theta={dml_results.get('score_v5',{}).get('theta',np.nan):.4f} "
                      f"p={dml_results.get('score_v5',{}).get('p',np.nan):.4f}"),
    )

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TASK 1B: HAZARD-LEVEL PfP                                                ║
# ║  Model: Pr(E_it=1|H_it) = g(λ(t) + β·S_{i,t-1} + γ·m(X_i))             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n" + "="*78)
print("  TASK 1B: HAZARD-LEVEL PfP")
print("="*78)

hazpfp_specs = [
    ("hpfp_1", "event ~ t_mid + t_mid_sq + mX"),
    ("hpfp_2", "event ~ t_mid + t_mid_sq + mX + tv_score_v4_lag1"),
    ("hpfp_3", "event ~ t_mid + t_mid_sq + mX + tv_score_v5_lag1"),
    ("hpfp_4", "event ~ t_mid + t_mid_sq + mX + tv_score_v4_lag1 + tv_score_v5_lag1"),
    ("hpfp_5", "event ~ t_mid + t_mid_sq + mX + tv_own_score_lag1"),
]

hazpfp_models = {}
cp_hpfp = cp.dropna(subset=["mX", "tv_score_v4_lag1", "tv_score_v5_lag1",
                              "tv_own_score_lag1"]).copy()

for name, fml in hazpfp_specs:
    m = safe_glm(fml, cp_hpfp, cluster_col="user_id")
    if m:
        hazpfp_models[name] = m
        stata_table(m, f"[{name}] {fml}")

# Evidence
for spec_name, score_var in [("hpfp_3", "tv_score_v5_lag1"), ("hpfp_2", "tv_score_v4_lag1")]:
    if spec_name in hazpfp_models:
        mm  = hazpfp_models[spec_name]
        ccc = mm.params.get(score_var, np.nan)
        ppp = mm.pvalues.get(score_var, np.nan)
        key = f"hazpfp_{score_var}"
        EV[key] = dict(
            hypothesis = f"Lagged {score_var} has residual hazard signal beyond m(X)",
            tested     = True,
            supported  = (ppp < CFG["alpha"]),
            detail     = f"coef={ccc:.4f} p={ppp:.4f} {stars(ppp)}",
        )
        print(f"  Hazard-PfP [{score_var}]: coef={ccc:.4f} p={ppp:.4f} {stars(ppp)}")

# ── Summary comparison table ──────────────────────────────────────────────
print(f"\n  {'Spec':<10} {'tv_score_v4_lag1':>20} {'tv_score_v5_lag1':>20} {'tv_own_score_lag1':>20}")
print("  " + "─"*72)
for nm, m in hazpfp_models.items():
    row = f"  {nm:<10}"
    for v in ["tv_score_v4_lag1", "tv_score_v5_lag1", "tv_own_score_lag1"]:
        if v in m.params.index:
            row += f" {m.params[v]:>+8.4f}{stars(m.pvalues[v]):<3}{' ':>9}"
        else:
            row += f" {'—':>20}"
    print(row)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TASK 2: RULE / POLICY MAPPING                                            ║
# ║  Distinguish: distribution shift / calibration shift /                    ║
# ║               mapping level shift / mapping slope difference              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n" + "="*78)
print("  TASK 2: RULE / POLICY MAPPING")
print("="*78)

# ── 2A: Score distribution comparison + KS test ───────────────────────────
print("\n  ── 2A: Score distribution comparison ─────────────────────────────────")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
ks_results = {}

for ax_row, score_col, title in [
    (0, "score_v4", "V4 score distribution by regime"),
    (1, "score_v5", "V5 score distribution by regime"),
]:
    s_v4 = us.loc[us["is_v5"]==0, score_col].dropna()
    s_v5 = us.loc[us["is_v5"]==1, score_col].dropna()
    ks_stat, ks_p = ks_2samp(s_v4, s_v5)
    ks_results[score_col] = dict(ks_stat=ks_stat, ks_p=ks_p)

    ax = axes[ax_row, 0]
    ax.hist(s_v4, bins=40, alpha=.5, density=True, label=f"V4 regime (n={len(s_v4)})")
    ax.hist(s_v5, bins=40, alpha=.5, density=True, label=f"V5 regime (n={len(s_v5)})")
    ax.set(title=f"{title}\nKS={ks_stat:.3f} p={ks_p:.4f} {stars(ks_p)}", xlabel=score_col)
    ax.legend(fontsize=7)

    ax = axes[ax_row, 1]
    for s, lab, c in [(s_v4, "V4 regime", "C0"), (s_v5, "V5 regime", "C1")]:
        ax.hist(s, bins=40, alpha=.4, cumulative=True, density=True, label=lab, color=c)
    ax.set(title=f"CDF: {score_col}", xlabel=score_col, ylabel="CDF")
    ax.legend(fontsize=7)

    print(f"  KS [{score_col}]: stat={ks_stat:.4f}  p={ks_p:.4f} {stars(ks_p)}")

fig.suptitle("Task 2A: Score Distributions by Regime", fontsize=12)
fig.tight_layout(); plt.show()

EV["score_dist_shift_v4"] = dict(
    hypothesis = "V4 score distribution shifts between V4 and V5 regimes",
    tested     = True,
    supported  = (ks_results["score_v4"]["ks_p"] < CFG["alpha"]),
    detail     = (f"KS={ks_results['score_v4']['ks_stat']:.4f} "
                  f"p={ks_results['score_v4']['ks_p']:.4f}"),
)
EV["score_dist_shift_v5"] = dict(
    hypothesis = "V5 score distribution shifts between V4 and V5 regimes",
    tested     = True,
    supported  = (ks_results["score_v5"]["ks_p"] < CFG["alpha"]),
    detail     = (f"KS={ks_results['score_v5']['ks_stat']:.4f} "
                  f"p={ks_results['score_v5']['ks_p']:.4f}"),
)

# ── 2B: Score percentile mapping ──────────────────────────────────────────
print("\n  ── 2B: Percentile mapping ─────────────────────────────────────────────")

# Model-internal percentile: rank within V4-regime users and V5-regime users separately
us["score_v4_pct_model"] = us.groupby("is_v5")["score_v4"].rank(pct=True)
us["score_v5_pct_model"] = us.groupby("is_v5")["score_v5"].rank(pct=True)

# Active score percentile: regime-appropriate score
us["active_score"]     = np.where(us["is_v5"]==1, us["score_v5"], us["score_v4"])
us["active_score_pct"] = us.groupby("is_v5")["active_score"].rank(pct=True)

# Inc rate by active score percentile decile
us["active_pct_d10"]  = safe_qcut_rank(us["active_score_pct"], q=10, labels=False)
pct_map = (us.groupby(["active_pct_d10", "is_v5"])
            .agg(inc_rate=("is_inc","mean"), n=("user_id","count"))
            .reset_index())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for regime, c, lab in [(0, "C0", "V4 regime"), (1, "C1", "V5 regime")]:
    sub = pct_map[pct_map["is_v5"]==regime]
    axes[0].plot(sub["active_pct_d10"], sub["inc_rate"], "o-", color=c, label=lab)
axes[0].set(xlabel="Active score percentile decile", ylabel="inc rate",
            title="Inc rate by score percentile (within regime)")
axes[0].legend()

# Delta: V5 - V4 inc rate at same percentile
p4 = pct_map[pct_map["is_v5"]==0].set_index("active_pct_d10")["inc_rate"]
p5 = pct_map[pct_map["is_v5"]==1].set_index("active_pct_d10")["inc_rate"]
delta_pct = (p5 - p4).dropna()
axes[1].bar(delta_pct.index, delta_pct.values*100, alpha=.7, color="C3")
axes[1].axhline(0, c="k", lw=.8)
axes[1].set(xlabel="Percentile decile", ylabel="V5 - V4 inc rate (pp)",
            title="Inc rate lift: V5 vs V4 at same percentile")
fig.tight_layout(); plt.show()

# ── 2C: Logistic mapping test ─────────────────────────────────────────────
# Model: Pr(inc=1) = g(α + β·q + η·is_v5 + θ·(q×is_v5))
# η ≠ 0 → level shift; θ ≠ 0 → slope difference
print("\n  ── 2C: Logistic mapping test ──────────────────────────────────────────")
print("  Decomposing level shift vs slope difference (active-score percentile)")

m_map_base = safe_glm(
    "is_inc ~ active_score_pct + is_v5",
    us, cluster_col=None,
)
m_map_full = safe_glm(
    "is_inc ~ active_score_pct * is_v5",
    us, cluster_col=None,
)

if m_map_base:
    stata_table(m_map_base, "Mapping [base]: inc ~ q + is_v5  (level-shift check)")
if m_map_full:
    stata_table(m_map_full, "Mapping [full]: inc ~ q * is_v5  (slope-difference check)")

# Evidence — carefully separated, no conflation
level_shift_p = slope_diff_p = np.nan
if m_map_base:
    level_shift_coef = m_map_base.params.get("is_v5", np.nan)
    level_shift_p    = m_map_base.pvalues.get("is_v5", np.nan)
if m_map_full:
    slope_diff_coef  = m_map_full.params.get("active_score_pct:is_v5", np.nan)
    slope_diff_p     = m_map_full.pvalues.get("active_score_pct:is_v5", np.nan)

EV["mapping_level_shift"] = dict(
    hypothesis = "Mapping level shift: V5 regime has higher P(inc) at same percentile",
    tested     = not np.isnan(level_shift_p),
    supported  = (level_shift_p < CFG["alpha"]) if not np.isnan(level_shift_p) else False,
    detail     = (f"is_v5 coef={level_shift_coef:.4f} p={level_shift_p:.4f} "
                  f"{stars(level_shift_p)}" if not np.isnan(level_shift_p) else "n/a"),
)
# IMPORTANT: no slope difference ≠ "mapping stable"; only means no evidence of slope reshaping
EV["mapping_slope_diff"] = dict(
    hypothesis = "Mapping slope difference: V5 slope differs from V4 slope",
    tested     = not np.isnan(slope_diff_p),
    supported  = (slope_diff_p < CFG["alpha"]) if not np.isnan(slope_diff_p) else False,
    detail     = (f"interaction coef={slope_diff_coef:.4f} p={slope_diff_p:.4f} "
                  f"{stars(slope_diff_p)}" if not np.isnan(slope_diff_p) else "n/a"
                 ) + ("  [p>α → no strong evidence of slope reshaping; "
                       "does NOT imply mapping is stable overall]"
                       if not np.isnan(slope_diff_p) and slope_diff_p >= CFG["alpha"]
                       else ""),
)
for key in ["mapping_level_shift", "mapping_slope_diff"]:
    print(f"  {EV[key]['hypothesis']}: {EV[key]['detail']}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TASK 3: HAZARD MAIN ANALYSIS — EXPOSURE-SPEC GRID                        ║
# ║  9 specifications covering event-level / cumulative / dose exposures      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n" + "="*78)
print("  TASK 3: HAZARD MAIN ANALYSIS — EXPOSURE SPEC GRID")
print("="*78)
print("  NOTE: tv_is_inc/dec are event-level (contemporaneous) — memo §7.10.")
print("  Cumulative/dose exposures use lag1 to reduce contemporaneous feedback.")

TIME = "t_mid + t_mid_sq"
REGIME = "is_v5"
EVENT  = "tv_is_inc + tv_is_dec"
MX     = "mX"
SCORE  = "tv_score_v4 + tv_score_v5"

# Exposure variable sets (lag1 versions preferred per memo)
CUM_VARS  = "cum_pos_rel_lag1 + cum_neg_rel_lag1"
DOSE_VARS = "dose_pos_rel_lag1 + dose_neg_rel_lag1"
NET_ABS   = "cum_net_rel_lag1 + cum_abs_rel_lag1"

hazard_specs = [
    ("H1", f"event ~ {TIME} + {REGIME}",
     "time + regime"),
    ("H2", f"event ~ {TIME} + {REGIME} + {EVENT}",
     "time + regime + event-actions"),
    ("H3", f"event ~ {TIME} + {REGIME} + {EVENT} + {MX}",
     "time + regime + event-actions + m(X)"),
    ("H4", f"event ~ {TIME} + {REGIME} + {EVENT} + {MX} + {SCORE}",
     "time + regime + event-actions + m(X) + scores"),
    ("H5", f"event ~ {TIME} + {REGIME} + {MX} + {CUM_VARS}",
     "time + regime + m(X) + cumulative lag1"),
    ("H6", f"event ~ {TIME} + {REGIME} + {MX} + {DOSE_VARS}",
     "time + regime + m(X) + dose lag1"),
    ("H7", f"event ~ {TIME} + {REGIME} + {EVENT} + {MX} + {CUM_VARS}",
     "time + regime + event-actions + m(X) + cumulative lag1"),
    ("H8", f"event ~ {TIME} + {REGIME} + {EVENT} + {MX} + {DOSE_VARS}",
     "time + regime + event-actions + m(X) + dose lag1"),
    ("H9", f"event ~ {TIME} + {REGIME} + {MX} + {NET_ABS}",
     "time + regime + m(X) + net/abs lag1"),
]

# Ensure required columns are filled for GLM
for col in ["cum_pos_rel_lag1","cum_neg_rel_lag1","cum_net_rel_lag1","cum_abs_rel_lag1",
            "dose_pos_rel_lag1","dose_neg_rel_lag1","dose_net_rel_lag1","dose_abs_rel_lag1",
            "tv_score_v4","tv_score_v5"]:
    if col in cp.columns:
        cp[col] = cp[col].fillna(0.0)

hazard_models = {}
for spec_id, formula, label in hazard_specs:
    m = safe_glm(formula, cp, cluster_col="user_id")
    if m:
        hazard_models[spec_id] = (m, label)
        stata_table(m, f"[{spec_id}] {label}")

# ── Summary table ─────────────────────────────────────────────────────────
KEY_VARS = ["is_v5", "tv_is_inc", "tv_is_dec", "mX",
            "cum_pos_rel_lag1", "cum_neg_rel_lag1",
            "dose_pos_rel_lag1", "dose_neg_rel_lag1",
            "cum_net_rel_lag1", "cum_abs_rel_lag1"]

summary_rows = []
for spec_id, (m, label) in hazard_models.items():
    row = {"spec": spec_id, "label": label[:40]}
    for v in KEY_VARS:
        if v in m.params.index:
            row[v] = f"{m.params[v]:+.3f}{stars(m.pvalues[v])}"
        else:
            row[v] = "—"
    row["N"]   = f"{int(m.nobs):,}"
    row["AIC"] = f"{m.aic:.0f}"
    summary_rows.append(row)

df_hazard_summary = pd.DataFrame(summary_rows).set_index("spec")
print("\n  === Hazard Model Summary Table ===")
display(df_hazard_summary)

# Evidence: regime effect stability
regime_coefs = {sid: m.params.get("is_v5", np.nan)
                for sid, (m, _) in hazard_models.items()}
regime_ps    = {sid: m.pvalues.get("is_v5", np.nan)
                for sid, (m, _) in hazard_models.items()}
n_regime_sig = sum(p < CFG["alpha"] for p in regime_ps.values() if not np.isnan(p))

EV["hazard_regime_positive"]  = dict(
    hypothesis = "Regime effect (is_v5) is positive in hazard models",
    tested     = len(regime_coefs) > 0,
    supported  = (np.nanmean(list(regime_coefs.values())) > 0 and
                  n_regime_sig >= len(hazard_models) // 2),
    detail     = (f"is_v5 coefs: {[f'{v:.3f}' for v in regime_coefs.values()]}  "
                  f"{n_regime_sig}/{len(hazard_models)} sig at α={CFG['alpha']}"),
)

# Event-level action stability across specs
inc_coefs = {sid: m.params.get("tv_is_inc", np.nan)
             for sid, (m, _) in hazard_models.items()
             if "tv_is_inc" in m.params.index}
inc_ps    = {sid: m.pvalues.get("tv_is_inc", np.nan) for sid in inc_coefs}
n_inc_sig = sum(p < CFG["alpha"] for p in inc_ps.values() if not np.isnan(p))

EV["hazard_event_inc_sig"]    = dict(
    hypothesis = "Event-level tv_is_inc has independent hazard effect",
    tested     = len(inc_coefs) > 0,
    supported  = (n_inc_sig >= max(1, len(inc_coefs) // 2)),
    detail     = (f"tv_is_inc {n_inc_sig}/{len(inc_coefs)} specs sig; "
                  f"mean coef={np.nanmean(list(inc_coefs.values())):+.3f}"),
)

# Cumulative exposure independent effect
cum_pos_ps = {sid: m.pvalues.get("cum_pos_rel_lag1", np.nan)
              for sid, (m, _) in hazard_models.items()
              if "cum_pos_rel_lag1" in m.params.index}
n_cum_sig  = sum(p < CFG["alpha"] for p in cum_pos_ps.values() if not np.isnan(p))
EV["hazard_cum_exposure_sig"] = dict(
    hypothesis = "Cumulative positive exposure (lag1) has independent hazard effect",
    tested     = len(cum_pos_ps) > 0,
    supported  = (n_cum_sig >= max(1, len(cum_pos_ps) // 2)),
    detail     = f"cum_pos_rel_lag1 {n_cum_sig}/{len(cum_pos_ps)} specs sig",
)

# Visualize regime + inc coefficients across specs
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, var, title in [
    (axes[0], "is_v5",    "Regime effect (is_v5) across specs"),
    (axes[1], "tv_is_inc","Inc action effect (tv_is_inc) across specs"),
]:
    coefs = []; ses = []; labels_ = []
    for sid, (m, lbl) in hazard_models.items():
        if var in m.params.index:
            coefs.append(m.params[var]); ses.append(m.bse[var])
            labels_.append(f"{sid}\n{lbl[:20]}")
    if coefs:
        y_pos = range(len(coefs))
        ax.barh(y_pos, coefs, xerr=ses, capsize=4, alpha=.7)
        ax.axvline(0, c="k", lw=.8)
        ax.set_yticks(y_pos); ax.set_yticklabels(labels_, fontsize=7)
        ax.set(title=title, xlabel="Coefficient")
fig.tight_layout(); plt.show()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TASK 4: EXTRA-INC PROPENSITY & LOCAL EFFECT                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
if CFG["run_extra_inc"]:
    print("\n" + "="*78)
    print("  TASK 4: EXTRA-INC PROPENSITY & LOCAL EFFECT")
    print("="*78)

    # ── 4A: First-stage inc propensity model ──────────────────────────────
    print("\n  ── 4A: Inc propensity model ──────────────────────────────────────────")

    # Features: history covariates + baseline + score-related + regime
    inc_feats_base = [f for f in feat_mX_us if f in us.columns]
    inc_feats_all  = inc_feats_base + ["is_v5", "score_v4", "score_v5", "lc_z",
                                        "pre_n_inc", "pre_n_dec", "pre_cum_dr"]
    inc_feats_all  = [f for f in inc_feats_all if f in us.columns]

    oof_inc, auc_inc, std_inc, n_inc_feat = xgb_crossfit(
        us, inc_feats_all, "is_inc", CFG,
        group_col=C["user_id"], mcw=10, num_rounds=300,
    )
    us["p_inc_oof"] = oof_inc
    print(f"  Inc propensity OOF AUC: {auc_inc:.4f} ± {std_inc:.4f}  "
          f"(n_features={n_inc_feat})")

    # ── 4B: Counterfactual Δp = P(inc|V5) - P(inc|V4) ────────────────────
    print("\n  ── 4B: Counterfactual delta_p construction ───────────────────────────")
    # Train full model including is_v5 as a feature
    # Then predict with is_v5=0 and is_v5=1 for every user
    if "is_v5" in inc_feats_all:
        cf_feats = [f for f in inc_feats_all if f in us.columns]
        cf_X     = us[cf_feats].values.astype(np.float32)
        cf_y     = us["is_inc"].values.astype(int)

        xgb_params_cf = dict(
            objective        = "binary:logistic",
            eval_metric      = "logloss",
            max_depth        = 4,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.6,
            min_child_weight = 10,
            reg_alpha        = 0.1,
            reg_lambda       = 1.0,
            device           = CFG.get("xgb_device", "cpu"),
            verbosity        = 0,
        )
        dt_full  = xgb.DMatrix(cf_X, label=cf_y, feature_names=cf_feats, missing=np.nan)
        bst_cf   = xgb.train(xgb_params_cf, dt_full, num_boost_round=300,
                              verbose_eval=False)

        is_v5_idx = cf_feats.index("is_v5")
        X_v4 = cf_X.copy(); X_v4[:, is_v5_idx] = 0
        X_v5 = cf_X.copy(); X_v5[:, is_v5_idx] = 1
        dm_v4 = xgb.DMatrix(X_v4, feature_names=cf_feats, missing=np.nan)
        dm_v5 = xgb.DMatrix(X_v5, feature_names=cf_feats, missing=np.nan)

        us["p_inc_if_v4"] = bst_cf.predict(dm_v4)
        us["p_inc_if_v5"] = bst_cf.predict(dm_v5)
        us["delta_p_inc"] = us["p_inc_if_v5"] - us["p_inc_if_v4"]
        us["delta_p_inc_z"] = (
            (us["delta_p_inc"] - us["delta_p_inc"].mean())
            / us["delta_p_inc"].std().clip(lower=1e-8)
        )

        print(f"  Mean p_inc_if_v4  : {us['p_inc_if_v4'].mean():.4f}")
        print(f"  Mean p_inc_if_v5  : {us['p_inc_if_v5'].mean():.4f}")
        print(f"  Mean delta_p_inc  : {us['delta_p_inc'].mean():+.4f}")
        print(f"  Pr(delta_p > 0)   : {(us['delta_p_inc'] > 0).mean():.3f}")

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].hist(us["delta_p_inc"], bins=50, alpha=.8)
        axes[0].axvline(0, c="k", lw=.8); axes[0].axvline(us["delta_p_inc"].mean(), c="r", lw=1.5)
        axes[0].set(title="Δp_inc distribution", xlabel="p_inc_if_v5 - p_inc_if_v4")

        axes[1].scatter(us["p_inc_if_v4"], us["p_inc_if_v5"],
                        alpha=.1, s=4, c=us["is_v5"], cmap="coolwarm")
        axes[1].plot([0,1],[0,1],"k--",alpha=.3)
        axes[1].set(xlabel="p_inc_if_v4", ylabel="p_inc_if_v5",
                    title="Counterfactual propensities")

        axes[2].scatter(us["score_v4"], us["delta_p_inc"],
                        alpha=.15, s=4, c=us["is_v5"], cmap="coolwarm")
        axes[2].axhline(0, c="k", lw=.8)
        axes[2].set(xlabel="V4 score", ylabel="delta_p_inc",
                    title="delta_p_inc vs V4 score")
        fig.tight_layout(); plt.show()

        # ── 4C: Local-effect hazard models ──────────────────────────────────
        print("\n  ── 4C: Local-effect hazard models ────────────────────────────────")
        # Merge delta_p into cp
        cp_le = cp.merge(us[["user_id", "delta_p_inc", "delta_p_inc_z"]],
                         on="user_id", how="left")
        cp_le["delta_p_inc_z"] = cp_le["delta_p_inc_z"].fillna(0.0)

        le_specs = [
            ("LE1", "event ~ t_mid + t_mid_sq + is_v5 + mX + delta_p_inc_z",
             "regime + m(X) + delta_p"),
            ("LE2", ("event ~ t_mid + t_mid_sq + is_v5 + mX + tv_is_inc "
                     "+ delta_p_inc_z + tv_is_inc:delta_p_inc_z"),
             "regime + m(X) + inc + delta_p + inc×delta_p"),
            ("LE3", ("event ~ t_mid + t_mid_sq + is_v5 + mX + tv_is_inc + tv_is_dec "
                     "+ delta_p_inc_z + tv_is_inc:delta_p_inc_z + tv_is_dec:delta_p_inc_z"),
             "regime + m(X) + inc + dec + delta_p + interactions"),
        ]

        le_models = {}
        for spec_id, formula, label in le_specs:
            m = safe_glm(formula, cp_le, cluster_col="user_id")
            if m:
                le_models[spec_id] = (m, label)
                stata_table(m, f"[{spec_id}] {label}")

        # Evidence: local effect (inc × delta_p interaction)
        for spec_id in ["LE2", "LE3"]:
            if spec_id in le_models:
                m_le, _ = le_models[spec_id]
                inter_var = "tv_is_inc:delta_p_inc_z"
                if inter_var in m_le.params.index:
                    ic = m_le.params[inter_var]
                    ip = m_le.pvalues[inter_var]
                    EV[f"local_effect_{spec_id}"] = dict(
                        hypothesis = ("Inc effect is stronger for high-delta_p users "
                                     f"(inc × delta_p, {spec_id})"),
                        tested     = True,
                        supported  = (ip < CFG["alpha"]),
                        detail     = (
                            f"tv_is_inc:delta_p coef={ic:.4f} p={ip:.4f} {stars(ip)}"
                            + ("  [no evidence that inc effect is stronger for high-delta_p users]"
                               if ip >= CFG["alpha"] else "")
                        ),
                    )
                    print(f"\n  Local effect [{spec_id}]: {EV[f'local_effect_{spec_id}']['detail']}")

        # V5 raises inc propensity (direction test)
        EV["v5_raises_inc_propensity"] = dict(
            hypothesis = "V5 raises inc propensity vs V4 (delta_p > 0 on average)",
            tested     = True,
            supported  = (us["delta_p_inc"].mean() > 0),
            detail     = (f"mean delta_p={us['delta_p_inc'].mean():+.4f}  "
                         f"Pr(delta_p>0)={(us['delta_p_inc']>0).mean():.3f}"),
        )
    else:
        print("  [SKIP] is_v5 not in inc_feats_all; skipping counterfactual step.")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FINAL SUMMARY — EVIDENCE-LAYERED                                         ║
# ║  All flags are derived from actual statistical results above.             ║
# ║  No flag is hardcoded to True.                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n" + "═"*78)
print("  FINAL SUMMARY — EVIDENCE-LAYERED ASSESSMENT")
print("═"*78)
print("  (based on continuing-user sample, overlap window "
      f"{CFG['overlap_start'].date()} → {CFG['overlap_end'].date()})")
print("  NOTE: results pertain to this selected sample; do not extrapolate to all users.")
print("  NOTE (memo §7.10): time-varying treatment is event-level in this version.")
print("  NOTE (memo §7.9) : default proxy = event_date; no explicit repayment date.")
print()

# Group evidence into categories
SUPPORTED          = []
PARTIALLY          = []
NOT_SUPPORTED      = []
NOT_ESTABLISHED    = []

for key, ev in EV.items():
    if not ev["tested"]:
        NOT_ESTABLISHED.append(ev)
    elif ev.get("partially", False):
        PARTIALLY.append(ev)
    elif ev["supported"]:
        SUPPORTED.append(ev)
    else:
        NOT_SUPPORTED.append(ev)

# Additional items that are definitionally not established
NOT_ESTABLISHED += [
    dict(hypothesis = "Full mechanism chain: inc→default (causal)",
         detail     = "Requires cleaner identification than current design allows."),
    dict(hypothesis = "Clean inc causal effect (free of selection)",
         detail     = ("Regime effect first; separating pure treatment effect "
                       "requires further decomposition.")),
    dict(hypothesis = "Principal stratum identification",
         detail     = "Not attempted in current code."),
]

def _print_group(items, header):
    if not items: return
    print(f"  ┌─ {header} {'─'*(70-len(header))}")
    for ev in items:
        print(f"  │  • {ev['hypothesis']}")
        if ev.get("detail"):
            print(f"  │      → {ev['detail']}")
    print(f"  └{'─'*71}")
    print()

_print_group(SUPPORTED,       "1. SUPPORTED")
_print_group(PARTIALLY,       "2. PARTIALLY SUPPORTED")
_print_group(NOT_SUPPORTED,   "3. NOT SUPPORTED")
_print_group(NOT_ESTABLISHED, "4. NOT ESTABLISHED")

print("═"*78)
print("  IMPORTANT CAUTIONS (research memo §9.2)")
print("─"*78)
print("  • Regime effect (is_v5) is the most robustly identified object here.")
print("  • Event-level tv_is_inc may reflect risk-marking / targeting, not pure causation.")
print("  • Cumulative / dose exposures are a first step toward user-path exposure;")
print("    independent explanatory power should be evaluated with caution.")
print("  • V4→V5 shock identifies a joint regime effect, not a pure credit-limit effect.")
print("  • The most plausible locally identified object is: extra-induced-inc users' risk.")
print("═"*78)
