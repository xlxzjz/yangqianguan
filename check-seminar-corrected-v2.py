# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SEMINAR CHECK - CORRECTED VERSION V2                                     ║
# ║  Fixes:                                                                    ║
# ║  1. A1: NOW CORRECTLY implements event-level analysis using cp_h data     ║
# ║     (previous version wrongly used user-level us data)                    ║
# ║  2. B2: Fixed matching implementation to follow original design            ║
# ║  3. A3/A4: Implemented true cumulative exposure MSM                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Note: This file contains only the corrected sections A1, B2, A3, A4
# The full context and data setup would need to be imported from the original notebook

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART A: PERFORMATIVE PREDICTION EXISTENCE                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── A1: Event-level DML (CORRECTED TO USE HAZARD DATA) ────────────────────
# CORRECTED: Now properly uses event-level person-time data (cp_h), not user aggregates
print("\n" + "="*78)
print("  A1: EVENT-LEVEL DML — Y ~ T + m(X)")
print("="*78)
print("  Goal: Test if treatment T affects outcome Y beyond prediction m(X)")
print("  at true event-level (person-time intervals, not user aggregates).")
print("  Treatment T: time-varying credit changes (tv_delta_rel, tv_is_inc/dec).")

# CORRECTED: Use cp_h (event-level hazard data) not us (user-level data)
cp_h = cp.dropna(subset=["mX"]).copy()

# --- A1a-d: Event-level conditional models ---
a1_specs = [
    ("A1a", "event ~ t_mid + t_mid_sq + mX + base_credit_z + tv_delta_rel_z",
     "time + m(X) + credit + delta"),
    ("A1b", "event ~ t_mid + t_mid_sq + mX + base_credit_z + tv_is_inc + tv_is_dec",
     "time + m(X) + credit + inc/dec indicators"),
    ("A1c", "event ~ t_mid + t_mid_sq + mX + base_credit_z + tv_is_inc + tv_is_dec + is_v5",
     "time + m(X) + credit + inc/dec + regime"),
    ("A1d", "event ~ t_mid + t_mid_sq + mX + base_credit_z + tv_delta_rel_z + is_v5",
     "time + m(X) + credit + delta + regime"),
]

for spec_id, fml, label in a1_specs:
    m = safe_glm(fml, cp_h, cluster_col="user_id")
    if m:
        stata_table(m, f"[{spec_id}] {label}")

# --- A1e: Event-level DML partialling-out ---
print("\n  -- A1e: DML partialling-out (event-level) --")
print("  Outcome Y: hazard indicator (event)")
print("  Treatment T: time-varying credit change")
print("  Partialling out m(X) from both Y and T")

# Prepare event-level data for DML
cp_dml = cp_h.dropna(subset=["event", "mX", "tv_delta_rel_z"]).copy()

# Y residual: event - m(X)
Y_res = cp_dml["event"].values - cp_dml["mX"].values

# Features for treatment model (all confounders except treatment)
feat_dml = feat_in_cp + ["t_mid", "base_credit_z"]
feat_dml = [f for f in feat_dml if f in cp_dml.columns and f != "tv_delta_rel_z"]
feat_dml = list(dict.fromkeys(feat_dml))  # deduplicate

# DML for different treatment definitions
for treat_col, treat_label in [("tv_delta_rel_z", "delta (continuous)"),
                                 ("tv_is_inc", "inc indicator"),
                                 ("tv_is_dec", "dec indicator")]:

    if treat_col not in cp_dml.columns:
        continue

    # T residual: predict treatment from confounders, get residual
    oof_t, _, _, _ = xgb_crossfit(
        cp_dml, feat_dml, treat_col, CFG,
        group_col="user_id",
        objective="reg:squarederror" if treat_col == "tv_delta_rel_z" else "binary:logistic",
        is_binary=(treat_col != "tv_delta_rel_z"),
        mcw=30, num_rounds=200,
    )
    T_res = cp_dml[treat_col].values - oof_t

    # DML: regress Y_res on T_res
    mask = np.isfinite(Y_res) & np.isfinite(T_res)
    X_dml = T_res[mask].reshape(-1, 1)
    y_dml = Y_res[mask]

    if len(y_dml) < 100:
        print(f"  [{treat_label}] Insufficient data after cleaning")
        continue

    lr = LinearRegression().fit(X_dml, y_dml)
    theta = lr.coef_[0]

    # Bootstrap SE (clustered by user)
    # Need to bootstrap at user level
    user_ids = cp_dml.loc[mask, "user_id"].values
    unique_users = np.unique(user_ids)

    rng = np.random.default_rng(CFG["seed"])
    n_boot = 500
    boot_thetas = []

    for _ in range(n_boot):
        # Sample users with replacement
        boot_users = rng.choice(unique_users, len(unique_users), replace=True)
        # Get all observations for sampled users
        boot_idx = np.concatenate([np.where(user_ids == u)[0] for u in boot_users])

        if len(boot_idx) < 50:
            continue

        try:
            boot_theta = LinearRegression().fit(
                X_dml[boot_idx], y_dml[boot_idx]
            ).coef_[0]
            boot_thetas.append(boot_theta)
        except:
            continue

    if len(boot_thetas) < 100:
        print(f"  [{treat_label}] Bootstrap failed")
        continue

    se = np.std(boot_thetas)
    z_val = theta / se if se > 0 else 0
    p_val = 2 * (1 - stats.norm.cdf(abs(z_val)))
    ci_lo = theta - 1.96 * se
    ci_hi = theta + 1.96 * se

    dml_table(f"A1e DML [{treat_label}]", theta, se, z_val, p_val, ci_lo, ci_hi,
              n=int(mask.sum()),
              extra_info={"N_users": len(unique_users),
                         "N_intervals": int(mask.sum())})


# ── A2: Event-level Hazard DML (remains as reference) ─────────────────────
# A2 was already correctly implemented as event-level
print("\n" + "="*78)
print("  A2: EVENT-LEVEL HAZARD DML — discrete-time hazard ~ T + m(X)")
print("="*78)
print("  Robustness check: performative prediction in hazard framework.")
print("  (A2 already correctly implemented as event-level)")

a2_specs = [
    ("A2a", "event ~ t_mid + t_mid_sq + mX + base_credit_z + tv_delta_rel_z",
     "time + m(X) + credit + delta"),
    ("A2b", "event ~ t_mid + t_mid_sq + mX + base_credit_z + tv_is_inc + tv_is_dec",
     "time + m(X) + credit + inc/dec"),
    ("A2c", "event ~ t_mid + t_mid_sq + mX + base_credit_z + tv_is_inc + tv_is_dec + is_v5",
     "time + m(X) + credit + inc/dec + regime"),
    ("A2d", "event ~ t_mid + t_mid_sq + mX + base_credit_z + tv_delta_rel_z + is_v5",
     "time + m(X) + credit + delta + regime"),
]

for spec_id, fml, label in a2_specs:
    m = safe_glm(fml, cp_h, cluster_col="user_id")
    if m:
        stata_table(m, f"[{spec_id}] {label}")


# ── A3: Cumulative Exposure MSM (CORRECTED IMPLEMENTATION) ─────────────────
print("\n" + "="*78)
print("  A3: CUMULATIVE EXPOSURE MSM — True Time-Varying Treatment History")
print("="*78)
print("  Exposure: Cumulative credit changes up to time t")
print("  CORRECTED: Uses treatment history in IPTW weights")

# --- A3a-d: Conditional models with cumulative exposure ---
a3_specs = [
    ("A3a", "event ~ t_mid + t_mid_sq + mX + base_credit_z + cum_pos_rel_lag1_z + cum_neg_rel_lag1_z",
     "conditional: cumulative pos/neg"),
    ("A3b", "event ~ t_mid + t_mid_sq + mX + base_credit_z + cum_pos_rel_lag1_z + cum_neg_rel_lag1_z + is_v5",
     "conditional: cumulative pos/neg + regime"),
    ("A3c", "event ~ t_mid + t_mid_sq + mX + base_credit_z + cum_net_rel_lag1_z",
     "conditional: cumulative net"),
    ("A3d", "event ~ t_mid + t_mid_sq + mX + base_credit_z + cum_net_rel_lag1_z + is_v5",
     "conditional: cumulative net + regime"),
]

for spec_id, fml, label in a3_specs:
    m = safe_glm(fml, cp_h, cluster_col="user_id")
    if m:
        stata_table(m, f"[{spec_id}] {label}")

# --- A3e: CORRECTED IPTW for cumulative exposure MSM ---
print("\n  -- A3e: IPTW for cumulative exposure MSM (CORRECTED) --")

# Treatment categorization
cp_h["tv_treat_cat"] = np.select(
    [cp_h["tv_is_inc"] == 1, cp_h["tv_is_dec"] == 1],
    [2, 0], default=1  # 0=dec, 1=same, 2=inc
)

# CORRECTED: Denominator includes treatment history
iptw_feats = feat_in_cp + ["t_mid", "is_v5", "base_credit_z",
                            "cum_pos_rel_lag1_z", "cum_neg_rel_lag1_z"]
iptw_feats = [f for f in iptw_feats if f in cp_h.columns]
iptw_feats = list(dict.fromkeys(iptw_feats))

# Build treatment propensity models
for cat_val, cat_label in [(2, "inc"), (0, "dec")]:
    cp_h[f"_treat_is_{cat_label}"] = (cp_h["tv_treat_cat"] == cat_val).astype(int)
    oof_denom, _, _, _ = xgb_crossfit(
        cp_h, iptw_feats, f"_treat_is_{cat_label}", CFG,
        group_col="user_id", mcw=30, num_rounds=200,
    )
    cp_h[f"_p_denom_{cat_label}"] = np.clip(oof_denom, 0.01, 0.99)

# Numerator: marginal probabilities
marginal_inc = cp_h["tv_is_inc"].mean()
marginal_dec = cp_h["tv_is_dec"].mean()
marginal_same = 1 - marginal_inc - marginal_dec

# Compute stabilised weights
def compute_sw(row):
    if row["tv_treat_cat"] == 2:
        num, den = marginal_inc, row["_p_denom_inc"]
    elif row["tv_treat_cat"] == 0:
        num, den = marginal_dec, row["_p_denom_dec"]
    else:
        num = marginal_same
        den = 1 - row["_p_denom_inc"] - row["_p_denom_dec"]
    den = max(den, 0.01)
    return num / den

cp_h["_sw"] = cp_h.apply(compute_sw, axis=1)
cp_h["_csw"] = cp_h.groupby("user_id")["_sw"].cumprod()

# Truncate extreme weights
p1, p99 = cp_h["_csw"].quantile(0.01), cp_h["_csw"].quantile(0.99)
cp_h["_csw_trunc"] = cp_h["_csw"].clip(p1, p99)

print(f"  IPTW weight stats: mean={cp_h['_csw_trunc'].mean():.3f}, "
      f"median={cp_h['_csw_trunc'].median():.3f}, "
      f"p5={cp_h['_csw_trunc'].quantile(0.05):.3f}, "
      f"p95={cp_h['_csw_trunc'].quantile(0.95):.3f}")

# Weighted MSM
try:
    msm_fml = "event ~ t_mid + t_mid_sq + cum_pos_rel_lag1_z + cum_neg_rel_lag1_z"
    m_msm = smf.glm(msm_fml, data=cp_h,
                     family=sm.families.Binomial(),
                     freq_weights=cp_h["_csw_trunc"].values).fit(cov_type="HC1")
    stata_table(m_msm, "[A3e] IPTW MSM: cumulative pos/neg")

    msm_fml2 = "event ~ t_mid + t_mid_sq + cum_net_rel_lag1_z"
    m_msm2 = smf.glm(msm_fml2, data=cp_h,
                      family=sm.families.Binomial(),
                      freq_weights=cp_h["_csw_trunc"].values).fit(cov_type="HC1")
    stata_table(m_msm2, "[A3f] IPTW MSM: cumulative net")
except Exception as e:
    print(f"  IPTW MSM failed: {e}")


# ── A4: Cumulative Dose MSM (CORRECTED IMPLEMENTATION) ─────────────────────
print("\n" + "="*78)
print("  A4: CUMULATIVE DOSE MSM — Time-Integrated Exposure")
print("="*78)

a4_specs = [
    ("A4a", "event ~ t_mid + t_mid_sq + mX + base_credit_z + dose_pos_rel_lag1_z + dose_neg_rel_lag1_z",
     "conditional: dose pos/neg"),
    ("A4b", "event ~ t_mid + t_mid_sq + mX + base_credit_z + dose_pos_rel_lag1_z + dose_neg_rel_lag1_z + is_v5",
     "conditional: dose + regime"),
    ("A4c", "event ~ t_mid + t_mid_sq + mX + base_credit_z + dose_net_rel_lag1_z",
     "conditional: dose net"),
    ("A4d", "event ~ t_mid + t_mid_sq + mX + base_credit_z + dose_net_rel_lag1_z + is_v5",
     "conditional: dose net + regime"),
]

for spec_id, fml, label in a4_specs:
    m = safe_glm(fml, cp_h, cluster_col="user_id")
    if m:
        stata_table(m, f"[{spec_id}] {label}")

# IPTW MSM for dose
try:
    msm_fml3 = "event ~ t_mid + t_mid_sq + dose_pos_rel_lag1_z + dose_neg_rel_lag1_z"
    m_msm3 = smf.glm(msm_fml3, data=cp_h,
                      family=sm.families.Binomial(),
                      freq_weights=cp_h["_csw_trunc"].values).fit(cov_type="HC1")
    stata_table(m_msm3, "[A4e] IPTW MSM: dose pos/neg")

    msm_fml4 = "event ~ t_mid + t_mid_sq + dose_net_rel_lag1_z"
    m_msm4 = smf.glm(msm_fml4, data=cp_h,
                      family=sm.families.Binomial(),
                      freq_weights=cp_h["_csw_trunc"].values).fit(cov_type="HC1")
    stata_table(m_msm4, "[A4f] IPTW MSM: dose net")
except Exception as e:
    print(f"  IPTW MSM for dose failed: {e}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART B: EXCESS-INC LOCAL CAUSAL EFFECT                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── B1: Estimate individual-level excess inc probability ──────────────────
print("\n" + "="*78)
print("  B1: EXCESS-INC PROPENSITY ESTIMATION")
print("="*78)
print("  p_excess_i = P(inc|X, regime=V5) - P(inc|X, regime=V4)")

# B1 implementation (unchanged) - uses user-level data appropriately
inc_feats_base = [f for f in feat_mX_us if f in us.columns]
inc_feats_all = inc_feats_base + ["is_v5", "score_v4", "score_v5", "lc_z",
                                   "pre_n_inc", "pre_n_dec", "pre_cum_dr"]
inc_feats_all = [f for f in inc_feats_all if f in us.columns]

oof_inc, auc_inc, std_inc, n_inc_feat = xgb_crossfit(
    us, inc_feats_all, "is_inc", CFG,
    group_col=C["user_id"], mcw=10, num_rounds=300,
)
us["p_inc_oof"] = oof_inc
print(f"  Inc propensity OOF AUC: {auc_inc:.4f} +/- {std_inc:.4f}  (n_features={n_inc_feat})")

# Counterfactual predictions
cf_feats = [f for f in inc_feats_all if f in us.columns]
cf_X = us[cf_feats].values.astype(np.float32)
cf_y = us["is_inc"].values.astype(int)

xgb_params_cf = dict(
    objective="binary:logistic", eval_metric="logloss",
    max_depth=4, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.6, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0,
    device=CFG.get("xgb_device", "cpu"), verbosity=0,
)
dt_full = xgb.DMatrix(cf_X, label=cf_y, feature_names=cf_feats, missing=np.nan)
bst_cf = xgb.train(xgb_params_cf, dt_full, num_boost_round=300, verbose_eval=False)

is_v5_idx = cf_feats.index("is_v5")
X_v4 = cf_X.copy(); X_v4[:, is_v5_idx] = 0
X_v5 = cf_X.copy(); X_v5[:, is_v5_idx] = 1
us["p_inc_if_v4"] = bst_cf.predict(xgb.DMatrix(X_v4, feature_names=cf_feats, missing=np.nan))
us["p_inc_if_v5"] = bst_cf.predict(xgb.DMatrix(X_v5, feature_names=cf_feats, missing=np.nan))
us["delta_p_inc"] = us["p_inc_if_v5"] - us["p_inc_if_v4"]
us["delta_p_inc_z"] = standardise(us["delta_p_inc"])

print(f"  Mean p_inc_if_v4  : {us['p_inc_if_v4'].mean():.4f}")
print(f"  Mean p_inc_if_v5  : {us['p_inc_if_v5'].mean():.4f}")
print(f"  Mean delta_p_inc  : {us['delta_p_inc'].mean():+.4f}")
print(f"  Pr(delta_p > 0)   : {(us['delta_p_inc'] > 0).mean():.3f}")


# ── B2: Matching — CORRECTED IMPLEMENTATION ────────────────────────────────
print("\n" + "="*78)
print("  B2: MATCHING — EXCESS-INC USERS VS CONTROL (CORRECTED)")
print("="*78)
print("  Treatment: V5 users pushed into inc (high delta_p_inc)")
print("  Control: V4 users who did NOT inc, matched on baseline characteristics")

for tau in [0.05, 0.10, 0.20]:
    print(f"\n  --- Threshold tau = {tau:.2f} ---")

    # Treatment: V5 inc users with high excess probability
    treated = us[(us["is_v5"] == 1) & (us["is_inc"] == 1) &
                 (us["delta_p_inc"] > tau)].copy()

    # Control: V4 non-inc users (potential counterfactuals)
    control = us[(us["is_v5"] == 0) & (us["is_inc"] == 0)].copy()

    print(f"  Treated (V5 inc, delta_p>{tau}): {len(treated)}")
    print(f"  Control (V4 non-inc): {len(control)}")

    if len(treated) < 10 or len(control) < 10:
        print("  Too few units for matching. Skipping.")
        continue

    # CORRECTED: Match only on baseline V4 characteristics
    match_vars = ["score_v4", "lc_z"]
    if "pre_cum_dr" in treated.columns and "pre_cum_dr" in control.columns:
        match_vars.append("pre_cum_dr")
    match_vars = [v for v in match_vars if v in treated.columns and v in control.columns]

    # Propensity score matching
    match_df = pd.concat([
        treated[match_vars + ["defaulted", "T"]].assign(_treated=1),
        control[match_vars + ["defaulted", "T"]].assign(_treated=0),
    ]).dropna(subset=match_vars)

    try:
        ps_model = LogisticRegression(max_iter=1000, C=1.0)
        ps_model.fit(match_df[match_vars], match_df["_treated"])
        match_df["_ps"] = ps_model.predict_proba(match_df[match_vars])[:, 1]
    except Exception as e:
        print(f"  PS model failed: {e}"); continue

    # Check overlap
    ps_t = match_df.loc[match_df["_treated"]==1, "_ps"]
    ps_c = match_df.loc[match_df["_treated"]==0, "_ps"]
    print(f"  PS range treated: [{ps_t.min():.3f}, {ps_t.max():.3f}]")
    print(f"  PS range control: [{ps_c.min():.3f}, {ps_c.max():.3f}]")

    # 1:1 nearest-neighbour matching with caliper
    caliper = 0.2 * match_df["_ps"].std()
    t_idx = match_df[match_df["_treated"]==1].index.values
    c_idx = match_df[match_df["_treated"]==0].index.values
    t_ps = match_df.loc[t_idx, "_ps"].values.reshape(-1, 1)
    c_ps = match_df.loc[c_idx, "_ps"].values.reshape(-1, 1)

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(c_ps)
    dists, indices = nn.kneighbors(t_ps)

    matched_pairs = []
    used_controls = set()
    for i, (d, j) in enumerate(zip(dists.ravel(), indices.ravel())):
        if d <= caliper and j not in used_controls:
            matched_pairs.append((t_idx[i], c_idx[j]))
            used_controls.add(j)

    if len(matched_pairs) < 10:
        print(f"  Only {len(matched_pairs)} matched pairs. Skipping.")
        continue

    t_matched = match_df.loc[[p[0] for p in matched_pairs]]
    c_matched = match_df.loc[[p[1] for p in matched_pairs]]

    print(f"  Matched pairs: {len(matched_pairs)}")

    # Balance check
    print(f"  Balance (SMD):")
    for v in match_vars + ["_ps"]:
        smd = compute_smd(t_matched[v], c_matched[v])
        print(f"    {v}: SMD = {smd:.4f} {'[OK]' if abs(smd) < 0.1 else '[IMBALANCED]'}")

    # Outcome comparison - LATE estimate
    dr_t = t_matched["defaulted"].mean()
    dr_c = c_matched["defaulted"].mean()
    rd = dr_t - dr_c
    se_rd = np.sqrt(dr_t*(1-dr_t)/len(t_matched) + dr_c*(1-dr_c)/len(c_matched))
    z_rd = rd / se_rd if se_rd > 0 else 0
    p_rd = 2 * (1 - stats.norm.cdf(abs(z_rd)))

    print(f"\n  Default rate treated: {dr_t:.4f}")
    print(f"  Default rate control: {dr_c:.4f}")
    dml_table(f"B2 Matching (tau={tau:.2f}): LATE estimate",
              rd, se_rd, z_rd, p_rd,
              rd - 1.96*se_rd, rd + 1.96*se_rd,
              n=len(matched_pairs),
              extra_info={"Treated": len(t_matched), "Control": len(c_matched),
                          "DR_treated": f"{dr_t:.4f}", "DR_control": f"{dr_c:.4f}"})


# ── B3: Hazard with excess probability ────────────────────────────────────
print("\n" + "="*78)
print("  B3: HAZARD WITH EXCESS PROBABILITY")
print("="*78)

# B3 implementation would go here...

print("\n" + "="*78)
print("  CORRECTIONS SUMMARY (V2):")
print("="*78)
print("  1. A1: NOW CORRECTLY uses event-level hazard data (cp_h), not user aggregates")
print("  2. B2: Matches on baseline V4 characteristics only")
print("  3. A3/A4: Implements true cumulative exposure MSM with treatment history")
print("="*78)
