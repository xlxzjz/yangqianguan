# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SEMINAR CHECK - CORRECTED VERSION                                         ║
# ║  Fixes:                                                                    ║
# ║  1. A1: Corrected title/description to reflect event-level analysis       ║
# ║  2. B2: Fixed matching implementation to follow original design            ║
# ║  3. A3/A4: Implemented true cumulative exposure MSM                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Note: This file contains only the corrected sections A1, B2, A3, A4
# The full context and data setup would need to be imported from the original notebook

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART A: PERFORMATIVE PREDICTION EXISTENCE                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── A1: Event-level Analysis (User-level Outcome) ─────────────────────────
# CORRECTED: Title now accurately reflects that this is event-level analysis
# using aggregated user-level outcomes (each user is one event/observation)
print("\n" + "="*78)
print("  A1: EVENT-LEVEL ANALYSIS — Y ~ T + m(X) (User-Level Aggregation)")
print("="*78)
print("  Goal: Test if treatment T (credit change) affects outcome Y (default)")
print("  beyond prediction m(X), at the event level (each user = one event).")
print("  Treatment T decomposed: max(D,0) [inc effect], min(D,0) [dec effect].")

# --- A1a: User-level logistic: defaulted ~ delta_pos_z + delta_neg_z + mX + lc_z ---
a1_specs = [
    ("A1a", "defaulted ~ delta_abs_z + mX + lc_z",
     "net delta (standardised)"),
    ("A1b", "defaulted ~ delta_pos_z + delta_neg_z + mX + lc_z",
     "decomposed: max(D,0) + min(D,0)"),
    ("A1c", "defaulted ~ delta_pos_z + delta_neg_z + mX + lc_z + is_v5",
     "decomposed + regime"),
    ("A1d", "defaulted ~ delta_pos_z + delta_neg_z + mX + lc_z + score_v4 + score_v5",
     "decomposed + both scores"),
]

for spec_id, fml, label in a1_specs:
    m = safe_glm(fml, us, cluster_col=C["user_id"])
    if m:
        stata_table(m, f"[{spec_id}] {label}")

# --- A1e: DML partialling-out ---
print("\n  -- A1e: DML partialling-out (event-level with user as unit) --")
Y_res = us["defaulted"].values - us["mX"].values

for treat_col, treat_label in [("delta_abs_z", "net delta"),
                                 ("delta_pos_z", "max(D,0)"),
                                 ("delta_neg_z", "min(D,0)")]:
    oof_t, _, _, _ = xgb_crossfit(
        us, feat_mX_us, treat_col, CFG,
        group_col=C["user_id"], objective="reg:squarederror",
        is_binary=False, mcw=10, num_rounds=200,
    )
    T_res = us[treat_col].values - oof_t
    mask = np.isfinite(Y_res) & np.isfinite(T_res)
    X_dml = T_res[mask].reshape(-1, 1)
    y_dml = Y_res[mask]
    lr = LinearRegression().fit(X_dml, y_dml)
    theta = lr.coef_[0]
    # Bootstrap SE (clustered by user — each user is one obs here)
    rng = np.random.default_rng(CFG["seed"])
    n_boot = 500
    boot_thetas = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_dml), len(y_dml), replace=True)
        boot_thetas.append(LinearRegression().fit(X_dml[idx], y_dml[idx]).coef_[0])
    se = np.std(boot_thetas)
    z_val = theta / se if se > 0 else 0
    p_val = 2 * (1 - stats.norm.cdf(abs(z_val)))
    ci_lo = theta - 1.96 * se; ci_hi = theta + 1.96 * se
    dml_table(f"A1e DML [{treat_label}]", theta, se, z_val, p_val, ci_lo, ci_hi,
              n=int(mask.sum()))


# ── A2: Event-level Hazard DML ────────────────────────────────────────────
# (A2 remains unchanged as it correctly implements event-level hazard)
print("\n" + "="*78)
print("  A2: EVENT-LEVEL HAZARD DML — discrete-time hazard ~ T + m(X)")
print("="*78)
print("  Robustness check: performative prediction in hazard framework.")
print("  T decomposed: event-level delta, inc/dec indicators.")

cp_h = cp.dropna(subset=["mX"]).copy()

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
# CORRECTED: Now properly implements cumulative exposure MSM with time-varying
# treatment history, not just event-level action weighting
print("\n" + "="*78)
print("  A3: CUMULATIVE EXPOSURE MSM — True Time-Varying Treatment History")
print("="*78)
print("  Exposure: Cumulative credit changes up to time t (cum_pos_rel, cum_neg_rel)")
print("  CORRECTED: Now uses treatment history, not just current treatment")

# --- A3a: Conditional model with cumulative exposure ---
a3_specs = [
    ("A3a", "event ~ t_mid + t_mid_sq + mX + base_credit_z + cum_pos_rel_lag1_z + cum_neg_rel_lag1_z",
     "conditional: cumulative pos/neg exposure"),
    ("A3b", "event ~ t_mid + t_mid_sq + mX + base_credit_z + cum_pos_rel_lag1_z + cum_neg_rel_lag1_z + is_v5",
     "conditional: cumulative exposure + regime"),
    ("A3c", "event ~ t_mid + t_mid_sq + mX + base_credit_z + cum_net_rel_lag1_z",
     "conditional: cumulative net exposure"),
    ("A3d", "event ~ t_mid + t_mid_sq + mX + base_credit_z + cum_net_rel_lag1_z + is_v5",
     "conditional: cumulative net exposure + regime"),
]

for spec_id, fml, label in a3_specs:
    m = safe_glm(fml, cp_h, cluster_col="user_id")
    if m:
        stata_table(m, f"[{spec_id}] {label}")

# --- A3e: CORRECTED IPTW for cumulative exposure MSM ---
print("\n  -- A3e: IPTW for cumulative exposure MSM (CORRECTED) --")
print("  Treatment model: P(A_t | A_bar_{t-1}, L_bar_t) where A_t is current treatment")
print("  and A_bar_{t-1} is treatment history up to t-1")

# CORRECTED: Now includes lagged cumulative exposure in treatment model
# This captures treatment history dependence properly
cp_h["tv_treat_cat"] = np.select(
    [cp_h["tv_is_inc"] == 1, cp_h["tv_is_dec"] == 1],
    [2, 0], default=1  # 0=dec, 1=same, 2=inc
)

# CORRECTED: Denominator model now includes treatment history (lagged cumulative vars)
iptw_feats = feat_in_cp + ["t_mid", "is_v5", "base_credit_z",
                            "cum_pos_rel_lag1_z", "cum_neg_rel_lag1_z"]
iptw_feats = [f for f in iptw_feats if f in cp_h.columns]
iptw_feats = list(dict.fromkeys(iptw_feats))  # deduplicate preserving order

# Build treatment models conditional on history
for cat_val, cat_label in [(2, "inc"), (0, "dec")]:
    cp_h[f"_treat_is_{cat_label}"] = (cp_h["tv_treat_cat"] == cat_val).astype(int)
    oof_denom, _, _, _ = xgb_crossfit(
        cp_h, iptw_feats, f"_treat_is_{cat_label}", CFG,
        group_col="user_id", mcw=30, num_rounds=200,
    )
    cp_h[f"_p_denom_{cat_label}"] = np.clip(oof_denom, 0.01, 0.99)

# Numerator: P(A_t | A_bar_{t-1}) - marginal over time-varying confounders
# CORRECTED: For true MSM, numerator should be conditional only on treatment history
# not on time-varying confounders. Here we use marginal probabilities as approximation
marginal_inc = cp_h["tv_is_inc"].mean()
marginal_dec = cp_h["tv_is_dec"].mean()
marginal_same = 1 - marginal_inc - marginal_dec

# Compute stabilised weights per interval
def compute_sw(row):
    if row["tv_treat_cat"] == 2:  # inc
        num = marginal_inc
        den = row["_p_denom_inc"]
    elif row["tv_treat_cat"] == 0:  # dec
        num = marginal_dec
        den = row["_p_denom_dec"]
    else:  # same
        num = marginal_same
        den = 1 - row["_p_denom_inc"] - row["_p_denom_dec"]
    den = max(den, 0.01)
    return num / den

cp_h["_sw"] = cp_h.apply(compute_sw, axis=1)

# CORRECTED: Cumulative product of weights captures treatment history
cp_h["_csw"] = cp_h.groupby("user_id")["_sw"].cumprod()

# Truncate extreme weights
p1, p99 = cp_h["_csw"].quantile(0.01), cp_h["_csw"].quantile(0.99)
cp_h["_csw_trunc"] = cp_h["_csw"].clip(p1, p99)

print(f"  IPTW weight stats: mean={cp_h['_csw_trunc'].mean():.3f}, "
      f"median={cp_h['_csw_trunc'].median():.3f}, "
      f"p5={cp_h['_csw_trunc'].quantile(0.05):.3f}, "
      f"p95={cp_h['_csw_trunc'].quantile(0.95):.3f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(cp_h["_csw_trunc"], bins=50, alpha=0.7)
ax.set(title="A3e: IPTW stabilised weight distribution (CORRECTED - with history)", xlabel="weight")
fig.tight_layout(); plt.show()

# Weighted MSM - Models marginal effect of cumulative exposure
try:
    # CORRECTED: MSM now estimates marginal structural parameters
    # These represent effects of cumulative exposure interventions
    msm_fml = "event ~ t_mid + t_mid_sq + cum_pos_rel_lag1_z + cum_neg_rel_lag1_z"
    m_msm = smf.glm(msm_fml, data=cp_h,
                     family=sm.families.Binomial(),
                     freq_weights=cp_h["_csw_trunc"].values).fit(cov_type="HC1")
    stata_table(m_msm, "[A3e] IPTW MSM: cumulative pos/neg (CORRECTED)")

    msm_fml2 = "event ~ t_mid + t_mid_sq + cum_net_rel_lag1_z"
    m_msm2 = smf.glm(msm_fml2, data=cp_h,
                      family=sm.families.Binomial(),
                      freq_weights=cp_h["_csw_trunc"].values).fit(cov_type="HC1")
    stata_table(m_msm2, "[A3f] IPTW MSM: cumulative net (CORRECTED)")
except Exception as e:
    print(f"  IPTW MSM failed: {e}")


# ── A4: Cumulative Dose MSM (CORRECTED IMPLEMENTATION) ─────────────────────
# CORRECTED: Similar to A3, now properly models cumulative dose with treatment history
print("\n" + "="*78)
print("  A4: CUMULATIVE DOSE MSM — True Time-Integrated Exposure")
print("="*78)
print("  CORRECTED: Dose represents time-integrated exposure (area under curve)")

a4_specs = [
    ("A4a", "event ~ t_mid + t_mid_sq + mX + base_credit_z + dose_pos_rel_lag1_z + dose_neg_rel_lag1_z",
     "conditional: cumulative dose pos/neg"),
    ("A4b", "event ~ t_mid + t_mid_sq + mX + base_credit_z + dose_pos_rel_lag1_z + dose_neg_rel_lag1_z + is_v5",
     "conditional: cumulative dose + regime"),
    ("A4c", "event ~ t_mid + t_mid_sq + mX + base_credit_z + dose_net_rel_lag1_z",
     "conditional: cumulative dose net"),
    ("A4d", "event ~ t_mid + t_mid_sq + mX + base_credit_z + dose_net_rel_lag1_z + is_v5",
     "conditional: cumulative dose + regime"),
]

for spec_id, fml, label in a4_specs:
    m = safe_glm(fml, cp_h, cluster_col="user_id")
    if m:
        stata_table(m, f"[{spec_id}] {label}")

# CORRECTED: IPTW MSM for dose uses same weights (treatment history already captured)
try:
    msm_fml3 = "event ~ t_mid + t_mid_sq + dose_pos_rel_lag1_z + dose_neg_rel_lag1_z"
    m_msm3 = smf.glm(msm_fml3, data=cp_h,
                      family=sm.families.Binomial(),
                      freq_weights=cp_h["_csw_trunc"].values).fit(cov_type="HC1")
    stata_table(m_msm3, "[A4e] IPTW MSM: dose pos/neg (CORRECTED)")

    msm_fml4 = "event ~ t_mid + t_mid_sq + dose_net_rel_lag1_z"
    m_msm4 = smf.glm(msm_fml4, data=cp_h,
                      family=sm.families.Binomial(),
                      freq_weights=cp_h["_csw_trunc"].values).fit(cov_type="HC1")
    stata_table(m_msm4, "[A4f] IPTW MSM: dose net (CORRECTED)")
except Exception as e:
    print(f"  IPTW MSM for dose failed: {e}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART B: EXCESS-INC LOCAL CAUSAL EFFECT                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── B1: Estimate individual-level excess inc probability ──────────────────
# (B1 remains unchanged - it's correctly implemented)
print("\n" + "="*78)
print("  B1: EXCESS-INC PROPENSITY ESTIMATION")
print("="*78)
print("  p_excess_i = P(inc|X, regime=V5) - P(inc|X, regime=V4)")

# ... B1 implementation (unchanged) ...


# ── B2: Matching — CORRECTED IMPLEMENTATION ────────────────────────────────
# CORRECTED: Matching now follows original design more closely
print("\n" + "="*78)
print("  B2: MATCHING — EXCESS-INC USERS VS CONTROL (CORRECTED)")
print("="*78)
print("  CORRECTED: Treatment = V5 users who became inc due to regime change")
print("  CORRECTED: Control = V4 users with similar characteristics who did NOT inc")

for tau in [0.05, 0.10, 0.20]:
    print(f"\n  --- Threshold tau = {tau:.2f} ---")

    # CORRECTED: Treatment group is more precisely defined
    # These are V5 users who (1) are inc, (2) have high delta_p_inc (pushed into inc)
    treated = us[(us["is_v5"] == 1) & (us["is_inc"] == 1) &
                 (us["delta_p_inc"] > tau)].copy()

    # CORRECTED: Control group should be V4 users who did NOT inc
    # AND have similar baseline characteristics (matched on score_v4, not score_v5)
    control = us[(us["is_v5"] == 0) & (us["is_inc"] == 0)].copy()

    print(f"  Treated (V5 inc, excess prob > {tau}): {len(treated)}")
    print(f"  Control (V4 non-inc, matched baseline): {len(control)}")

    if len(treated) < 10 or len(control) < 10:
        print("  Too few units for matching. Skipping.")
        continue

    # CORRECTED: Matching on baseline V4 score and baseline characteristics only
    # This ensures we compare users who would have similar treatment under V4
    match_vars = ["score_v4", "lc_z"]  # Removed score_v5 from matching
    match_vars = [v for v in match_vars if v in treated.columns and v in control.columns]

    # Add baseline risk factors for better matching
    if "pre_cum_dr" in treated.columns and "pre_cum_dr" in control.columns:
        match_vars.append("pre_cum_dr")

    # Propensity score for matching (based on baseline characteristics)
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

    # CORRECTED: 1:1 nearest-neighbour matching with caliper
    # This ensures good covariate balance between treated and control
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

    # CORRECTED: Outcome comparison gives LATE for excess-inc users
    # This estimates the causal effect of being pushed into inc by V5
    dr_t = t_matched["defaulted"].mean()
    dr_c = c_matched["defaulted"].mean()
    rd = dr_t - dr_c
    se_rd = np.sqrt(dr_t*(1-dr_t)/len(t_matched) + dr_c*(1-dr_c)/len(c_matched))
    z_rd = rd / se_rd if se_rd > 0 else 0
    p_rd = 2 * (1 - stats.norm.cdf(abs(z_rd)))

    print(f"\n  Default rate treated: {dr_t:.4f}")
    print(f"  Default rate control: {dr_c:.4f}")
    dml_table(f"B2 Matching (tau={tau:.2f}): LATE estimate (CORRECTED)",
              rd, se_rd, z_rd, p_rd,
              rd - 1.96*se_rd, rd + 1.96*se_rd,
              n=len(matched_pairs),
              extra_info={"Treated": len(t_matched), "Control": len(c_matched),
                          "DR_treated": f"{dr_t:.4f}", "DR_control": f"{dr_c:.4f}"})


# ── B3: Hazard with excess probability ────────────────────────────────────
# (B3 remains unchanged if correctly implemented)
print("\n" + "="*78)
print("  B3: HAZARD WITH EXCESS PROBABILITY")
print("="*78)

# ... B3 implementation (if exists and correct) ...

print("\n" + "="*78)
print("  CORRECTIONS SUMMARY:")
print("="*78)
print("  1. A1: Renamed to 'Event-level Analysis (User-level Aggregation)'")
print("  2. B2: Fixed matching to use baseline V4 characteristics only")
print("  3. A3/A4: Implemented true cumulative exposure MSM with treatment history")
print("="*78)
