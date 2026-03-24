# Seminar Check 代码修正说明 - 最终版本

## 修正文件版本历史
- **原始文件**: `check-seminar.html`
- **V1**: `check-seminar-corrected.py` (初步修正,理解有误)
- **V2**: `check-seminar-corrected-v2.py` (A1理解错误)
- **V3**: `check-seminar-corrected-v3.py` (A1理解错误,但B2正确添加4种特征集)
- **最终版本**: `check-seminar-corrected-v3-final.py` ✅

## 关键理解修正

### "Event-level" 的正确理解

在本项目中,**"event-level"** 有特定含义:
- **A1中的"event-level"**: 将每个用户(user)视为一个事件/观察单位
  - 使用 `us` 数据框(user-level aggregated data)
  - 每个用户=一个事件
  - 这是**正确的**原始实现

- **A2中的"hazard-level"**: 真正的离散时间生存分析
  - 使用 `cp_h` 数据框(person-time data)
  - 每个person-time interval=一个观察单位

**关键教训**: "Event-level" 不一定意味着person-time data。在A1的上下文中,它简单地表示将数据视为event-level进行DML分析,其中每个用户就是一个事件。

---

## 核心修正内容

### 1. A1 部分: 保持原始实现(使用user-level数据)

#### 最终理解
- **原标题**: "A1: EVENT-LEVEL DML — Y ~ T + m(X)" ✅ **正确**
- **原实现**: 使用 `us` (user-level) 数据框 ✅ **正确**
- **含义**: Event-level = 每个用户作为一个事件/观察单位

#### 最终代码
```python
# ── A1: Event-level DML (User-level data, each user = one event) ──────────
# NOTE: "Event-level" here means treating each user as one event/observation
# NOT person-time hazard analysis (that's A2)
print("  A1: EVENT-LEVEL DML — Y ~ T + m(X)")
print("  Event-level: each user is one event/observation.")

a1_specs = [
    ("A1a", "defaulted ~ delta_abs_z + mX + lc_z", "net delta"),
    ("A1b", "defaulted ~ delta_pos_z + delta_neg_z + mX + lc_z", "decomposed"),
]

for spec_id, fml, label in a1_specs:
    m = safe_glm(fml, us, cluster_col=C["user_id"])  # 正确使用 us
    stata_table([m])
    dml_table(m, idx=0, spec_id=spec_id, res=res)
```

**关键点**:
- 使用 `us` 数据框(user-level aggregated)
- 结果变量: `defaulted` (用户是否违约)
- 治疗变量: `delta_abs_z`, `delta_pos_z`, `delta_neg_z` (用户级别的信用额度变化)
- 这就是正确的"event-level DML"实现

---

### 2. A2 部分: 保持原实现(person-time hazard分析)

```python
# ── A2: HAZARD-LEVEL — Event(t) ~ T(t) + covariates ───────────────────────
# NOTE: This is true person-time hazard analysis using cp_h
print("  A2: HAZARD-LEVEL — Event(t) ~ T(t) + covariates")
print("  Hazard-level: each person-time interval is one observation.")

# 使用 cp_h (person-time data)
a2_specs = [
    ("A2a", "event ~ tv_delta_rel_z + ... + lc_z", "net delta"),
    ("A2b", "event ~ tv_is_inc + tv_is_dec + ... + lc_z", "inc/dec"),
]

for spec_id, fml, label in a2_specs:
    m = safe_glm(fml, cp_h, cluster_col=C["user_id"])
    # ...
```

---

### 3. B2 部分: 4种特征集合的Matching实现

#### 修正方案
原实现只测试一种特征集,现改为测试4种特征集组合:

```python
feature_set_configs = [
    ("V4_only", ["score_v4", "lc_z"], "V4 score + baseline"),
    ("V5_only", ["score_v5", "lc_z"], "V5 score + baseline"),
    ("V4_and_V5", ["score_v4", "score_v5", "lc_z"], "Both V4 and V5 scores"),
    ("V4_or_V5", ["score_v4", "score_v5", "lc_z", "pre_cum_dr", ...], "All available features"),
]

# 对每个caliper值(tau)测试所有4种特征集
for tau in [0.05, 0.10, 0.20]:
    for feat_set_name, match_vars, desc in feature_set_configs:
        # 执行matching
        # 计算SMD验证balance
        # 估计LATE
```

#### 输出结果
- **12种组合**: 3个tau值 × 4种特征集 = 12个结果
- 每个结果包括:
  - Matching后样本大小
  - SMD (Standardized Mean Difference)
  - LATE估计及标准误

**关键改进**:
1. 系统地比较不同特征集的matching效果
2. V4_only: 只用基线V4分数
3. V5_only: 只用V5分数
4. V4_and_V5: 同时使用两个分数
5. V4_or_V5: 使用所有可用特征(最丰富的特征集)

---

### 4. A3/A4 部分: 验证MSM实现正确

#### A3: Cumulative Exposure MSM

**验证要点**:
1. ✅ IPTW分母模型包含treatment history:
   ```python
   iptw_feats = feat_in_cp + ["t_mid", "is_v5", "base_credit_z",
                               "cum_pos_rel_lag1_z",  # 历史累积正向变化
                               "cum_neg_rel_lag1_z"]   # 历史累积负向变化
   ```

2. ✅ 累积稳定权重计算正确:
   ```python
   cp_h["_sw"] = cp_h.apply(compute_sw, axis=1)
   cp_h["_csw"] = cp_h.groupby("user_id")["_sw"].cumprod()
   cp_h["_csw_trunc"] = cp_h["_csw"].clip(0.1, 10)
   ```

3. ✅ MSM模型包含累积暴露:
   ```python
   msm_fml = "event ~ t_mid + t_mid_sq + cum_pos_rel_lag1_z + cum_neg_rel_lag1_z"
   m_msm = smf.glm(msm_fml, data=cp_h,
                    family=sm.families.Binomial(),
                    freq_weights=cp_h["_csw_trunc"].values).fit(cov_type="HC1")
   ```

**理论依据**:
- 分母: P(A_t | A̅_{t-1}, L̅_t) - 包含treatment history
- 分子: P(A_t | A̅_{t-1}) - 只依赖treatment history
- SW_t = 分子/分母
- CSW_t = ∏_{k=1}^{t} SW_k

#### A4: Cumulative Dose MSM

类似A3的验证,但关注点是累积剂量(dose = 时间积分暴露):
```python
# dose 变量表示area under the curve
cp_h["dose"] = cp_h.groupby("user_id")["cum_pos_rel_z"].transform(
    lambda x: (x + x.shift(1)).fillna(0) * 0.5
)
```

---

## 版本修正历史

### V1 (check-seminar-corrected.py)
- ❌ 错误: 只修改了描述,没有真正理解问题
- ❌ A1保持原样但理解不正确

### V2 (check-seminar-corrected-v2.py)
- ❌ 错误: 将A1改为使用`cp_h`数据,误解了"event-level"含义
- ✅ 正确: 开始实现B2 matching改进
- ✅ 正确: A3/A4 MSM验证

### V3 (check-seminar-corrected-v3.py)
- ❌ 错误: A1仍使用`cp_h`数据
- ✅ 正确: B2完整实现4种特征集matching
- ✅ 正确: A3/A4 MSM详细验证

### V3-Final (check-seminar-corrected-v3-final.py) ✅
- ✅ 正确: A1使用`us`数据,每个用户=一个事件
- ✅ 正确: A2使用`cp_h`数据,person-time hazard分析
- ✅ 正确: B2实现4种特征集matching
- ✅ 正确: A3/A4 MSM实现和验证

---

## 使用说明

### 运行最终版本

```python
# 在jupyter notebook中,需要先运行原始notebook的数据准备部分
# 然后运行:
exec(open('check-seminar-corrected-v3-final.py').read())
```

### 关键依赖
- 数据框: `us`, `cp`, `cp_h`
- 辅助函数: `safe_glm`, `xgb_crossfit`, `stata_table`, `dml_table`, `compute_smd`
- 特征列表: `feat_in_cp`, `feat_mX_us`
- 配置: `C` dictionary

### 预期输出

#### A1输出
```
══════════════════════════════════════════════════════════════
  A1: EVENT-LEVEL DML — Y ~ T + m(X)
  Event-level: each user is one event/observation.
══════════════════════════════════════════════════════════════

SPEC: A1a — net delta
[Stata-style table with coefficients]

SPEC: A1b — decomposed
[Stata-style table with coefficients]
```

#### B2输出
```
B2 Results: tau=0.05, V4_only
  Treated: N, Control: N
  SMD: [list of SMD values]
  LATE = X.XXX (SE=X.XXX)

B2 Results: tau=0.05, V5_only
  ...

[Total: 12 results]
```

#### A3/A4输出
```
MSM Verification for A3:
  1. IPTW features include treatment history: ✓
  2. Cumulative stabilized weights calculated: ✓
  3. Weight distribution reasonable: ✓

[Regression results]
```

---

## 总结

### 三个核心修正

1. ✅ **A1**:
   - **理解正确**: "Event-level" = 每个用户作为一个事件
   - **实现正确**: 使用 `us` 数据框(原始实现就是对的)

2. ✅ **B2**:
   - **增强功能**: 从1种特征集扩展到4种特征集
   - **系统比较**: V4 only, V5 only, V4&V5, V4|V5
   - **输出丰富**: 12个matching结果组合

3. ✅ **A3/A4**:
   - **验证MSM**: 确认treatment history包含在IPTW分母
   - **验证权重**: 确认累积稳定权重计算正确
   - **验证模型**: 确认MSM包含累积暴露变量

### 关键教训

- **术语理解**: "Event-level" 在不同上下文有不同含义
  - A1: User-level aggregation (每个用户=一个事件)
  - A2: Person-time intervals (每个时间段=一个观察)

- **原始代码**: A1的原始实现其实是正确的,不需要修改

- **增强功能**: B2从简单matching扩展到多特征集系统比较

---

## 作者注释

这个修正过程揭示了理解项目特定术语的重要性。"Event-level DML" 在这里并不意味着需要使用person-time数据,而是简单地表示将用户级别的数据视为事件级别进行DML分析。

最终版本(`check-seminar-corrected-v3-final.py`)正确实现了:
- A1: User-level event DML
- A2: Person-time hazard analysis
- B2: Multi-feature-set matching (4种组合)
- A3/A4: Proper cumulative exposure MSM with treatment history

修正后的代码既保持了原始设计的正确性,又增强了B2部分的分析深度。
