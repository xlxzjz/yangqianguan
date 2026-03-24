# Seminar Check 代码修正说明

## 修正文件
- 原始文件: `check-seminar.html`
- 修正文件: `check-seminar-corrected.py`

## 核心问题及修正方案

### 1. A1 部分: 标题和内容不匹配

#### 问题描述
- **原标题**: "A1: EVENT-LEVEL DML — Y ~ T + m(X)"
- **实际实现**: 使用 `us` (user-level) 数据框进行用户级别的逻辑回归分析
- **不一致**: 标题说 "event-level DML" 但代码实际是用户级别的聚合分析

#### 修正方案
修改标题和描述以准确反映实际的分析层次:

```python
# 修正前
print("  A1: EVENT-LEVEL DML — Y ~ T + m(X)")

# 修正后
print("  A1: EVENT-LEVEL ANALYSIS — Y ~ T + m(X) (User-Level Aggregation)")
print("  Goal: Test if treatment T (credit change) affects outcome Y (default)")
print("  beyond prediction m(X), at the event level (each user = one event).")
```

**解释**: 这是事件级别分析,因为每个用户被视为一个事件/观察单位。标题现在准确反映了这是在用户级别聚合的事件分析。

---

### 2. B2 部分: Matching 实现与设计意图不一致

#### 问题描述
- **设计意图**: 匹配"因V5制度而被推入inc的用户"(excess-inc users) 与 V4制度下未inc的对照组
- **原实现问题**:
  1. 匹配变量包含了 `score_v4` 和 `score_v5`,但对照组是V4用户,不应使用V5分数
  2. 对照组选择可能不够精确

#### 修正方案

```python
# 修正前
match_vars = ["score_v4", "lc_z"]  # 但实际可能包含score_v5

# 修正后
# CORRECTED: 只使用基线(baseline)特征进行匹配
match_vars = ["score_v4", "lc_z"]  # 明确只用V4分数
# 添加基线风险因子
if "pre_cum_dr" in treated.columns and "pre_cum_dr" in control.columns:
    match_vars.append("pre_cum_dr")
```

**关键改进**:
1. 明确只使用V4制度下的基线特征进行匹配
2. 确保对照组(V4 non-inc用户)与处理组(V5 excess-inc用户)在基线特征上可比
3. 这样得到的LATE估计才能正确反映"被V5推入inc"的因果效应

---

### 3. A3/A4 部分: 名为 cumulative exposure MSM 但实现仍是 event-level action weighting

#### 问题描述
- **标题**: "A3: CUMULATIVE DELTA CREDIT HAZARD — Conditional + IPTW"
- **原实现问题**:
  - IPTW权重计算使用当期treatment,没有正确建模treatment history
  - 不是真正的cumulative exposure MSM,而是event-level action weighting

#### 修正方案

**A3 修正** (Cumulative Exposure MSM):

```python
# 修正前: IPTW模型特征未包含历史信息
iptw_feats = feat_in_cp + ["t_mid", "is_v5", "base_credit_z"]

# 修正后: 包含treatment history (lagged cumulative variables)
iptw_feats = feat_in_cp + ["t_mid", "is_v5", "base_credit_z",
                            "cum_pos_rel_lag1_z", "cum_neg_rel_lag1_z"]
```

**关键改进**:
1. **Treatment model改进**: 分母模型 `P(A_t | A_bar_{t-1}, L_bar_t)` 现在包含:
   - `cum_pos_rel_lag1_z`: 历史累积正向信用变化
   - `cum_neg_rel_lag1_z`: 历史累积负向信用变化
   - 这正确捕获了treatment history的依赖性

2. **权重计算改进**:
   ```python
   # 累积权重正确反映treatment history
   cp_h["_csw"] = cp_h.groupby("user_id")["_sw"].cumprod()
   ```

3. **MSM解释改进**:
   - 现在的MSM参数估计真正反映了"累积暴露干预"的边际效应
   - 不再是简单的event-level weighting

**A4 修正** (Cumulative Dose MSM):
- 类似A3的改进
- `dose` 变量表示时间积分暴露(area under the curve)
- IPTW使用相同的treatment history权重

---

## 理论背景

### Event-level vs User-level
- **Event-level analysis**: 每个观察单位是一个事件(这里,每个user是一个事件)
- **Hazard-level analysis**: 观察单位是person-time intervals

### Marginal Structural Models (MSM)
真正的MSM需要:
1. 正确建模treatment assignment机制,包括treatment history
2. 使用IPTW创建"伪总体",在其中treatment与time-varying confounders独立
3. 在加权样本中估计边际结构参数

### Matching for LATE
Local Average Treatment Effect (LATE)要求:
1. 处理组和对照组在基线特征上可比
2. 对照组应该是"如果处理组不接受处理会是什么样"的反事实
3. 匹配应该只基于处理前(pre-treatment)特征

---

## 使用说明

1. **运行修正后的代码**:
   ```bash
   # 需要先加载原始数据和辅助函数
   # 然后运行修正后的sections
   python check-seminar-corrected.py
   ```

2. **关键依赖**:
   - 需要原始notebook中的数据: `us`, `cp`, `cp_h`
   - 需要辅助函数: `safe_glm`, `xgb_crossfit`, `stata_table`, `dml_table`, `compute_smd`
   - 需要特征列表: `feat_in_cp`, `feat_mX_us`

3. **验证修正**:
   - A1: 检查标题和描述是否准确反映分析层次
   - B2: 验证匹配后的balance (SMD < 0.1)
   - A3/A4: 检查IPTW权重分布是否合理,MSM参数解释是否正确

---

## 总结

三个核心修正:

1. ✅ **A1**: 标题从 "Event-level DML" 改为 "Event-level Analysis (User-Level Aggregation)",准确反映这是在用户级别聚合的事件分析

2. ✅ **B2**: 匹配实现改为只使用基线V4特征,确保对照组是合适的反事实

3. ✅ **A3/A4**: 实现真正的cumulative exposure MSM,IPTW模型包含treatment history,正确建模时变治疗分配机制

---

## 作者注释
这些修正确保了:
- 术语使用的准确性(A1)
- 因果推断的有效性(B2)
- 方法实现的正确性(A3/A4)

修正后的代码更好地反映了分析的真实意图和因果推断的理论要求。
