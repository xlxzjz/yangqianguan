# Seminar Check 代码修正说明 (V2 - 最终版本)

## 重要更新
**第一版修正是错误的！** A1本身就应该实现event-level分析，之前的实现错误地使用了user聚合数据。

## 修正文件
- 原始文件: `check-seminar.html`
- ~~修正文件V1: `check-seminar-corrected.py`~~ ❌ (错误 - 只改了标题)
- **修正文件V2: `check-seminar-corrected-v2.py`** ✅ (正确 - 实际修改了实现)

## 核心问题及修正方案

### 1. A1 部分: 实现错误，使用了user聚合而非event-level数据

#### 问题描述
- **原标题**: "A1: EVENT-LEVEL DML — Y ~ T + m(X)" ✓ (标题是对的)
- **原实现**: 使用 `us` (user-level) 数据框，outcome是 `defaulted` (用户级聚合结果)
- **问题**: **实现本身就是错的**，应该使用event-level数据 (`cp` 或 `cp_h`)，outcome应该是 `event` (离散时间hazard)

#### 对比 A2 (正确的event-level实现):
```python
# A2 (正确的event-level):
cp_h = cp.dropna(subset=["mX"]).copy()
a2_specs = [
    ("A2a", "event ~ t_mid + t_mid_sq + mX + base_credit_z + tv_delta_rel_z",
     "time + m(X) + credit + delta"),
    # ...
]
for spec_id, fml, label in a2_specs:
    m = safe_glm(fml, cp_h, cluster_col="user_id")  # 使用 cp_h!
```

#### 修正方案 (V2)

**错误的V1修正** (只改了标题和描述):
```python
# V1 - 错误! 只改了标题，数据还是用us
print("  A1: EVENT-LEVEL ANALYSIS — Y ~ T + m(X) (User-Level Aggregation)")
# ...
m = safe_glm(fml, us, cluster_col=C["user_id"])  # 还是用us! ❌
```

**正确的V2修正** (改了实现):
```python
# V2 - 正确! 使用event-level数据
print("  A1: EVENT-LEVEL DML — Y ~ T + m(X)")
print("  at true event-level (person-time intervals, not user aggregates)")

# 使用 cp_h (event-level hazard data)
cp_h = cp.dropna(subset=["mX"]).copy()

a1_specs = [
    ("A1a", "event ~ t_mid + t_mid_sq + mX + base_credit_z + tv_delta_rel_z",
     "time + m(X) + credit + delta"),
    ("A1b", "event ~ t_mid + t_mid_sq + mX + base_credit_z + tv_is_inc + tv_is_dec",
     "time + m(X) + credit + inc/dec indicators"),
    # ...
]

for spec_id, fml, label in a1_specs:
    m = safe_glm(fml, cp_h, cluster_col="user_id")  # 使用 cp_h! ✅
```

#### 关键变化:
1. **数据源**: `us` (用户级) → `cp_h` (事件级person-time)
2. **Outcome**: `defaulted` (用户是否违约) → `event` (在该时间区间是否发生事件)
3. **Treatment**: `delta_pos_z`, `delta_neg_z` (用户总变化) → `tv_delta_rel_z`, `tv_is_inc`, `tv_is_dec` (时变治疗)
4. **DML实现**: 也改为使用event-level数据，并在用户层面做cluster bootstrap

#### 理论解释:

**User-level (聚合) 分析**:
- 每个用户一个观察值
- Outcome: 用户是否最终违约
- 无法捕捉时变(time-varying)效应

**Event-level (真正的) 分析**:
- 每个person-time interval一个观察值
- Outcome: 该时间点的hazard (是否发生事件)
- 可以建模时变治疗和时变混淆因子
- A1和A2都应该是这种分析

---

### 2. B2 部分: Matching 实现与设计意图不一致

#### 问题描述
- 匹配应该只基于baseline V4特征
- 对照组应该是V4非inc用户，与处理组在基线可比

#### 修正方案 (保持不变)

```python
# 只使用baseline特征
match_vars = ["score_v4", "lc_z"]
if "pre_cum_dr" in treated.columns and "pre_cum_dr" in control.columns:
    match_vars.append("pre_cum_dr")

# 对照组: V4非inc用户
control = us[(us["is_v5"] == 0) & (us["is_inc"] == 0)].copy()
```

---

### 3. A3/A4 部分: 实现真正的cumulative exposure MSM

#### 修正方案 (保持不变)

IPTW treatment model包含treatment history:
```python
iptw_feats = feat_in_cp + ["t_mid", "is_v5", "base_credit_z",
                            "cum_pos_rel_lag1_z", "cum_neg_rel_lag1_z"]
```

---

## 数据结构说明

### `us` (user-level data)
- 每行: 一个用户
- 变量: `user_id`, `defaulted`, `is_v5`, `delta_pos_z`, `delta_neg_z`, etc.
- 用途: B1, B2 (用户级分析)

### `cp` / `cp_h` (person-time / event-level data)
- 每行: 一个person-time interval
- 变量: `user_id`, `t_mid`, `event`, `tv_delta_rel_z`, `tv_is_inc`, `tv_is_dec`, etc.
- 用途: **A1 (修正后)**, A2, A3, A4 (事件级/hazard分析)

## 完整修正对比表

| 部分 | 原实现 | V1修正 (错误) | V2修正 (正确) |
|------|--------|--------------|--------------|
| **A1** | 使用`us`数据 ❌ | 只改标题 ❌ | 改用`cp_h`数据 ✅ |
| **A2** | 使用`cp_h`数据 ✅ | 不变 ✅ | 不变 ✅ |
| **B2** | 匹配有问题 ❌ | 修正匹配变量 ✅ | 保持修正 ✅ |
| **A3/A4** | event-level weighting ❌ | 实现真MSM ✅ | 保持修正 ✅ |

## 使用说明

1. **运行修正后的代码**:
   ```bash
   # 使用V2版本 (正确版本)
   python check-seminar-corrected-v2.py
   ```

2. **关键依赖**:
   - 需要原始数据: `us` (user-level), `cp` (person-time), `cp_h` (cleaned person-time)
   - 需要辅助函数: `safe_glm`, `xgb_crossfit`, `stata_table`, `dml_table`, `compute_smd`
   - 需要特征列表: `feat_in_cp`, `feat_mX_us`

3. **验证修正**:
   - **A1**: 确认使用 `cp_h` 数据，outcome是 `event`
   - **A2**: 保持使用 `cp_h` 数据 (作为参考)
   - **B2**: 验证匹配balance (SMD < 0.1)
   - **A3/A4**: 检查IPTW包含treatment history

---

## 为什么第一版修正是错的？

### 误解:
我误以为"A1标题说event-level但实现是user-level"意味着应该改标题。

### 实际情况:
- 标题本来就是对的："EVENT-LEVEL DML"
- **实现是错的**: 应该用event-level数据但用了user-level数据
- 需要改的是**代码实现**，不是标题

### 证据:
- A2叫"Event-level Hazard DML"，使用`cp_h`数据 ✓
- A1叫"Event-level DML"，但用`us`数据 ✗
- 用户说"A1部分本身就应该是实现event level，是之前实现错误了，不应该做user聚合"

---

## 总结

修正版本V2的核心变化:

1. ✅ **A1**: 从user-level聚合改为真正的event-level分析
   - 数据: `us` → `cp_h`
   - Outcome: `defaulted` → `event`
   - Treatment: user-level deltas → time-varying treatments

2. ✅ **B2**: 匹配只用baseline V4特征，确保因果推断有效性

3. ✅ **A3/A4**: 实现真正的cumulative exposure MSM，IPTW包含treatment history

修正后的代码准确反映了event-level分析的设计意图，A1和A2现在都正确地使用person-time数据进行discrete-time hazard建模。
