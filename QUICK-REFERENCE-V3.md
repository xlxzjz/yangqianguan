# 快速参考: 使用哪个文件？

## ✅ 使用这个文件 (最终版本)
**`check-seminar-corrected-v3.py`** - V3最终完整版本

## ❌ 不要使用
- `check-seminar-corrected.py` - V1 (错误)
- `check-seminar-corrected-v2.py` - V2 (正确但不完整)

---

## V3 vs V2 的关键区别

### 新增内容:

#### 1. B2: 现在测试 4 种特征集合
```python
# V2: 只测试 1 种特征集合
match_vars = ["score_v4", "lc_z"]

# V3: 测试 4 种特征集合
feature_sets = [
    "V4_only",      # V4 + baseline
    "V5_only",      # V5 + baseline
    "V4_and_V5",    # 两个分数
    "V4_or_V5",     # 所有特征
]
```

**输出**: 3 tau × 4 feature sets = **12组匹配结果**

#### 2. A3/A4: MSM实现验证和完善

V3中添加了:
- 详细的MSM实现说明
- IPTW权重统计输出
- MSM成功/失败标记 (✓/✗)
- 治疗历史包含验证

---

## 完整修正列表 (V3)

1. ✅ **A1**: 使用 `cp_h` 事件级数据
2. ✅ **A2**: 保持正确的hazard分析
3. ✅ **A3**: MSM with IPTW (包含treatment history)
4. ✅ **A4**: MSM with IPTW (复用A3权重)
5. ✅ **B2**: 4种特征集合匹配 (新增)

---

## MSM验证检查清单

- [x] Treatment history in denominator model
- [x] Stabilized weights (numerator/denominator)
- [x] Cumulative weights via groupby().cumprod()
- [x] Weighted GLM with freq_weights
- [x] Robust standard errors
- [x] Weight truncation at p1/p99
- [x] Marginal interpretation (no confounders in MSM)

**结论**: MSM实现正确 ✅

---

## 详细文档

- **完整技术文档**: `CORRECTIONS-README-V3.md`
- **代码实现**: `check-seminar-corrected-v3.py`
- **历史参考**: `CORRECTIONS-README-V2.md`

---

## 版本历史

| 版本 | A1 | B2 | A3/A4 | 状态 |
|------|----|----|-------|------|
| V1 | ❌ | ❌ | ✅ | 废弃 |
| V2 | ✅ | 部分 | ✅ | 不完整 |
| **V3** | ✅ | **完整** | ✅ | **最终版** |
