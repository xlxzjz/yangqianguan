# 快速参考: 使用哪个文件？

## ✅ 使用这个文件
**`check-seminar-corrected-v2.py`** - 正确版本

## ❌ 不要使用
- `check-seminar-corrected.py` - V1版本，**修正是错误的**

---

## V2 vs V1 的关键区别

### A1 实现:

**V1 (错误):**
```python
# 只改了标题，数据还是用 us (用户级)
m = safe_glm(fml, us, cluster_col=C["user_id"])
```

**V2 (正确):**
```python
# 实际改了实现，使用 cp_h (事件级)
cp_h = cp.dropna(subset=["mX"]).copy()
m = safe_glm(fml, cp_h, cluster_col="user_id")
```

---

## 完整修正列表 (V2)

1. ✅ **A1**: 使用 `cp_h` 数据，outcome是 `event` (hazard)
2. ✅ **B2**: 匹配只用baseline V4特征
3. ✅ **A3/A4**: 真正的cumulative exposure MSM

---

## 详细文档

- **技术细节**: 见 `CORRECTIONS-README-V2.md`
- **代码实现**: 见 `check-seminar-corrected-v2.py`
