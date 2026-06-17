# TROUBLESHOOTING

本文件记录项目中遇到的所有错误及其解决方案。

---

## 1. `NDArray.permute` — shape 和 strides 不同步导致结果错误

**日期**: 2026-06-17

**现象**: `test_permute` 中非恒等排列的测试用例失败（params0 `axes=(0,1,2)` 通过，params1 `axes=(1,0,2)` 和 params2 `axes=(2,1,0)` 失败）。`np.testing.assert_allclose` 报错，90%+ 的元素不匹配。

**根因**:

`permute` 的目标是不复制内存，只通过调整 `shape` 和 `strides` 来改变对同一块内存的视角。原始代码只按 `new_axes` 重新排列了 shape，然后调用 `reshape(new_shape)`。`reshape` 会重新计算 C-contiguous strides（假设最后一维 stride=1，依次往前乘），完全丢弃了原始 strides 中的排列信息。

元素 `A[i0, i1, i2]` 的内存偏移量由 strides 决定：`offset = i0*stride[0] + i1*stride[1] + i2*stride[2]`。仅改 shape 不改 strides，意味着坐标到内存的映射保持不变，但 shape 暗示了错误的维度范围。只有恒等排列 `(0,1,2)` 碰巧通过测试，因为 shape 不变，reshape 反推出的 strides 恰好和原 strides 相同。

**错误代码** (`ndarray.py:304-310`，修复前):

```python
def permute(self, new_axes):
    old_shape = self.shape
    new_shape = []
    for i in range(len(new_axes)):
        new_shape.append(old_shape[new_axes[i]])
    return self.reshape(new_shape)   # reshape 会重算 C-contiguous strides
```

**正确代码**:

```python
def permute(self, new_axes):
    new_shape = tuple(self._shape[ax] for ax in new_axes)
    new_strides = tuple(self._strides[ax] for ax in new_axes)
    return NDArray.make(
        new_shape, strides=new_strides, device=self.device,
        handle=self._handle, offset=self._offset
    )
```

**关键点**: shape 和 strides 必须按相同的 `new_axes` 同步排列。不能委托给 `reshape`，因为 `reshape` 是合并/拆分连续维度（假设 C-contiguous），而 `permute` 是重新排列维度顺序（需要保留原始 strides）。

**涉及文件**: `python/needle/backend_ndarray/ndarray.py`
