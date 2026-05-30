# gSLICr

GPU 加速的 SLIC 超像素分割库，C++ / CUDA 实现。

## 编译

```bash
mkdir build && cd build
cmake .. -DGSLICR_CUDA_ARCH=86   # 按 GPU 计算能力设置
make -j
```

产物：

| 文件 | 需要 OpenCV | 说明 |
|------|------------|------|
| `gslicr_npy` | 否 | NPY 输入输出 |
| `libgslicr_bridge.so` | 否 | Python ctypes 动态库 |
| `demo` | 是 | 摄像头实时演示 |
| `demo_image` | 是 | 单张图片处理 |
| `demo_benchmark` | 是 | 性能测试 |

## 使用

**命令行（NPY）：**

```bash
./gslicr_npy input.npy labels.npy 512
# input.npy: (H, W, 3) uint8 RGB
# labels.npy: (H, W) int32

# 可选环境变量
GSLICR_N_ITERS=5 GSLICR_COMPACTNESS=0.6 ./gslicr_npy input.npy labels.npy 200
```

**Python：**

```python
from hints.utils.gslicr_bridge import gslicr_segment
import numpy as np

rgb = np.load("slice.npy")  # (H, W, 3) uint8
labels = gslicr_segment(rgb, n_segments=512, n_iters=5, compactness=0.6)
```
