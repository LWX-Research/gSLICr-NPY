# gSLICr 部署说明

## 当前环境

- GPU: `RTX 3090`（compute capability 8.6）
- CUDA 编译器: `12.1`
- OpenCV: `4.5.4`

## 构建

```bash
cd /root/autodl-tmp/third_party/gSLICr
mkdir -p build
cd build
cmake .. -DGSLICR_CUDA_ARCH=86
make -j
```

产物：

- `libgSLICr_lib.a` — 静态库
- `libgslicr_bridge.so` — Python ctypes 共享库（零文件 I/O）
- `demo` / `demo_image` / `demo_benchmark` — 可执行文件

## 验证

离线图片 demo:

```bash
./demo_image ../sample.bmp gslicr_sample
# → gslicr_sample_overlay.png + gslicr_sample_labels.png
```

Python bridge:

```python
from hints.utils.gslicr_bridge import gslicr_segment
labels = gslicr_segment(rgb_uint8, n_segments=512)
# → (H, W) int32 label map, ~8ms on RTX 3090 (640×480)
```
