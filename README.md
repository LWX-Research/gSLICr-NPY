# gSLICr: SLIC superpixels at over 250Hz

![sample](sample.bmp)

This is the software bundle "gSLICr", a library for real-time superpixel segmentation written in C++ and CUDA. The current version is maintained by:

  [Carl Yuheng Ren](http://www.carlyuheng.com/) : <carl@robots.ox.ac.uk>  
  [Victor Adrian Prisacariu](http://www.robots.ox.ac.uk/~victor/) : <victor@robots.ox.ac.uk>  
  Ian D Reid : <ian.reid@adelaide.edu.au>  

For more information about gSLICr please visit the project website <http://www.robots.ox.ac.uk/~victor/gslicr>. 

Other related projects can be found in the Oxford Active Vision Library <http://www.oxvisionlib.org>.

# 1. Updates from gSLIC
- Works for any size / number of super pixels
- With GTX Titan X, 3.4ms@640x480, 12ms@1280x960, 20ms@1920x1080 image
- Multi-platform supported
  - Win8 Visual Studio
  - Ubuntu 14.04
  - Mac OSX 10.10

**This fork adds:**
- **Native NPY I/O** — `gslicr_npy` reads/writes `.npy` files directly, no OpenCV needed
- **Python ctypes bridge** — `libgslicr_bridge.so` for zero-copy in-process segmentation (~8ms)
- **OpenCV is optional** — only required for legacy webcam/PNG demos
- **CUDA arch auto-detect** — set `-DGSLICR_CUDA_ARCH=86` for RTX 3090, `=89` for RTX 4080 SUPER, etc.

# 2. Building the System

### 2.1 Requirements

  - cmake (>= 3.10)
  - CUDA (>= 11.0)
  - OpenCV (optional — only for legacy webcam/PNG demos)

### 2.2 Build Process

```bash
mkdir build && cd build
cmake .. -DGSLICR_CUDA_ARCH=86   # set to your GPU's compute capability
make -j
```

This produces:

| Binary | Needs OpenCV | Description |
|--------|-------------|-------------|
| `gslicr_npy` | **no** | NPY in → NPY out (recommended) |
| `libgslicr_bridge.so` | **no** | Python ctypes shared library |
| `demo` | yes | Webcam real-time demo |
| `demo_image` | yes | Legacy PNG/BMP tool |
| `demo_benchmark` | yes | Performance benchmark |

### 2.3 Usage

**CLI (NPY):**
```bash
./gslicr_npy input.npy labels.npy 512
# input.npy:  (H, W, 3) uint8 RGB
# labels.npy: (H, W) int32 label map

# Env overrides:
GSLICR_N_ITERS=5 GSLICR_COMPACTNESS=0.6 GSLICR_COLOR_SPACE=xyz ./gslicr_npy input.npy labels.npy 200
```

**Python bridge (in-process, ~8ms):**
```python
from hints.utils.gslicr_bridge import gslicr_segment
import numpy as np

rgb = np.load("slice.npy")  # (H, W, 3) uint8
labels = gslicr_segment(rgb, n_segments=512, n_iters=5, compactness=0.6)
# labels: (H, W) int32
```
