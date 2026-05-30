// gSLICr C bridge — zero-copy interface for Python ctypes.
// Takes raw numpy pointers, runs GPU SLIC, returns labels.
// No file I/O, no OpenCV dependency.

#include "gSLICr_Lib/gSLICr.h"
#include <cstring>
#include <stdexcept>

extern "C" {

int gslicr_segment(
    const unsigned char* rgb_data,  // (H, W, 3) uint8, RGB order, row-major
    int width, int height,
    int num_segments,
    int num_iterations,
    float compactness,
    int* labels_out                 // (H, W) int32, row-major, output
) {
    using namespace gSLICr;

    if (rgb_data == nullptr || labels_out == nullptr) return -1;
    if (width <= 0 || height <= 0 || num_segments <= 0) return -2;

    try {
        objects::settings s;
        s.img_size = Vector2i(width, height);
        s.no_segs = num_segments;
        s.spixel_size = 16;
        s.no_iters = num_iterations;
        s.coh_weight = compactness;
        s.color_space = XYZ;
        s.seg_method = GIVEN_NUM;
        s.do_enforce_connectivity = true;

        engines::core_engine engine(s);

        // CPU-only input image — no GPU allocation needed.
        // SetFrom(CPU_TO_CUDA) copies from our CPU buffer to engine's internal GPU buffer.
        UChar4Image* in_img = new UChar4Image(s.img_size, true, false);
        Vector4u* cpu = in_img->GetData(MEMORYDEVICE_CPU);

        for (int i = 0; i < width * height; i++) {
            // Vector4u layout: .x=B, .y=G, .z=R, .w=A
            cpu[i].x = rgb_data[i * 3 + 2];  // Blue
            cpu[i].y = rgb_data[i * 3 + 1];  // Green
            cpu[i].z = rgb_data[i * 3];      // Red
            cpu[i].w = 255;
        }

        engine.Process_Frame(in_img);

        const IntImage* label_img = engine.Get_Seg_Res();
        const int* src = label_img->GetData(MEMORYDEVICE_CPU);
        std::memcpy(labels_out, src, width * height * sizeof(int));

        delete in_img;
        return 0;
    } catch (const std::exception& e) {
        return -3;
    } catch (...) {
        return -4;
    }
}

}  // extern "C"
