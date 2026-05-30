// demo_npy.cpp — gSLICr with native NPY I/O. No OpenCV needed.
//
// Usage: ./gslicr_npy input.npy [output.npy] [num_segments]
//   input.npy:      (H, W, 3) uint8 RGB image
//   output.npy:     (H, W) int32 label map (default: labels.npy)
//   num_segments:   target superpixel count (default: 512)
//
// Overrides via env:
//   GSLICR_N_ITERS      k-means iterations (default: 5)
//   GSLICR_COMPACTNESS  spatial weight (default: 0.6)
//   GSLICR_COLOR_SPACE  rgb|xyz|cielab (default: xyz)

#include "gSLICr_Lib/gSLICr.h"
#include "npy_io.h"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    const char* input_path  = argc > 1 ? argv[1] : nullptr;
    const char* output_path = argc > 2 ? argv[2] : "labels.npy";
    int num_segs            = argc > 3 ? std::stoi(argv[3]) : 512;

    if (!input_path) {
        std::cerr << "Usage: gslicr_npy input.npy [output.npy] [n_segments]\n";
        return 1;
    }

    // --- env overrides ---
    int n_iters = 5;
    float compactness = 0.6f;
    gSLICr::COLOR_SPACE cs = gSLICr::XYZ;
    {
        const char* v;
        if ((v = std::getenv("GSLICR_N_ITERS"))) n_iters = std::stoi(v);
        if ((v = std::getenv("GSLICR_COMPACTNESS"))) compactness = std::stof(v);
        if ((v = std::getenv("GSLICR_COLOR_SPACE"))) {
            std::string s(v);
            if (s == "rgb") cs = gSLICr::RGB;
            else if (s == "cielab") cs = gSLICr::CIELAB;
        }
    }

    // --- load NPY ---
    auto arr = npy::load(input_path);
    if (arr.shape.size() != 3 || arr.shape[2] != 3) {
        std::cerr << "Expected (H, W, 3) uint8 NPY, got shape:";
        for (auto d : arr.shape) std::cerr << " " << d;
        std::cerr << "\n";
        return 1;
    }
    int H = static_cast<int>(arr.shape[0]);
    int W = static_cast<int>(arr.shape[1]);
    const unsigned char* rgb = reinterpret_cast<const unsigned char*>(arr.raw.data());

    // --- configure gSLICr ---
    gSLICr::objects::settings s;
    s.img_size = gSLICr::Vector2i(W, H);
    s.no_segs  = num_segs;
    s.spixel_size = 16;
    s.no_iters = n_iters;
    s.coh_weight = compactness;
    s.color_space = cs;
    s.seg_method = gSLICr::GIVEN_NUM;
    s.do_enforce_connectivity = true;

    // --- segment ---
    auto* engine = new gSLICr::engines::core_engine(s);
    auto* in_img = new gSLICr::UChar4Image(s.img_size, true, false);

    {
        auto* cpu = in_img->GetData(MEMORYDEVICE_CPU);
        for (int i = 0; i < H * W; i++) {
            // Vector4u layout: .x=B, .y=G, .z=R, .w=A
            cpu[i].x = rgb[i * 3 + 2];  // B
            cpu[i].y = rgb[i * 3 + 1];  // G
            cpu[i].z = rgb[i * 3];      // R
            cpu[i].w = 255;
        }
    }

    engine->Process_Frame(in_img);

    const gSLICr::IntImage* label_img = engine->Get_Seg_Res();
    const int* labels = label_img->GetData(MEMORYDEVICE_CPU);

    int n_labels = 0;
    for (int i = 0; i < H * W; i++) {
        if (labels[i] > n_labels) n_labels = labels[i];
    }

    // --- save NPY ---
    npy::save(output_path, labels, {static_cast<size_t>(H), static_cast<size_t>(W)});

    std::cout << "input=" << input_path
              << " shape=" << W << "x" << H
              << " segments=" << (n_labels + 1)
              << " output=" << output_path
              << std::endl;

    delete in_img;
    delete engine;
    return 0;
}
