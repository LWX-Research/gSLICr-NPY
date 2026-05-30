// Batch benchmark for in-process gSLICr throughput.

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gSLICr_Lib/gSLICr.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"

using namespace cv;

static void load_image_to_gslicr(const Mat& inimg, gSLICr::UChar4Image* outimg) {
  gSLICr::Vector4u* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);
  for (int y = 0; y < outimg->noDims.y; y++) {
    for (int x = 0; x < outimg->noDims.x; x++) {
      const int idx = x + y * outimg->noDims.x;
      const Vec3b& pixel = inimg.at<Vec3b>(y, x);
      outimg_ptr[idx].b = pixel[0];
      outimg_ptr[idx].g = pixel[1];
      outimg_ptr[idx].r = pixel[2];
      outimg_ptr[idx].a = 255;
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: ./demo_benchmark '/path/to/images/*.png' [num_segs]" << std::endl;
    return 1;
  }

  const std::string pattern = argv[1];
  const int num_segs = argc > 2 ? std::stoi(argv[2]) : 512;

  std::vector<String> paths;
  glob(pattern, paths, false);
  if (paths.empty()) {
    std::cerr << "no images matched pattern: " << pattern << std::endl;
    return 1;
  }

  std::vector<Mat> images;
  images.reserve(paths.size());
  for (const auto& path : paths) {
    Mat image = imread(path, IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error("failed to read image: " + std::string(path));
    }
    images.push_back(image);
  }

  const int width = images.front().cols;
  const int height = images.front().rows;
  for (const auto& image : images) {
    if (image.cols != width || image.rows != height) {
      throw std::runtime_error("all images must have the same resolution");
    }
  }

  gSLICr::objects::settings settings;
  settings.img_size.x = width;
  settings.img_size.y = height;
  settings.no_segs = num_segs;
  settings.spixel_size = 16;
  settings.coh_weight = 0.6f;
  settings.no_iters = 5;
  settings.color_space = gSLICr::XYZ;
  settings.seg_method = gSLICr::GIVEN_NUM;
  settings.do_enforce_connectivity = true;

  auto* engine = new gSLICr::engines::core_engine(settings);
  auto* in_img = new gSLICr::UChar4Image(settings.img_size, true, true);

  const auto t0 = std::chrono::high_resolution_clock::now();
  for (const auto& image : images) {
    load_image_to_gslicr(image, in_img);
    engine->Process_Frame(in_img);
  }
  cudaDeviceSynchronize();
  const auto t1 = std::chrono::high_resolution_clock::now();

  const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double per_image_ms = total_ms / static_cast<double>(images.size());
  const double fps = 1000.0 / per_image_ms;

  std::cout << "images=" << images.size()
            << " resolution=" << width << "x" << height
            << " num_segs=" << num_segs
            << " total_ms=" << total_ms
            << " per_image_ms=" << per_image_ms
            << " fps=" << fps
            << std::endl;

  delete in_img;
  delete engine;
  return 0;
}
