// Minimal offline demo for image-based validation on headless servers.

#include <cstring>
#include <iostream>
#include <stdexcept>

#include "gSLICr_Lib/gSLICr.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

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

static void load_gslicr_to_mat(const gSLICr::UChar4Image* inimg, Mat& outimg) {
  const gSLICr::Vector4u* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);
  for (int y = 0; y < inimg->noDims.y; y++) {
    for (int x = 0; x < inimg->noDims.x; x++) {
      const int idx = x + y * inimg->noDims.x;
      outimg.at<Vec3b>(y, x)[0] = inimg_ptr[idx].b;
      outimg.at<Vec3b>(y, x)[1] = inimg_ptr[idx].g;
      outimg.at<Vec3b>(y, x)[2] = inimg_ptr[idx].r;
    }
  }
}

int main(int argc, char** argv) {
  const char* input_path = argc > 1 ? argv[1] : "sample.bmp";
  const char* output_prefix = argc > 2 ? argv[2] : "gslicr_output";
  const int num_segs = argc > 3 ? std::stoi(argv[3]) : 512;

  Mat image = imread(input_path, IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "failed to read image: " << input_path << std::endl;
    return 1;
  }

  gSLICr::objects::settings settings;
  settings.img_size.x = image.cols;
  settings.img_size.y = image.rows;
  settings.no_segs = num_segs;
  settings.spixel_size = 16;
  settings.coh_weight = 0.6f;
  settings.no_iters = 5;
  settings.color_space = gSLICr::XYZ;
  settings.seg_method = gSLICr::GIVEN_NUM;
  settings.do_enforce_connectivity = true;

  auto* engine = new gSLICr::engines::core_engine(settings);
  auto* in_img = new gSLICr::UChar4Image(settings.img_size, true, true);
  auto* out_img = new gSLICr::UChar4Image(settings.img_size, true, true);

  load_image_to_gslicr(image, in_img);
  engine->Process_Frame(in_img);
  engine->Draw_Segmentation_Result(out_img);

  Mat overlay(image.rows, image.cols, CV_8UC3);
  load_gslicr_to_mat(out_img, overlay);
  imwrite(std::string(output_prefix) + "_overlay.png", overlay);

  const gSLICr::IntImage* labels = engine->Get_Seg_Res();
  const int* label_data = labels->GetData(MEMORYDEVICE_CPU);
  if (label_data == nullptr) {
    throw std::runtime_error("failed to access CPU label buffer");
  }

  Mat label_mat(labels->noDims.y, labels->noDims.x, CV_32SC1);
  std::memcpy(label_mat.data, label_data, label_mat.total() * sizeof(int));

  double min_label = 0.0;
  double max_label = 0.0;
  minMaxLoc(label_mat, &min_label, &max_label);
  std::cout << "input=" << input_path
            << " shape=" << image.cols << "x" << image.rows
            << " segments=" << (static_cast<int>(max_label) + 1)
            << " label_min=" << min_label
            << " label_max=" << max_label
            << std::endl;

  Mat normalized_labels;
  label_mat.convertTo(normalized_labels, CV_16U);
  imwrite(std::string(output_prefix) + "_labels.png", normalized_labels);

  delete out_img;
  delete in_img;
  delete engine;
  return 0;
}
