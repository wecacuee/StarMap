#ifndef STARMAP_STARMAP_H
#define STARMAP_STARMAP_H

#include <type_traits>
#include <tuple>

#include "opencv2/opencv.hpp"
#include "torch/script.h"

namespace starmap {

cv::Mat crop(const cv::Mat& img,
             const cv::Point2i center,
             const int desired_side);

 at::Tensor run_starmap_on_img(const std::string starmap_filepath,
                               const std::string img_filepath,
                               const int input_res,
                               const int gpu_id);
}

#endif // STARMAP_STARMAP_H
