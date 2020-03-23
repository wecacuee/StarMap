#ifndef STARMAP_STARMAP_H
#define STARMAP_STARMAP_H

#include <type_traits>
#include <tuple>

#include "opencv2/opencv.hpp"
#include "torch/script.h"

namespace starmap {

cv::Mat crop(const cv::Mat& img,
             const int desired_side);

typedef std::vector<cv::Point2i> Points;
Points run_starmap_on_img(const std::string& starmap_filepath,
                          const std::string& img_filepath,
                          const int input_res,
                          const int gpu_id,
                          const bool visualize = true);

std::tuple<Points, std::vector<cv::Vec3f>, std::vector<float>, std::vector<float>>
 find_semantic_keypoints_prob_depth(torch::jit::script::Module& model,
                                    const cv::Mat& img,
                                    const int input_res,
                                    const bool visualize);

cv::Mat nms(const cv::Mat& det, const int size = 3);

std::tuple<cv::Mat, cv::Mat, cv::Mat>
  model_forward(torch::jit::script::Module& model,
                const cv::Mat& imgfloat);

std::vector<cv::Point2i> parse_heatmap(cv::Mat & det, const float thresh = 0.05);

 void visualize_keypoints(cv::Mat& vis, const Points& pts, const std::vector<cv::Vec3f>& colors);

}

#endif // STARMAP_STARMAP_H
