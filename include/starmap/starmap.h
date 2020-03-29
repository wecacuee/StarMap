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

std::tuple<Points, std::vector<std::string>, std::vector<float>, std::vector<float>>
 find_semantic_keypoints_prob_depth(torch::jit::script::Module model,
                                    const cv::Mat& img,
                                    const int input_res,
                                    const bool visualize);

cv::Mat nms(const cv::Mat& det, const int size = 3);

torch::jit::script::Module
  model_load(const std::string& model_path, const int gpu_id);

std::tuple<cv::Mat, cv::Mat, cv::Mat>
  model_forward(torch::jit::script::Module model,
                const cv::Mat& imgfloat);

std::vector<cv::Point2i> parse_heatmap(cv::Mat & det, const float thresh = 0.05);

 void visualize_keypoints(cv::Mat& vis, const Points& pts, const
                          std::vector<std::string>& label_list,
                          bool draw_labels = false);


 /**
  * Represents canonical semantic points of a car
  */
class CarStructure {
public:
  CarStructure();
  const std::string&
    find_semantic_part(const cv::Matx<float, 3, 1>& cam_view_feat) const;
  const cv::Scalar get_label_color(const std::string& label) const;
  const size_t get_label_index(const std::string& label) const;

protected:
  const cv::Matx<float, 12, 3> canonical_points_;
  const std::vector<std::string> labels_;
  const cv::Matx<uint8_t, 12, 3> colors_;
};

static const CarStructure GLOBAL_CAR_STRUCTURE;


}

#endif // STARMAP_STARMAP_H
