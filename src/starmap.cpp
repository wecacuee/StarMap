#include <type_traits> // enable_if
#include <tuple> // tuple, tie
#include <cmath> // floor
#include <algorithm> // max

#include "starmap/starmap.h" // starmap
#include "opencv2/opencv.hpp" // cv::*
#include "gsl/gsl-lite.hpp" // gsl::*
#include "torch/script.h"

namespace starmap {

using namespace std;


/**
 * @brief crop img to a square image with each side as desired side
 * @param img
 * @param desired_side
 * @return Converted image
 */
cv::Mat crop(const cv::Mat& img,
             const int desired_side)
{
    gsl_Expects(desired_side > 0);
    // ht, wd = img.shape[0], img.shape[1]
    auto height = img.size[0];
    auto width = img.size[1];

    auto biggest_side = max(height, width);
    auto desired_height = desired_side / biggest_side * height;
    auto desired_width = desired_side / biggest_side * width;

    auto src_start_row = max(height / 2 - desired_height / 2, 0);
    auto src_end_row   = min(height / 2 + desired_height / 2, height);
    auto dst_start_row = desired_side / 2 - desired_height / 2;
    auto dst_end_row   = desired_side / 2 + desired_height / 2;

    auto src_start_col  = max(width / 2 - desired_width / 2, 0);
    auto src_end_col   = max(width / 2 + desired_width / 2, width);
    auto dst_start_col = desired_side / 2 - desired_width / 2;
    auto dst_end_col   = desired_side / 2 + desired_width / 2;


    const int size[3] = {desired_side, desired_side, 3};
    auto output_img = cv::Mat::zeros(3, size, CV_32F);
    cv::Mat output_roi = output_img(
                cv::Rect(dst_start_col, dst_start_row, desired_side, desired_side));
    img(cv::Range(src_start_row, src_end_row),
            cv::Range(src_start_col, src_end_col)).copyTo(
                output_roi);
    return output_roi;
}

at::Tensor run_starmap_on_img(const std::string starmap_filepath,
                        const std::string img_filepath,
                        const int input_res,
                        const int gpu_id)
{
    gsl_Expects(input_res > 0);

    // img = cv2.imread(opt.demo)
    const auto img = cv::imread(img_filepath);

    // img2 = Crop(img, center, scale, input_res) / 256.;
    cv::Mat img2 = crop(img, input_res) / 256.0;
    auto input = torch::from_blob(img2.ptr<float>(),
                                  {1, img2.size[0], img2.size[1], img2.size[2]});

    // model = torch.load(opt.loadModel)
    auto model = torch::jit::load(starmap_filepath);
    if (gpu_id >= 0) {
        auto device = torch::Device(torch::DeviceType::CUDA, gpu_id);
        model.to(device);
        input.to(device);
    }
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    auto output = model.forward(inputs).toTensor();
    return output;
}

}
