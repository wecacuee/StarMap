#include <type_traits> // enable_if
#include <tuple> // tuple, tie
#include <cmath> // floor
#include <algorithm> // max

#include "starmap/starmap.h" // starmap
#include "opencv2/opencv.hpp" // cv::*
#include "opencv2/core/mat.hpp"
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
    constexpr int D = 2;
    gsl_Expects(desired_side > 0);
    gsl_Expects(img.dims >= 2 && img.dims <= 3);
    int max_side = max(img.size[0], img.size[1]);
    // scale the image first
    cv::Mat resized_img;
    cv::resize(img, resized_img,
               cv::Size((img.size[1] * desired_side / max_side),
                        (img.size[0] * desired_side / max_side)));

    // Cropping begins here
    // The image rectangle clockwise
    cv::Rect2f rect_resized(0, 0, resized_img.size[1], resized_img.size[0]);
    auto resized_max_side = max(resized_img.size[0], resized_img.size[1]);


    // Project the rectangle from source image to target image
    // TODO account for rotation
    cv::Point2f target_center( desired_side / 2, desired_side / 2);
    cv::Point2f resized_img_center( resized_img.size[1] / 2, resized_img.size[0] / 2);
    auto rect_target = (rect_resized - resized_img_center) + target_center;

    // img.size[2] might not be accessible
    const int size[3] = {desired_side, desired_side, img.size[2]};
    cv::Mat output_img = cv::Mat::zeros(img.dims, size, img.type());
    cv::Mat output_roi(output_img, rect_target);
    cv::Mat source_roi = resized_img(cv::Rect(0, 0, rect_target.width, rect_target.height));
    source_roi.copyTo(output_roi);
    return output_img;
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
