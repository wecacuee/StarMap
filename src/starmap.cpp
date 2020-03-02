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
 *
 * @param img            The image to crop
 * @param desired_side   Desired size of output cropped image
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

/**
 * @brief nms
 * @param det
 * @param size
 * @return
 */
cv::Mat nms(const cv::Mat& det, const int size) {
  gsl_Expects(det.type() == CV_32F);
  cv::Mat pooled = cv::Mat::zeros(det.size(), det.type());
  int start = size / 2;
  for (int i = start; i < det.size[0] - start; ++i) {
    for (int j = start; j < det.size[1] - start; ++j) {
      cv::Mat window = det(cv::Range(i-start, i-start+size),
                           cv::Range(j-start, j-start+size));
      pooled.at<float>(i, j) = *std::max_element(window.begin<float>(),
                                                 window.end<float>());
    }
  }
  // Suppress the non-max parts
  cv::Mat nonmax = pooled != det;
  pooled.setTo(0, nonmax);
  return pooled;
}

/**
 * @brief Parse heatmap for points above a threshold
 *
 * @param det     The heatmap to parse
 * @param thresh  Threshold over which points are kept
 * @return        Vector of points above threshold
 */
std::vector<cv::Point2i>
    parse_heatmap(cv::Mat & det, const float thresh) {
  gsl_Expects(det.dims == 2);
  det.setTo(0, det < thresh);
  cv::Mat pooled = nms(det);
  std::vector<cv::Point2i> pts;
  cv::findNonZero(pooled > 0, pts);
  return pts;
}

// Convert a char/float mat to torch Tensor
at::Tensor matToTensor(const cv::Mat &image)
{
  bool isChar = (image.type() & 0xF) < 2;
  std::vector<int64_t> dims = {image.rows, image.cols, image.channels()};
  return torch::from_blob(image.data, dims,
                          isChar ? torch::kChar : torch::kFloat).to(torch::kFloat);
}

cv::Mat tensorToMat(const at::Tensor &tensor)
{
  auto sizes = tensor.sizes();
  return cv::Mat(sizes[0], sizes[1], CV_32FC(sizes[2]), tensor.data_ptr());
}


std::vector<cv::Point2i> run_starmap_on_img(const std::string& starmap_filepath,
                           const std::string& img_filepath,
                           const int input_res,
                           const int gpu_id)
{
    gsl_Expects(input_res > 0);

    // img = cv2.imread(opt.demo)
    const auto img = cv::imread(img_filepath, cv::IMREAD_COLOR);
    assert(img.type() == CV_8UC3);

    // img2 = Crop(img, center, scale, input_res) / 256.;
    cv::Mat img2 = crop(img, input_res);
    cv::Mat imgfloat;
    img2.convertTo(imgfloat, CV_32FC3, 1/255.0);
    assert(imgfloat.type() == CV_32FC3);
    auto input = matToTensor(imgfloat);
    input = at::transpose(input, 0, 2); // Make channel the first dimension CWH from WHC
    input = at::unsqueeze(input, 0); // Make it NCWH

    // model = torch.load(opt.loadModel)
    auto model = torch::jit::load(starmap_filepath);
    if (gpu_id >= 0) {
        auto device = torch::Device(torch::DeviceType::CUDA, gpu_id);
        model.to(device);
        input.to(device);
    }
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    torch::jit::IValue out = model.forward(inputs);
    auto outele = out.toTuple()->elements();
    auto heatmap = at::squeeze(outele[0].toTensor(), 0); // NCWH -> CWH
    //auto heatmap2 = at::squeeze(outele[1].toTensor(), 0); // NCWH -> CWH
    heatmap = at::transpose(heatmap, 0, 2); // CWH -> WHC
    cv::Mat cvout = tensorToMat(heatmap);
    cv::Mat cvgray;
    cv::cvtColor(cvout, cvgray, cv::COLOR_BGR2GRAY);
    auto pts = parse_heatmap(cvgray);
    return pts;
}

}
