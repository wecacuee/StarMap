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
  gsl_Expects(tensor.ndimension() == 3 || tensor.ndimension() == 2);
  auto sizes = tensor.sizes();
  if (tensor.ndimension() == 3) {
      return cv::Mat(sizes[0], sizes[1], CV_32FC(sizes[2]), tensor.data_ptr());
  } else if (tensor.ndimension() == 2) {
      return cv::Mat(sizes[0], sizes[1], CV_32F, tensor.data_ptr());
  }
}

std::tuple<cv::Mat, cv::Mat, cv::Mat>
  model_forward(torch::jit::script::Module& model,
                         const cv::Mat& imgfloat)
{
  gsl_Expects(imgfloat.type() == CV_32FC3);
  auto input = matToTensor(imgfloat);
  // Make channel the first dimension CWH from WHC
  input = at::transpose(input, 1, 2); // WHC -> WCH
  input = at::transpose(input, 0, 1); // WCH -> CWH
  input = at::unsqueeze(input, 0); // Make it NCWH
  input.to((*model.parameters().begin()).device());
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input);
  torch::jit::IValue out = model.forward(inputs);
  auto outele = out.toTuple()->elements();
  auto heatmap = outele[0].toTensor();
  cv::Mat cvout = tensorToMat(heatmap[0][0]);
  cv::Mat xyz = tensorToMat(at::slice(heatmap[0], 1, 4) * imgfloat.size[0]);
  cv::Mat depth = tensorToMat(heatmap[0][4] * imgfloat.size[0]);
  gsl_Ensures(cvout.type() == CV_32FC1);
  // gsl_Ensures(xyz.type() == CV_32FC3);
  gsl_Ensures(depth.type() == CV_32FC1);
  return std::make_tuple(cvout, xyz, depth);
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
    // model = torch.load(opt.loadModel)
    auto model = torch::jit::load(starmap_filepath);
    if (gpu_id >= 0) {
      auto device = torch::Device(torch::DeviceType::CUDA, gpu_id);
      model.to(device);
    }
    cv::Mat hm00, xyz, depth;
    std::tie(hm00, xyz, depth)  = model_forward(model, imgfloat);
    auto pts = parse_heatmap(hm00, 0.1);
    std::cerr << "pts.size : " << pts.size() << "\n";

    auto vis = img2;
    cv::Mat star = hm00 * 255;
    cv::Mat tmp, starvis;
    cv::cvtColor(star, tmp, cv::COLOR_GRAY2BGR);
    cv::resize(tmp, starvis, cv::Size(imgfloat.size[0], imgfloat.size[1]));
    starvis = starvis * 0.4 + imgfloat * 255 * 0.6;
    cv::Mat starvisimg;
    starvis.convertTo(starvisimg, CV_8UC1);
    for (const auto& pt: pts) {
      cv::Point2i pt4(pt.x * 4, pt.y * 4);
      cv::circle(vis, pt4, 4, cv::Scalar(255, 255, 255), -1);
      cv::circle(vis, pt4, 2, cv::Scalar(xyz.at<cv::Scalar>(pt.y, pt.x)[0],
                                         xyz.at<cv::Scalar>(pt.y, pt.x)[1],
                                         xyz.at<cv::Scalar>(pt.y, pt.x)[2]),
                 -1);
    }
    cv::imshow("vis", vis);
    cv::imshow("starvis", starvisimg);
    cv::waitKey(-1);
    return pts;
}

}
