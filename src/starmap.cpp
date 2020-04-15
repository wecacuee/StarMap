#include <type_traits> // enable_if
#include <tuple> // tuple, tie
#include <cmath> // floor
#include <algorithm> // max
#include <unordered_map>
#include <boost/range/counting_range.hpp>
#include <boost/format.hpp>

#include "starmap/starmap.h" // starmap
#include "opencv2/opencv.hpp" // cv::*
#include "opencv2/core/mat.hpp"
#include "gsl/gsl-lite.hpp" // gsl::*
#include "torch/script.h"


using std::vector;
using std::string;
using std::unordered_map;
using std::cout;
using std::tuple;
using std::tie;
using std::max;
using std::make_tuple;
using std::swap;
using cv::Mat;
using cv::Matx;
using cv::Scalar;
using cv::circle;
using cv::imread;
using cv::imwrite;
using cv::imshow;
using cv::waitKey;
using cv::IMREAD_COLOR;
using cv::findNonZero;
using cv::Point2i;
using cv::Point2f;
using cv::Point3f;
using cv::Vec3f;
using cv::Rect;
using cv::Rect2f;
using cv::Range;
using cv::Size;
using cv::cvtColor;
using cv::COLOR_GRAY2BGR;
using boost::format;

namespace starmap {

CarStructure::CarStructure() :
    canonical_points_{
                      -0.09472257, -0.07266671,  0.10419698,
                      0.09396329, -0.07186594,  0.10468729,
                      0.100639  , 0.26993483, 0.11144333,
                      -0.100402 ,  0.2699945,  0.111474 ,
                      -0.12014713, -0.40062513, -0.02047777,
                      0.1201513 , -0.4005558 , -0.02116918,
                      0.12190333, 0.40059162, 0.02385612,
                      -0.12194733,  0.40059462,  0.02387712,
                      -0.16116614, -0.2717491 , -0.07981283,
                      -0.16382502,  0.25057048, -0.07948726,
                      0.1615844 , -0.27168764, -0.07989835,
                      0.16347528,  0.2507412 , -0.07981754 },
    labels_{
            "upper_left_windshield",
            "upper_right_windshield",
            "upper_right_rearwindow",
            "upper_left_rearwindow",
            "left_front_light",
            "right_front_light",
            "right_back_trunk",
            "left_back_trunk",
            "left_front_wheel",
            "left_back_wheel",
            "right_front_wheel",
            "right_back_wheel"},
    colors_{0, 0, 0,
            0, 0, 128,
            0, 0, 255,
            0, 128, 0,
            0, 128, 128,
            0, 128, 255,
            0, 255, 0,
            0, 255, 128,
            0, 255, 255,
            255, 0, 0,
            255, 0, 128,
            255, 0, 255}
  {
  }


const std::string&
  CarStructure::find_semantic_part(const cv::Matx<float, 3, 1>& cam_view_feat) const
{
  Matx<float, 1, 3> cam_view_feat_mat(cam_view_feat.reshape<1, 3>());
  Matx<float, 12, 1> distances;
  for (int i = 0; i < canonical_points_.rows; ++i) {
    distances(i, 0) = cv::norm((cam_view_feat_mat - canonical_points_.row(i)),
                               cv::NORM_L2SQR);
  }
  float* it = std::min_element(distances.val, distances.val + 12);
  size_t min_index = std::distance(distances.val, it);
  return labels_[min_index];
}

const cv::Scalar
  CarStructure::get_label_color(const std::string& label) const
{
  auto col = colors_.row(get_label_index(label));
  return cv::Scalar(col(0,0), col(1,0), col(2,0));
}


const size_t
  CarStructure::get_label_index(const std::string& label) const
{
  auto it = std::find(labels_.begin(), labels_.end(), label);
  return std::distance(labels_.begin(), it);
}


double scale_for_crop(const Point2i& img_size,
                      const int desired_side)
{
  int max_side = max(img_size.x, img_size.y);
  double scale_factor = static_cast<double>(desired_side) / static_cast<double>(max_side);
  gsl_Ensures(scale_factor > 0);
  return scale_factor;
}


Points convert_to_precrop(const Points& keypoints,
                          const Point2i& pre_crop_size,
                          const int desired_side,
                          const double addnl_scale_factor )
{
  Points pre_crop_kp_vec;
  Point2i curr_size(desired_side, desired_side);
  double scale_factor = scale_for_crop(pre_crop_size, desired_side);
  for (auto& kp: keypoints) {
    Point2i pre_crop_kp = (kp * addnl_scale_factor - curr_size / 2) / scale_factor
      + pre_crop_size / 2;
    pre_crop_kp_vec.push_back(pre_crop_kp);
  }
  return pre_crop_kp_vec;
}


/**
 * @brief crop img to a square image with each side as desired side
 *
 * @param img            The image to crop
 * @param desired_side   Desired size of output cropped image
 * @return Converted image
 */
Mat crop(const Mat& img,
             const int desired_side)
{
    constexpr int D = 2;
    gsl_Expects(desired_side > 0);
    gsl_Expects(img.dims >= 2 && img.dims <= 3);
    double scale_factor = scale_for_crop({img.size[1], img.size[0]}, desired_side);
    // scale the image first
    Mat resized_img;
    resize(img, resized_img,
               Size((img.size[1] * scale_factor),
                    (img.size[0] * scale_factor)));

    // Cropping begins here
    // The image rectangle clockwise
    Rect2f rect_resized(0, 0, resized_img.size[1], resized_img.size[0]);
    auto resized_max_side = max(resized_img.size[0], resized_img.size[1]);


    // Project the rectangle from source image to target image
    // TODO account for rotation
    Point2f target_center( desired_side / 2, desired_side / 2);
    Point2f resized_img_center( resized_img.size[1] / 2, resized_img.size[0] / 2);
    auto translate = target_center - resized_img_center ;
    auto rect_target = (rect_resized + translate);

    // img.size[2] might not be accessible
    const int size[3] = {desired_side, desired_side, img.size[2]};
    Mat output_img = Mat::zeros(img.dims, size, img.type());
    Mat output_roi(output_img, rect_target);
    Mat source_roi = resized_img(Rect(0, 0, rect_target.width, rect_target.height));
    source_roi.copyTo(output_roi);
    return output_img;
}

/**
 * @brief nms
 * @param det
 * @param size
 * @return
 */
Mat nms(const Mat& det, const int size) {
  gsl_Expects(det.type() == CV_32F);
  Mat pooled = Mat::zeros(det.size(), det.type());
  int start = size / 2;
  for (int i = start; i < det.size[0] - start; ++i) {
    for (int j = start; j < det.size[1] - start; ++j) {
      Mat window = det(Range(i-start, i-start+size),
                       Range(j-start, j-start+size));
      double minval, maxval;
      minMaxLoc(window, &minval, &maxval);
      // auto mele = max_element(window.begin<float>(), window.end<float>());
      pooled.at<float>(i, j) = maxval;
    }
  }
  // Suppress the non-max parts
  Mat nonmax = pooled != det;
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
vector<Point2i>
    parse_heatmap(Mat & det, const float thresh) {
  gsl_Expects(det.dims == 2);
  gsl_Expects(det.data != nullptr);
  Mat mask = det < thresh;
  det.setTo(0, mask);
  Mat pooled = nms(det);
  vector<Point2i> pts;
  findNonZero(pooled > 0, pts);
  return pts;
}

// Convert a char/float mat to torch Tensor
at::Tensor matToTensor(const Mat &image)
{
  bool isChar = (image.type() & 0xF) < 2;
  vector<int64_t> dims = {image.rows, image.cols, image.channels()};
  return torch::from_blob(image.data, dims,
                          isChar ? torch::kChar : torch::kFloat).to(torch::kFloat);
}

Mat tensorToMat(const at::Tensor &tensor)
{
  gsl_Expects(tensor.ndimension() == 3 || tensor.ndimension() == 2);
  gsl_Expects(tensor.dtype() == torch::kFloat32);
  auto tensor_c = tensor.contiguous();
  auto sizes = tensor.sizes();
  if (tensor.ndimension() == 3) {
    return Mat(sizes[0], sizes[1], CV_32FC(sizes[2]), tensor_c.data_ptr());
  } else if (tensor.ndimension() == 2) {
    return Mat(sizes[0], sizes[1], CV_32F, tensor_c.data_ptr());
  }
}

tuple<Mat, Mat, Mat>
  model_forward(torch::jit::script::Module model,
                const Mat& imgfloat)
{
  gsl_Expects(imgfloat.type() == CV_32FC3);
  auto input = matToTensor(imgfloat);
  // Make channel the first dimension CWH from WHC
  input = input.permute({2, 0, 1}); // WHC -> CWH
  input.unsqueeze_(0); // Make it NCWH
  torch::Device device = (*model.parameters().begin()).device();
  auto input_device = input.to(device);
  vector<torch::jit::IValue> inputs;
  inputs.push_back(input_device);
  torch::jit::IValue out = model.forward(inputs);
  auto outele = out.toTuple()->elements();
  auto heatmap_device = outele[0].toTensor();
  torch::Device cpu = torch::Device(torch::DeviceType::CPU, 0);
  auto heatmap = heatmap_device.to(cpu);
  Mat cvout = tensorToMat(heatmap[0][0]);
  auto heatmap1to3 = at::slice(heatmap[0], /*dim=*/0, /*start=*/1, /*end=*/4);
  heatmap1to3 = heatmap1to3.permute({ 1, 2, 0}); // CWH -> WHC
  Mat xyz = tensorToMat(heatmap1to3 * imgfloat.size[0]);
  Mat depth = tensorToMat(heatmap[0][4] * imgfloat.size[0]);
  gsl_Ensures(cvout.type() == CV_32FC1);
  gsl_Ensures(xyz.type() == CV_32FC3);
  gsl_Ensures(depth.type() == CV_32FC1);
  return make_tuple(cvout.clone(), xyz.clone(), depth.clone());
}

template<typename T, typename T2>
vector<T>
mean_grouped_by(vector<T2> const& key_vect, vector<T> const& value_vec,
                T const& zero)
{
  size_t idx;
  unordered_map<T2, vector<size_t>> label2idx;
  for (auto const& label : key_vect) {
    if (label2idx.count(label)) {
      label2idx.at(label).push_back(++idx);
    } else {
      vector<size_t> indices({idx});
      label2idx[label] = indices;
    }
  }
  vector<T> value_uniq;

  for (auto const& keyval: label2idx) {
    T value_mean = std::accumulate(value_vec.begin(), value_vec.end(), zero);
    if (keyval.second.size()) {
      float ksize = keyval.second.size();
      value_uniq.emplace_back(value_mean / ksize);
    }
  }
  return value_uniq;
}


tuple<Points, vector<string>, vector<float>, vector<float>>
   find_semantic_keypoints_prob_depth(torch::jit::script::Module model,
                                      const Mat& img,
                                      const int input_res,
                                      const bool visualize,
                                      const bool unique_labels)
{
  const int ADDNL_SCALE_FACTOR = 4;
  // img2 = Crop(img, center, scale, input_res) / 256.;
  gsl_Expects(img.type() == CV_32FC3);
  Mat img_cropped = crop(img, input_res);

  Mat hm00, xyz, depth;
  tie(hm00, xyz, depth)  = model_forward(model, img_cropped);
  auto pts = parse_heatmap(hm00, 0.1);
  vector<Vec3f> xyz_list;
  vector<float> depth_list;
  vector<float> hm_list;
  for (auto& pt: pts) {
    Point3f xyz_at = xyz.at<Point3f>(pt.y, pt.x);
    xyz_list.emplace_back(xyz_at.x, xyz_at.y, xyz_at.z);
    depth_list.push_back(depth.at<float>(pt.y, pt.x));
    hm_list.push_back(hm00.at<float>(pt.y, pt.x));
  }

  vector<string> label_list;
  transform(xyz_list.begin(), xyz_list.end(),
            std::back_inserter(label_list),
            std::bind(&CarStructure::find_semantic_part,
                      GLOBAL_CAR_STRUCTURE, std::placeholders::_1));

  if (unique_labels) {
    Point2i zero(0, 0);
    Points pts_uniq = mean_grouped_by(label_list, pts, zero);
    Vec3f z(0, 0, 0);
    vector<Vec3f> xyz_uniq = mean_grouped_by(label_list, xyz_list, z);
    vector<float> depth_uniq = mean_grouped_by(label_list, depth_list, 0.0f);
    vector<float> hm_uniq = mean_grouped_by(label_list, hm_list, 0.0f);
  }

  if (visualize) {
    Mat star;
    resize(hm00 * 255, star, {img_cropped.size[0], img_cropped.size[1]});
    Mat starvis;
    cvtColor(star, starvis, COLOR_GRAY2BGR);
    starvis = starvis * 0.5 + img_cropped * 255 * 0.5;
    Mat starvisimg;
    starvis.convertTo(starvisimg, CV_8UC1);
    imshow("starvis", starvisimg);
    waitKey(-1);
  }

  auto pts_old_kp = convert_to_precrop(pts, {img.size[1], img.size[0]}, input_res,
                                       /*addnl_scale_factor=*/ADDNL_SCALE_FACTOR);

  return make_tuple(pts_old_kp, label_list, depth_list, hm_list);
}


torch::jit::script::Module
  model_load(const std::string& starmap_filepath,
             const int gpu_id)
{
  // model = torch.load(opt.loadModel)
  auto model = torch::jit::load(starmap_filepath);
  torch::DeviceType device_type = gpu_id >= 0 ? torch::DeviceType::CUDA : torch::DeviceType::CPU;
  int device_id = gpu_id >= 0 ? gpu_id : 0;
  torch::Device device = torch::Device(device_type, gpu_id);
  model.to(device);
  return model;
}


vector<Point2i> run_starmap_on_img(const string& starmap_filepath,
                                   const string& img_filepath,
                                   const int input_res,
                                   const int gpu_id,
                                   const bool visualize)
{
    gsl_Expects(input_res > 0);

    // img = cv2.imread(opt.demo)
    const auto img = imread(img_filepath, IMREAD_COLOR);
    assert(img.type() == CV_8UC3);
    Mat imgfloat;
    img.convertTo(imgfloat, CV_32FC3, 1/255.0);

    auto model = model_load(starmap_filepath, gpu_id);

    Points pts;
    vector<string> label_list;
    vector<float> depth_list;
    vector<float> hm_list;
    tie(pts, label_list, depth_list, hm_list) =
      find_semantic_keypoints_prob_depth(model, imgfloat, input_res, visualize);

    if (visualize) {
      auto vis = img;
      visualize_keypoints(vis, pts, label_list);
      imshow("vis", vis);
      waitKey(-1);
    }

    return pts;
}

void visualize_keypoints(Mat& vis, const Points& pts, const vector<string>& label_list,
                         bool draw_labels) {
  for (int i: boost::counting_range<size_t>(0, pts.size())) {
    auto& pt4 = pts[i];
    auto& col = GLOBAL_CAR_STRUCTURE.get_label_color(label_list[i]);
    circle(vis, pt4, 2, Scalar(255, 255, 255), -1);
    circle(vis, pt4, 1, col, -1);
    if (draw_labels)
      putText(vis, label_list[i], pt4,
              cv::FONT_HERSHEY_SIMPLEX,
              /*fontSize=*/std::max(0.4, 0.01 * vis.rows),
              /*color=*/col, /*lineThickness=*/1);
  }
}

}
