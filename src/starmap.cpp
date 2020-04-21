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
  return cv::Scalar(col(0,0), col(0,1), col(0,2));
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
  findNonZero(pooled > thresh, pts);
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
  Mat xyz = tensorToMat(heatmap1to3);
  Mat depth = tensorToMat(heatmap[0][4]);
  gsl_Ensures(cvout.type() == CV_32FC1);
  gsl_Ensures(xyz.type() == CV_32FC3);
  gsl_Ensures(depth.type() == CV_32FC1);
  return make_tuple(cvout.clone(), xyz.clone(), depth.clone());
}

vector<SemanticKeypoint>
mean_grouped_by_label(vector<SemanticKeypoint> const& semkp_list)
{
  unordered_map<string, vector<SemanticKeypoint>> label2idx;
  for (size_t idx = 0; idx < semkp_list.size(); idx ++) {
    SemanticKeypoint const& semkp = semkp_list[idx];
    string const& label = semkp.label;
    if (label2idx.count(label)) {
      label2idx.at(label).push_back(semkp);
    } else {
      vector<SemanticKeypoint> values({semkp});
      label2idx[label] = values;
    }
  }

  vector<SemanticKeypoint> value_uniq;
  for (auto const& keyval: label2idx) {
    vector<SemanticKeypoint> values = keyval.second;
    SemanticKeypoint value_mean = std::accumulate(values.begin(), values.end(),
                                                  SemanticKeypoint::Zero());
    float ksize = values.size();
    if (ksize) {
      value_uniq.emplace_back(value_mean / ksize);
    }
  }
  return value_uniq;
}


//tuple<Points, vector<string>, vector<float>, vector<float>>
vector<SemanticKeypoint>
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

  if (visualize) {
    Mat star;
    resize(hm00 * 255, star, {img_cropped.size[0], img_cropped.size[1]});
    Mat starvis;
    cvtColor(star, starvis, COLOR_GRAY2BGR);
    starvis = starvis * 0.5 + img_cropped * 255 * 0.5;
    Mat starvisimg;
    starvis.convertTo(starvisimg, CV_8UC1);
    imshow("starvis", starvisimg);
    imwrite("/tmp/starvis.png", starvisimg);
    waitKey(-1);
  }


  vector<SemanticKeypoint> semkp_list;
  for (auto const& pt: pts) {
    Point3f xyz_at = xyz.at<Point3f>(pt.y, pt.x);
    Vec3f xyz_vec{ xyz_at.x, xyz_at.y, xyz_at.z };
    semkp_list.emplace_back(pt,
                            xyz_vec,
                            depth.at<float>(pt.y, pt.x),
                            hm00.at<float>(pt.y, pt.x),
                            GLOBAL_CAR_STRUCTURE.find_semantic_part(xyz_vec));
  }
  auto pts_old_kp = convert_to_precrop(pts, {img.size[1], img.size[0]}, input_res,
                                       /*addnl_scale_factor=*/ADDNL_SCALE_FACTOR);
  for (size_t i = 0; i < pts_old_kp.size(); i++) {
    semkp_list[i].pos2d = pts_old_kp[i];
  }

  if (unique_labels) {
    semkp_list = mean_grouped_by_label(semkp_list);
  }

  return semkp_list;
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

    vector<SemanticKeypoint> semkp_list =
      find_semantic_keypoints_prob_depth(model, imgfloat, input_res, visualize, /*unique_labels=*/true);

    if (visualize) {
      auto vis = img;
      visualize_keypoints(vis, semkp_list, /*draw_labels=*/true);
      imshow("vis", vis);
      imwrite("/tmp/vis.png", vis);
      waitKey(-1);
    }

    vector<Point2i> pts;
    std::transform(semkp_list.begin(), semkp_list.end(),
                   std::back_inserter(pts),
                   [](SemanticKeypoint const & semkp) -> Point2i {
                     return semkp.pos2d;
                   });

    return pts;
}

void visualize_keypoints(Mat& vis, const vector<SemanticKeypoint>& semkp_list,
                         bool draw_labels) {
  for (auto const& semkp : semkp_list) {
    auto& pt4 = semkp.pos2d;
    auto col = GLOBAL_CAR_STRUCTURE.get_label_color(semkp.label);
    if (draw_labels) {
        circle(vis, pt4, 3, Scalar(255, 255, 255), -1);
        circle(vis, pt4, 2, col, -1);
        putText(vis, semkp.label, pt4,
                cv::FONT_HERSHEY_SIMPLEX,
                /*fontSize=*/std::max(0.3, 0.3 * vis.rows / 480),
                /*color=*/Scalar(255, 255, 255), /*lineThickness=*/1);
    } else {
        circle(vis, pt4, 3, Scalar(255, 255, 255), -1);
        circle(vis, pt4, 2, col, -1);
    }
  }
}

std::ostream& operator<< (std::ostream& o, const SemanticKeypoint& semkp) {
  o << "SemanticKeypoint(" << "pos2d=" << semkp.pos2d << ", "
    << "xyz=" << semkp.xyz << ","
    << "depth=" << semkp.depth << ", "
    << "hm=" << semkp.hm << ", "
    << "label=" << semkp.label
    << ")";
  return o;
}

}
