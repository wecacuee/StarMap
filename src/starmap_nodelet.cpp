#include <memory>
#include <functional>

#include "boost/range/counting_range.hpp"
#include "opencv2/opencv.hpp" // cv::*

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "nodelet/nodelet.h"
#include <pluginlib/class_list_macros.h>

#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/subscriber.h>
#include <darknet_ros_msgs/BoundingBoxes.h>
#include <cv_bridge/cv_bridge.h>
#include <kp_detector/KeypointsList.h>
#include "starmap/starmap.h"
#include "boost/filesystem.hpp"

using namespace std;
using cv::Mat;
using cv::Vec3f;
namespace bfs = boost::filesystem;

namespace starmap
{

  template<typename T, typename... Args>
  std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
  }


  class Starmap : public nodelet::Nodelet
  {
  public:
    Starmap() {}

  private:
  virtual void onInit() {
    NODELET_DEBUG("Initializing ");
    namespace sph = std::placeholders; // for _1, _2, ...
    std::string image_topic, bbox_topic, keypoint_topic, starmap_model_path;
    int gpu_id;
    auto nh = getNodeHandle();
    auto private_nh = getPrivateNodeHandle();
    NODELET_DEBUG("Got node handles");
    if ( ! private_nh.getParam("starmap_model_path", starmap_model_path) ) {
      NODELET_FATAL("starmap_model_path is required");
      throw std::runtime_error("starmap_model_path is required");
    }
    if ( ! bfs::exists(starmap_model_path) ) {
      NODELET_FATAL("starmap_model_path '%s' does not exists", starmap_model_path.c_str());
      throw std::runtime_error("starmap_model_path does not exists");
    }
    private_nh.param<std::string>("image_topic", image_topic, "image");
    private_nh.param<std::string>("bbox_topic", bbox_topic, "bounding_boxes");
    private_nh.param<std::string>("keypoint_topic", keypoint_topic, "keypoints");
    private_nh.param<int>("gpu_id", gpu_id, -1);
    //timer_ = nh.createTimer(ros::Duration(1.0),
    //                        std::bind(& Starmap::timerCb, this, sph::_1));

    // model = torch.load(opt.loadModel)
    NODELET_DEBUG("Loading  model from %s", starmap_model_path.c_str());
    model_ = torch::jit::load(starmap_model_path);
    NODELET_DEBUG("gpu_id : %d", gpu_id);
    torch::DeviceType device_type = gpu_id >= 0 ? torch::DeviceType::CUDA : torch::DeviceType::CPU;
    int device_id = gpu_id >= 0 ? gpu_id : 0;
    torch::Device device = torch::Device(device_type, gpu_id);
    model_.to(device);

    NODELET_DEBUG("Subscribing to %s", image_topic.c_str());
    image_sub_ = make_unique<message_filters::Subscriber<sensor_msgs::Image>>(nh, image_topic, 1);
    bbox_sub_ = make_unique<
      message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes>>(nh, bbox_topic, 1);
    sub_ = make_unique<
      message_filters::TimeSynchronizer<
        sensor_msgs::Image, darknet_ros_msgs::BoundingBoxes>>(*image_sub_, *bbox_sub_, 10);
    sub_->registerCallback(std::bind(&Starmap::messageCb, this, sph::_1, sph::_2));
    pub_ = private_nh.advertise<kp_detector::KeypointsList>(keypoint_topic, 10);

  }

  // must use a ConstPtr callback to use zero-copy transport
  void messageCb(const sensor_msgs::ImageConstPtr& message,
                 const darknet_ros_msgs::BoundingBoxesConstPtr& bboxes) {
    NODELET_DEBUG("Callback called ... ");
    auto private_nh = getPrivateNodeHandle();
    int input_res;
    private_nh.param<int>("input_res", input_res, 256);
    bool visualize;
    private_nh.param<bool>("visualize", visualize, false);
    cv_bridge::CvImageConstPtr img = cv_bridge::toCvShare(message);
    kp_detector::KeypointsList keypoints;
    keypoints.header.stamp = message->header.stamp;
    for (auto& bbox: bboxes->bounding_boxes) {
      auto bboxroi = img->image(cv::Rect2f(bbox.xmin, bbox.ymin,
                                           bbox.xmax-bbox.xmin, bbox.ymax-bbox.ymin));
      Points pts;
      vector<Vec3f> xyz_list;
      vector<float> depth_list;
      Mat bboxfloat;
      bboxroi.convertTo(bboxfloat, CV_32FC3, 1/255.0);
      NODELET_DEBUG("Calling  model ... ");
      tie(pts, xyz_list, depth_list) =
        find_semantic_keypoints_prob_depth(model_, bboxfloat, input_res,
                                           /*visualize=*/false);

      kp_detector::Keypoints kpts;
      kpts.Class = bbox.Class;
      kpts.probability = bbox.probability;
      kpts.xmin = bbox.xmin;
      kpts.xmax = bbox.xmax;
      kpts.ymin = bbox.ymin;
      kpts.ymax = bbox.ymax;
      std::cerr << "message: " << img << "bboxes: " << bboxes << "\n";
      for (size_t i: boost::counting_range<size_t>(0, pts.size())) {
        auto& pt = pts[i];
        kpts.hm_keypoints.push_back( pt.x );
        kpts.hm_keypoints.push_back( pt.y );
      }
      keypoints.keypoints_list.push_back(kpts);
    }
    pub_.publish(keypoints);
  }

  std::unique_ptr<message_filters::Subscriber<sensor_msgs::Image>> image_sub_;
  std::unique_ptr<message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes>> bbox_sub_;
  std::unique_ptr<message_filters::TimeSynchronizer<sensor_msgs::Image, darknet_ros_msgs::BoundingBoxes>> sub_;
  ros::Publisher pub_;
    // ros::Timer timer_;
  torch::jit::script::Module model_;
};

} // namespace Starmap

PLUGINLIB_EXPORT_CLASS( starmap::Starmap, nodelet::Nodelet );

