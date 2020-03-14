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

namespace starmap
{
  using namespace std;
  using cv::Mat;
  using cv::Vec3f;

  template<typename T, typename... Args>
  std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
  }


  class Starmap : public nodelet::Nodelet
  {
  public:
    Starmap() {}

  private:
  virtual void onInit(){
    namespace sph = std::placeholders; // for _1, _2, ...
    std::string image_topic, bbox_topic, keypoint_topic, starmap_model_path;
    int gpu_id;
    auto nh = getNodeHandle();
    auto private_nh = getPrivateNodeHandle();
    if ( ! nh.param<std::string>("image_topic", image_topic, "image") )
      throw std::runtime_error("Need image_topic");
    if ( ! nh.param<std::string>("bbox_topic", bbox_topic, "bounding_boxes") )
      throw std::runtime_error("Need bbox_topic");
    if ( ! nh.param<std::string>("keypoint_topic", keypoint_topic, "keypoints") )
      throw std::runtime_error("Need keypoint_topic");
    if ( ! nh.param<std::string>("starmap_model_path", starmap_model_path,
                                 "models/model_cpu-jit.pth") )
      throw std::runtime_error("Need starmap_model_path");
    if ( ! nh.param<int>("gpu_id", gpu_id, -1) )
      throw std::runtime_error("Need gpu_id");
    timer_ = nh.createTimer(ros::Duration(1.0),
                            std::bind(& Starmap::timerCb, this, sph::_1));
    image_sub_ = make_unique<message_filters::Subscriber<sensor_msgs::Image>>(nh, image_topic, 1);
    bbox_sub_ = make_unique<message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes>>(nh, bbox_topic, 1);
    sub_ = make_unique<message_filters::TimeSynchronizer<sensor_msgs::Image, darknet_ros_msgs::BoundingBoxes>>(*image_sub_, *bbox_sub_, 10);
    sub_->registerCallback(std::bind(&Starmap::messageCb, this, sph::_1, sph::_2));
    pub_ = private_nh.advertise<kp_detector::KeypointsList>(keypoint_topic, 10);

    // model = torch.load(opt.loadModel)
    model_ = torch::jit::load(starmap_model_path);
    if (gpu_id >= 0) {
      auto device = torch::Device(torch::DeviceType::CUDA, gpu_id);
      model_.to(device);
    }
  };

  void timerCb(const ros::TimerEvent& event){
    // Using timers is the preferred 'ROS way' to manual threading
    NODELET_INFO_STREAM("The time is now " << event.current_real);
  }

  // must use a ConstPtr callback to use zero-copy transport
  void messageCb(const sensor_msgs::ImageConstPtr& message,
                 const darknet_ros_msgs::BoundingBoxesConstPtr& bboxes) {
    auto nh = getNodeHandle();
    int input_res;
    if ( ! nh.param<int>("input_res", input_res, 256) )
      throw std::runtime_error("Need input_res");
    bool visualize;
    if ( ! nh.param<bool>("visualize", visualize, false) )
      throw std::runtime_error("Need visualize");
    cv_bridge::CvImageConstPtr img = cv_bridge::toCvShare(message);
    kp_detector::KeypointsList keypoints;
    keypoints.header.stamp = keypoints.header.stamp;
    for (auto& bbox: bboxes->bounding_boxes) {
      auto bboxroi = img->image(cv::Rect2f(bbox.xmin, bbox.ymin,
                                           bbox.xmax-bbox.xmin, bbox.ymax-bbox.ymin));
      Points pts;
      vector<Vec3f> xyz_list;
      vector<float> depth_list;
      Mat hm00;
      Mat bboxfloat;
      bboxroi.convertTo(bboxfloat, CV_32FC3, 1/255.0);
      tie(pts, xyz_list, depth_list, hm00) =
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
  ros::Timer timer_;
  torch::jit::script::Module model_;
};

} // namespace Starmap

PLUGINLIB_EXPORT_CLASS( starmap::Starmap, nodelet::Nodelet );

