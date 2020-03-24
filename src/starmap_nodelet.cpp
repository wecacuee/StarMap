#include <memory>
#include <functional>
#include <mutex>
#include <queue>

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
#include <sort_ros/TrackedBoundingBoxes.h>
#include <cv_bridge/cv_bridge.h>
#include <starmap/SemanticKeypointWithCovariance.h>
#include <starmap/TrackedBBoxListWithKeypoints.h>
#include <starmap/TrackedBBoxWithKeypoints.h>
#include "starmap/starmap.h"
#include "boost/filesystem.hpp"
#include "image_transport/subscriber_filter.h"

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
    Starmap() : sub_(10) {}

  private:
  virtual void onInit() {
    NODELET_DEBUG("Initializing ");
    namespace sph = std::placeholders; // for _1, _2, ...
    std::string image_topic, bbox_topic, keypoint_topic, visualization_topic,
    starmap_model_path;
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
    private_nh.param<std::string>("visualization_topic", visualization_topic, "visualization");
    private_nh.param<int>("gpu_id", gpu_id, -1);
    timer_ = nh.createTimer(ros::Duration(0.03),
                            std::bind(& Starmap::timerCb, this, sph::_1));

    // model = torch.load(opt.loadModel)
    NODELET_DEBUG("Loading  model from %s", starmap_model_path.c_str());
    model_ = torch::jit::load(starmap_model_path);
    NODELET_DEBUG("gpu_id : %d", gpu_id);
    torch::DeviceType device_type = gpu_id >= 0 ? torch::DeviceType::CUDA : torch::DeviceType::CPU;
    int device_id = gpu_id >= 0 ? gpu_id : 0;
    torch::Device device = torch::Device(device_type, gpu_id);
    model_.to(device);

    NODELET_DEBUG("Subscribing to %s", image_topic.c_str());
    image_trans_ = make_unique<image_transport::ImageTransport>(private_nh);
    image_sub_.subscribe(nh, image_topic, 10);
    bbox_sub_.subscribe(nh, bbox_topic, 10);
    sub_.connectInput(image_sub_, bbox_sub_);
    sub_.registerCallback(std::bind(&Starmap::messageCb, this, sph::_1, sph::_2));
    pub_ = private_nh.advertise<starmap::TrackedBBoxListWithKeypoints>(keypoint_topic, 10);

    vis_ = image_trans_->advertise(visualization_topic, 10);

  }

  cv::Rect2i safe_rect_bbox(const sort_ros::TrackedBoundingBox& bbox,
                            const cv::Mat& image) {
    int xmin = max<int>(0, bbox.xmin);
    int ymin = max<int>(0, bbox.ymin);
    int xmax = min<int>(image.cols, bbox.xmax);
    int ymax = min<int>(image.rows, bbox.ymax);
    cv::Rect2i bbox_rect(xmin, ymin, xmax-xmin, ymax - ymin);
    return bbox_rect;
  }

  void  visualize_all_bbox(cv::Mat& image,
                           const TrackedBBoxListWithKeypointsConstPtr& bbox_with_kp_list)
  {
    for (auto& bbox_with_kp : bbox_with_kp_list->bounding_boxes) {
      auto bbox_rect = safe_rect_bbox(bbox_with_kp.bbox, image);
      if (bbox_rect.area() < 1)
        continue;
      cv::rectangle(image, bbox_rect, cv::Scalar(0, 255, 0));
      auto bboxroi = image(bbox_rect);
      Points pts;
      std::vector<cv::Vec3f> colors;
      for (auto& semkp: bbox_with_kp.keypoints) {
        pts.emplace_back(semkp.x, semkp.y);
        colors.emplace_back(semkp.cov[0], semkp.cov[0]);
      }
      starmap::visualize_keypoints(bboxroi, pts, colors);
    }
  }


  // must use a ConstPtr callback to use zero-copy transport
  void messageCb(const sensor_msgs::ImageConstPtr& message,
                 const sort_ros::TrackedBoundingBoxesConstPtr& bboxes) {
    NODELET_DEBUG("Callback called ... ");
    auto private_nh = getPrivateNodeHandle();
    int input_res;
    private_nh.param<int>("input_res", input_res, 256);
    bool visualize;
    private_nh.param<bool>("visualize", visualize, false);
    cv_bridge::CvImageConstPtr img = cv_bridge::toCvShare(message);
    starmap::TrackedBBoxListWithKeypointsPtr bbox_with_kp_list =
      boost::make_shared<starmap::TrackedBBoxListWithKeypoints>();
    bbox_with_kp_list->header.stamp = message->header.stamp;
    bbox_with_kp_list->header.frame_id = message->header.frame_id;
    for (auto& bbox: bboxes->bounding_boxes) {
      auto bbox_rect = safe_rect_bbox(bbox, img->image);
      starmap::TrackedBBoxWithKeypoints bbox_with_kp;
      if (bbox_rect.area() >= 1) {
        auto bboxroi = img->image(bbox_rect);
        Points pts;
        vector<Vec3f> xyz_list;
        vector<float> depth_list;
        vector<float> hm_list;
        Mat bboxfloat;
        bboxroi.convertTo(bboxfloat, CV_32FC3, 1/255.0);
        NODELET_DEBUG("Calling  model ... ");
        tie(pts, xyz_list, depth_list, hm_list) =
          find_semantic_keypoints_prob_depth(model_, bboxfloat, input_res,
                                            /*visualize=*/visualize);

        bbox_with_kp.bbox = bbox; // Duplicate information
        for (size_t i: boost::counting_range<size_t>(0, pts.size())) {
          auto& pt = pts[i];
          starmap::SemanticKeypointWithCovariance kpt;
          kpt.x = pt.x;
          kpt.y = pt.y;
          kpt.cov.insert(kpt.cov.end(), {hm_list[i], 0, 0, hm_list[i]});
          bbox_with_kp.keypoints.push_back(kpt);
        }
      }
      bbox_with_kp_list->bounding_boxes.push_back(bbox_with_kp);
    }
    pub_.publish(bbox_with_kp_list);
    if (vis_.getNumSubscribers() >= 1) {
      cv::Mat vis = img->image.clone();
      visualize_all_bbox(vis, bbox_with_kp_list);
      cv_bridge::CvImage cvImage;
      cvImage.header.stamp = bbox_with_kp_list->header.stamp;
      cvImage.header.frame_id = bbox_with_kp_list->header.frame_id;
      cvImage.encoding = sensor_msgs::image_encodings::BGR8;
      cvImage.image = vis;
      {
        const std::lock_guard<std::mutex> lock(image_to_publish_mutex_);
        image_to_publish_queue_.push(cvImage.toImageMsg());
        // vis_.publish(*cvImage.toImageMsg());
      }
    }
  }

  void timerCb(const ros::TimerEvent &) {
    const std::lock_guard<std::mutex> lock(image_to_publish_mutex_);
    if ( ! image_to_publish_queue_.empty()) {
      NODELET_DEBUG("publishing visualization... ");
      vis_.publish(image_to_publish_queue_.front());
      image_to_publish_queue_.pop();
    }
  }

  message_filters::Subscriber<sensor_msgs::Image> image_sub_;
    //image_transport::SubscriberFilter image_sub_;
  message_filters::Subscriber<sort_ros::TrackedBoundingBoxes> bbox_sub_;
  message_filters::TimeSynchronizer<sensor_msgs::Image, sort_ros::TrackedBoundingBoxes> sub_;
  ros::Publisher pub_;
  image_transport::Publisher vis_;
  std::unique_ptr<image_transport::ImageTransport> image_trans_;
  ros::Timer timer_;
  torch::jit::script::Module model_;
  std::queue<sensor_msgs::ImageConstPtr> image_to_publish_queue_;
  std::mutex image_to_publish_mutex_;
};

} // namespace Starmap

PLUGINLIB_EXPORT_CLASS( starmap::Starmap, nodelet::Nodelet );

