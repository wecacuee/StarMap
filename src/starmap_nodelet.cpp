#include <memory>
#include <functional>

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
  virtual void onInit(){
    namespace sph = std::placeholders; // for _1, _2, ...
    std::string image_topic, bbox_topic, keypoint_topic;
    auto nh = getNodeHandle();
    auto private_nh = getPrivateNodeHandle();
    if ( ! nh.param("image_topic", image_topic, std::string("image")))
      throw std::runtime_error("Need image_topic");
    if ( ! nh.param("bbox_topic", bbox_topic, std::string("bounding_boxes")))
      throw std::runtime_error("Need bbox_topic");
    if ( ! nh.param("keypoint_topic", keypoint_topic, std::string("keypoints")))
      throw std::runtime_error("Need keypoint_topic");
    timer_ = nh.createTimer(ros::Duration(1.0),
                            std::bind(& Starmap::timerCb, this, sph::_1));
    image_sub_ = make_unique<message_filters::Subscriber<sensor_msgs::Image>>(nh, image_topic, 1);
    bbox_sub_ = make_unique<message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes>>(nh, bbox_topic, 1);
    sub_ = make_unique<message_filters::TimeSynchronizer<sensor_msgs::Image, darknet_ros_msgs::BoundingBoxes>>(*image_sub_, *bbox_sub_, 10);
    sub_->registerCallback(std::bind(&Starmap::messageCb, this, sph::_1, sph::_2));
    pub_ = private_nh.advertise<std_msgs::String>(keypoint_topic, 10);
  };

  void timerCb(const ros::TimerEvent& event){
    // Using timers is the preferred 'ROS way' to manual threading
    NODELET_INFO_STREAM("The time is now " << event.current_real);
  }

  // must use a ConstPtr callback to use zero-copy transport
  void messageCb(const sensor_msgs::ImageConstPtr& message,
                 const darknet_ros_msgs::BoundingBoxesConstPtr& bboxes) {
    std::cerr << "message: " << message << "bboxes: " << bboxes << "\n";
  }

  std::unique_ptr<message_filters::Subscriber<sensor_msgs::Image>> image_sub_;
  std::unique_ptr<message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes>> bbox_sub_;
  std::unique_ptr<message_filters::TimeSynchronizer<sensor_msgs::Image, darknet_ros_msgs::BoundingBoxes>> sub_;
  ros::Publisher pub_;
  ros::Timer timer_;
};

} // namespace Starmap

PLUGINLIB_EXPORT_CLASS( starmap::Starmap, nodelet::Nodelet );

