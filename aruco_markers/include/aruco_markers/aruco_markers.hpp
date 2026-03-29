#ifndef ARUCO_MARKERS_HPP_
#define ARUCO_MARKERS_HPP_

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <aruco_markers_msgs/msg/marker.hpp>
#include <aruco_markers_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "aruco_markers/utils.hpp"

#include <memory>
#include <string>
#include <unordered_map>

using namespace std::chrono_literals;

class ArucoMarkersNode : public rclcpp::Node
{
public:
  ArucoMarkersNode();
  void initialize();
  void process_camera_info(const sensor_msgs::msg::CameraInfo & msg);

  bool hasReceivedCameraInfo() const
  {
    return received_camera_info_;
  }

  cv::Mat getCameraMatrix() const
  {
    return camera_matrix_;
  }

  cv::Mat getCameraDistortion() const
  {
    return camera_distortion_;
  }

private:
  void log_marker_ids(const std::vector<int> & ids);
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
  void logCvMat(const cv::Mat & mat, const std::string & name);
  void logVec3d(const cv::Vec3d & vec, const std::string & name);
  void draw3dAxis(cv::Mat & Image, const cv::Vec3d & tvec, const cv::Vec3d & rvec, int lineSize);
  cv::aruco::PREDEFINED_DICTIONARY_NAME dictNameToEnum(const std::string & dict_name);

  // ROS 2 Publisher for ArUco marker info
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr marker_info_publisher_;
  rclcpp::Publisher<aruco_markers_msgs::msg::MarkerArray>::SharedPtr marker_array_pub_;

  // Image subscriber (using image_transport)
  std::unique_ptr<image_transport::ImageTransport> it_;
  image_transport::Subscriber image_subscriber_;

  // Camera info subscriber
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_subscriber_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;

  // TF broadcaster
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // ArUco marker detector variables
  cv::aruco::Dictionary aruco_dict_;
  cv::aruco::DetectorParameters aruco_parameters_;
  cv::aruco::ArucoDetector aruco_detector_;

  cv::Mat camera_matrix_;
  cv::Mat camera_distortion_;
  bool received_camera_info_;
  double marker_size_;
  std::string camera_frame_;
  std::string image_topic_;
  std::string camera_info_topic_;
  std::string dictionary_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
};

#endif // ARUCO_MARKERS_HPP_
