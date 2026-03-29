#include "aruco_markers/aruco_markers.hpp"
#include "aruco_markers/utils.hpp"
#include <rclcpp/wait_for_message.hpp>

using namespace aruco_markers;

ArucoMarkersNode::ArucoMarkersNode()
: Node("aruco_markers"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
{
  this->declare_parameter("marker_size", 0.1);
  this->declare_parameter("camera_frame", "camera_rgb_optical_frame");
  this->declare_parameter("image_topic", "camera/color/image_raw");
  this->declare_parameter("camera_info_topic", "camera/color/camera_info");
  this->declare_parameter("dictionary", "DICT_ARUCO_ORIGINAL");

  marker_size_ = this->get_parameter("marker_size").as_double();
  camera_frame_ = this->get_parameter("camera_frame").as_string();
  image_topic_ = this->get_parameter("image_topic").as_string();
  camera_info_topic_ = this->get_parameter("camera_info_topic").as_string();
  dictionary_ = this->get_parameter("dictionary").as_string();

  RCLCPP_INFO(this->get_logger(), "marker_size: %f", marker_size_);
  RCLCPP_INFO(this->get_logger(), "camera_frame: %s", camera_frame_.c_str());
  RCLCPP_INFO(this->get_logger(), "image_topic: %s", image_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "camera_info_topic: %s", camera_info_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "dictionary: %s", dictionary_.c_str());
}

void ArucoMarkersNode::initialize()
{
  RCLCPP_INFO(this->get_logger(), "Initializing.");

  // Image transport subscriber
  it_ = std::make_unique<image_transport::ImageTransport>(shared_from_this());
  image_subscriber_ = it_->subscribe(
    image_topic_, 1,
    std::bind(&ArucoMarkersNode::image_callback, this, std::placeholders::_1));

  // Publisher for marker information
  marker_info_publisher_ = this->create_publisher<std_msgs::msg::String>("aruco_marker_info", 10);
  marker_array_pub_ = this->create_publisher<aruco_markers_msgs::msg::MarkerArray>(
    "aruco/markers", 10);

  // Image publisher
  image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("aruco/result", 10);

  // TF broadcaster for publishing transformsa
  tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

  // Set up ArUco marker detector
  aruco_dict_ = cv::aruco::getPredefinedDictionary(utils::dictNameToEnum(dictionary_));
  aruco_parameters_ = cv::aruco::DetectorParameters();
  aruco_detector_ = cv::aruco::ArucoDetector(aruco_dict_, aruco_parameters_);

  RCLCPP_INFO(this->get_logger(), "Waiting for camera info.");
  sensor_msgs::msg::CameraInfo camera_info;
  rclcpp::wait_for_message<sensor_msgs::msg::CameraInfo>(
    camera_info,
    shared_from_this(), camera_info_topic_);
  RCLCPP_INFO(this->get_logger(), "Camera info received.");
  process_camera_info(camera_info);
}

void ArucoMarkersNode::process_camera_info(const sensor_msgs::msg::CameraInfo & msg)
{
  camera_matrix_ = cv::Mat(3, 3, CV_64F, (void *)msg.k.data()).clone();
  camera_distortion_ = cv::Mat::zeros(1, 4, CV_64F);
  if (!msg.d.empty()) {
    camera_distortion_ = cv::Mat(1, static_cast<int>(msg.d.size()), CV_64F);
    for (size_t i = 0; i < msg.d.size(); ++i) {
      camera_distortion_.at<double>(0, i) = msg.d[i];
    }
  }

  if (!received_camera_info_) {
    RCLCPP_INFO(this->get_logger(), "Received camera info.");
    RCLCPP_INFO(
      this->get_logger(), "Camera Info:\n"
      "\tWidth: %d\n"
      "\tHeight: %d\n"
      "\tK (intrinsic matrix): [%f, %f, %f, %f, %f, %f, %f, %f, %f]\n"
      "\tD (distortion coefficients): [%f, %f, %f, %f, %f]",
      msg.width,
      msg.height,
      msg.k[0], msg.k[1], msg.k[2], msg.k[3], msg.k[4], msg.k[5], msg.k[6], msg.k[7], msg.k[8],
      msg.d[0], msg.d[1], msg.d[2], msg.d[3], msg.d[4]);
    received_camera_info_ = true;
  }
}

void ArucoMarkersNode::log_marker_ids(const std::vector<int> & ids)
{
  std::stringstream ss;
  for (size_t i = 0; i < ids.size(); ++i) {
    ss << ids[i];
    if (i != ids.size() - 1) {
      ss << ", "; // Add a separator between elements
    }
  }
  RCLCPP_INFO(this->get_logger(), "marker ids: %s", ss.str().c_str());
}

void ArucoMarkersNode::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  if (!received_camera_info_) {
    RCLCPP_INFO(this->get_logger(), "Waiting for camera info.");
    return;
  }

  aruco_markers_msgs::msg::MarkerArray marker_array;
  marker_array.header.stamp = this->get_clock()->now();
  marker_array.header.frame_id = camera_frame_;

  try {
    // Convert ROS image message to OpenCV image
    cv_bridge::CvImagePtr cv_image_ptr = cv_bridge::toCvCopy(
      msg,
      sensor_msgs::image_encodings::BGR8);
    cv::Mat & image = cv_image_ptr->image;

    // Detect ArUco markers
    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners, rejected_candidates;
    aruco_detector_.detectMarkers(image, marker_corners, marker_ids, rejected_candidates);

    if (!marker_ids.empty()) {
      // Estimate the pose of the ArUco markers (using solvePnP)
      std::vector<cv::Vec3d> tvecs;
      std::vector<cv::Vec3d> rvecs;

      // Define marker 3D object points (corners in marker local frame)
      const float half_size = static_cast<float>(marker_size_) / 2.0f;
      std::vector<cv::Point3f> obj_points = {
        {-half_size, half_size, 0.0f},
        {half_size, half_size, 0.0f},
        {half_size, -half_size, 0.0f},
        {-half_size, -half_size, 0.0f}
      };

      for (size_t i = 0; i < marker_corners.size(); ++i) {
        cv::Vec3d rvec, tvec;
        cv::solvePnP(
          obj_points, marker_corners[i], camera_matrix_, camera_distortion_, rvec, tvec,
          false, cv::SOLVEPNP_IPPE_SQUARE);
        rvecs.push_back(rvec);
        tvecs.push_back(tvec);
      }

      if (tvecs.empty() || rvecs.empty()) {
        RCLCPP_WARN(this->get_logger(), "Pose estimation failed for marker.");
        return;
      }

      for (size_t i = 0; i < marker_ids.size(); ++i) {
        const cv::Vec3d & rvec = rvecs[i]; // Rotation vector for marker i
        const cv::Vec3d & tvec = tvecs[i]; // Translation vector for marker i

        if (utils::isVec3dZero(tvec)) {
          RCLCPP_WARN(this->get_logger(), "tvec is zero");
          continue;
        }
        // Broadcast transform from 'camera_frame' to 'aruco_marker_<id>'
        geometry_msgs::msg::TransformStamped marker_transform;
        marker_transform.header.stamp = this->get_clock()->now();
        marker_transform.header.frame_id = camera_frame_;                                  // Parent frame
        marker_transform.child_frame_id = "aruco_marker_" + std::to_string(marker_ids[i]); // Marker-specific frame
        marker_transform.transform.translation.x = tvec[0];
        marker_transform.transform.translation.y = tvec[1];
        marker_transform.transform.translation.z = tvec[2];

        tf2::Quaternion quaternion;
        cv::Mat rotation_matrix;
        cv::Rodrigues(rvec, rotation_matrix); // Convert rvec to a rotation matrix
        tf2::Matrix3x3 tf_rotation_matrix(
          rotation_matrix.at<double>(0, 0), rotation_matrix.at<double>(0, 1),
          rotation_matrix.at<double>(0, 2),
          rotation_matrix.at<double>(1, 0), rotation_matrix.at<double>(1, 1),
          rotation_matrix.at<double>(1, 2),
          rotation_matrix.at<double>(2, 0), rotation_matrix.at<double>(2, 1),
          rotation_matrix.at<double>(2, 2));
        tf_rotation_matrix.getRotation(quaternion);

        marker_transform.transform.rotation.x = quaternion.getX();
        marker_transform.transform.rotation.y = quaternion.getY();
        marker_transform.transform.rotation.z = quaternion.getZ();
        marker_transform.transform.rotation.w = quaternion.getW();

        tf_broadcaster_->sendTransform(marker_transform);

        // Convert to geometry_msgs::Pose
        geometry_msgs::msg::PoseStamped marker_pose;
        marker_pose.header.stamp = msg->header.stamp;
        marker_pose.header.frame_id = camera_frame_;
        marker_pose.pose.position.x = marker_transform.transform.translation.x;
        marker_pose.pose.position.y = marker_transform.transform.translation.y;
        marker_pose.pose.position.z = marker_transform.transform.translation.z;
        marker_pose.pose.orientation.x = marker_transform.transform.rotation.x;
        marker_pose.pose.orientation.y = marker_transform.transform.rotation.y;
        marker_pose.pose.orientation.z = marker_transform.transform.rotation.z;
        marker_pose.pose.orientation.w = marker_transform.transform.rotation.w;

        // Populate Marker message
        aruco_markers_msgs::msg::Marker marker;
        marker.header.frame_id = camera_frame_;
        marker.header.stamp = msg->header.stamp;
        marker.id = marker_ids[i];
        marker.pose = marker_pose;
        marker.pixel_x = marker_corners[i][0].x;
        marker.pixel_y = marker_corners[i][0].y;

        // Add marker to array
        marker_array.markers.push_back(marker);

        // Draw 3D axis on the marker in the image
        cv::drawFrameAxes(
          image, camera_matrix_, camera_distortion_, rvec, tvec,
          static_cast<float>(marker_size_) * 0.7f);
        draw3dAxis(image, tvec, rvec, 1);
      }

      cv::aruco::drawDetectedMarkers(image, marker_corners, marker_ids);
    }

    // Convert OpenCV image back to ROS message
    auto overlay_msg = cv_bridge::CvImage(msg->header, "bgr8", image).toImageMsg();
    image_pub_->publish(*overlay_msg);

    // Publish the marker array
    if (!marker_array.markers.empty()) {
      marker_array_pub_->publish(marker_array);
    }
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "CV Bridge exception: %s", e.what());
  } catch (const tf2::TransformException & e) {
    RCLCPP_WARN(this->get_logger(), "TF2 exception: %s", e.what());
  }
}

void ArucoMarkersNode::logCvMat(const cv::Mat & mat, const std::string & name)
{
  std::cout << name << " (" << mat.rows << "x" << mat.cols << ", type=" << mat.type() << "):\n";

  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      std::cout << mat.at<double>(i, j) << " ";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
}

void ArucoMarkersNode::logVec3d(const cv::Vec3d & vec, const std::string & name)
{
  std::cout << name << " Vec3d(" << vec[0] << ", " << vec[1] << ", " << vec[2] << ")" << std::endl;
}

void ArucoMarkersNode::draw3dAxis(
  cv::Mat & Image, const cv::Vec3d & tvec, const cv::Vec3d & rvec,
  int lineSize)
{
  float size = marker_size_ * 0.6;
  cv::Mat objectPoints(4, 3, CV_32FC1);

  // origin
  objectPoints.at<float>(0, 0) = 0;
  objectPoints.at<float>(0, 1) = 0;
  objectPoints.at<float>(0, 2) = 0;

  // (1,0,0)
  objectPoints.at<float>(1, 0) = size;
  objectPoints.at<float>(1, 1) = 0;
  objectPoints.at<float>(1, 2) = 0;

  // (0,1,0)
  objectPoints.at<float>(2, 0) = 0;
  objectPoints.at<float>(2, 1) = size;
  objectPoints.at<float>(2, 2) = 0;

  // (0,0,1)
  objectPoints.at<float>(3, 0) = 0;
  objectPoints.at<float>(3, 1) = 0;
  objectPoints.at<float>(3, 2) = size;

  std::vector<cv::Point2f> imagePoints;
  cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);

  cv::projectPoints(objectPoints, rvec, tvec, camera_matrix_, dist_coeffs, imagePoints);
  cv::line(Image, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255, 255), lineSize);
  cv::line(Image, imagePoints[0], imagePoints[2], cv::Scalar(0, 255, 0, 255), lineSize);
  cv::line(Image, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0, 255), lineSize);

  putText(
    Image, "x", imagePoints[1], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255, 255),
    2);
  putText(
    Image, "y", imagePoints[2], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0, 255),
    2);
  putText(
    Image, "z", imagePoints[3], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0, 255),
    2);
}
