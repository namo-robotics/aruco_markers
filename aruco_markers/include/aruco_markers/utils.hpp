#ifndef ARUCO_MARKERS_UTILS_HPP_
#define ARUCO_MARKERS_UTILS_HPP_

#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>

namespace aruco_markers
{
namespace utils
{
bool isVec3dZero(const cv::Vec3d & vec);
cv::aruco::PredefinedDictionaryType dictNameToEnum(const std::string & dict_name);
}     // namespace utils
} // namespace aruco_markers

#endif // ARUCO_MARKERS_UTILS_HPP_
