/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

#include "myslam/common_include.h"
#include "myslam/map.h"
#include "myslam/frame.h"

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/sfm/fundamental.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <stdlib.h>

namespace myslam
{
class VisualOdometry: public enable_shared_from_this<VisualOdometry>
{
public:
    typedef shared_ptr<VisualOdometry> Ptr;
    enum VOState {
        INITIALIZING=-1,
        OK=0,
        LOST
    };

    VOState     state_;     // current VO status
    Map::Ptr    map_;       // map with all frames and map points

    Frame::Ptr  ref_;       // reference key-frame
    Frame::Ptr  curr_;      // current frame

    cv::Ptr<cv::ORB> orb_;  // orb detector and computer
    cv::Ptr<cv::xfeatures2d::SurfFeatureDetector> surf_;
    cv::Ptr<cv::xfeatures2d::SurfDescriptorExtractor> surf_desc_;
    vector<cv::KeyPoint>    keypoints_curr_;    // keypoints in current frame
    vector<cv::KeyPoint>    keypoints_prev_;    // keypoints in current frame
    Mat                     descriptors_curr_;  // descriptor in current frame
    Mat                     descriptors_prev_;  // descriptor in current frame

    //Minwoo start
    cv::Ptr<cv::DescriptorMatcher> matcher;
    Frame::Ptr InitialFrame_;
    std::vector<cv::Point3f> vIniP3D_;


    //Minwoo end
    cv::FlannBasedMatcher   matcher_flann_;     // flann matcher
    vector<MapPoint::Ptr>   match_3dpts_;       // matched 3d points
    vector<int>             match_2dkp_index_;  // matched 2d pixels (index of kp_curr)

    SE3 T_c_w_estimated_;    // the estimated pose of current frame
    int num_inliers_;        // number of inlier features in icp
    int num_lost_;           // number of lost times

    // parameters
    int num_of_features_;   // number of features
    double scale_factor_;   // scale in image pyramid
    int level_pyramid_;     // number of pyramid levels
    float match_ratio_;     // ratio for selecting  good matches
    int max_num_lost_;      // max number of continuous lost times
    int min_inliers_;       // minimum inliers
    double key_frame_min_rot;   // minimal rotation of two key-frames
    double key_frame_min_trans; // minimal translation of two key-frames
    double  map_point_erase_ratio_; // remove map point ratio

public: // functions
    VisualOdometry();
    ~VisualOdometry();

    bool addFrame( Frame::Ptr frame );      // add a new frame

protected:
    // inner operation
    void extractKeyPoints();
    void computeDescriptors();
    void featureMatching();
    void poseEstimationPnP();
    void optimizeMap();

    void addKeyFrame();
    void addMapPoints();
    bool checkEstimatedPose();
    bool checkKeyFrame();

    double getViewAngle( Frame::Ptr frame, MapPoint::Ptr point );
    bool FindMotion(const Mat& F, const Mat& K,
                                 const vector<cv::Point2f>& pts1, const vector<cv::Point2f>& pts2,
                                 const Mat& inlierMask, SE3 &pose,
                                 //vector<Vector3d> &vP3D,
                                 vector<cv::Point3f> &vP3D,
                                 vector<bool> vbTriangulated,
                                 float minParallax, int minTriangulated);

    int countGoodDecompose(const cv::Mat& R,
                           const cv::Mat& t,
                           const vector<cv::Point2f>& pts1,
                           const vector<cv::Point2f>& pts2,
                           const Mat& inliers,
                           //vector<Eigen::Vector3d> &vP3D,
                           vector<cv::Point3f> &vP3D,
                           const float& th2,
                           const cv::Mat& K,
                           vector<bool>& vbGood, float& parallax);

    typedef Eigen::Matrix<double,3,4> Projection;
    void Triangulate(const cv::Point2f p1,const cv::Point2f p2, Projection P1, Projection P2, Eigen::Vector3d& p3dC1);


    //Fundamental matrix Find
    void FindFundamental(std::vector<bool> &vbMatchesInliers, float &score,
                         cv::Mat &F21, std::vector<cv::Point2f> &src, std::vector<cv::Point2f> &dst);

    void Normalize(const vector<cv::Point2f>& pts, vector<cv::Point2f> &res, Mat &T);
    cv::Mat ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);
    float CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers,
                                       const vector<cv::Point2f> &mvKeys1, const vector<cv::Point2f> &mvKeys2,
                                        float sigma);
    Vector3d Down(Eigen::Matrix3d M);
void drawepipolarlines(const std::string& title, const cv::Mat &F,
                const cv::Mat& img1, const cv::Mat& img2,
                const std::vector<cv::Point2f> points1,
                const std::vector<cv::Point2f> points2,
                const float inlierDistance = -1);
float distancePointLine(const cv::Point2f& point, const cv::Point3f& line);
void Triangulate(const cv::Point2f &p1, const cv::Point2f &p2, const cv::Mat &P1, const cv::Mat &P2, Mat &x3D);

void CreateInitialMapMonocular();
};
}

#endif // VISUALODOMETRY_H
