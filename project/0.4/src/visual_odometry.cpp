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

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <boost/timer.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"
#include "myslam/g2o_types.h"
#include "myslam/initializer.h"

namespace myslam
{

VisualOdometry::VisualOdometry() :
    state_ ( INITIALIZING ), ref_ ( nullptr ), curr_ ( nullptr ), map_ ( new Map ), num_lost_ ( 0 ), num_inliers_ ( 0 ), matcher_flann_ ( new cv::flann::LshIndexParams ( 5,10,2 ) )
{
    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    num_of_features_    = Config::get<int> ( "number_of_features" );
    scale_factor_       = Config::get<double> ( "scale_factor" );
    level_pyramid_      = Config::get<int> ( "level_pyramid" );
    match_ratio_        = Config::get<float> ( "match_ratio" );
    max_num_lost_       = Config::get<float> ( "max_num_lost" );
    min_inliers_        = Config::get<int> ( "min_inliers" );
    key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );
    key_frame_min_trans = Config::get<double> ( "keyframe_translation" );
    map_point_erase_ratio_ = Config::get<double> ( "map_point_erase_ratio" );
    orb_ = cv::ORB::create ( num_of_features_, scale_factor_, level_pyramid_ );
}

VisualOdometry::~VisualOdometry()
{

}

bool VisualOdometry::addFrame ( Frame::Ptr frame ) //Tracking module
{
    switch ( state_ )
    {
    case INITIALIZING:
    {
        curr_ = frame;
        // extract features from first frame and add them into map
        extractKeyPoints();
        if(keypoints_curr_.size() > 100)
        {
            computeDescriptors();
            if(ref_ == nullptr)
            {
                ref_ = std::move(curr_);
                keypoints_prev_ = std::move(keypoints_curr_);
                descriptors_prev_ = std::move(descriptors_curr_);
                return false;
            }
            else if(curr_ != nullptr && ref_ != nullptr)
            {
                std::vector<cv::DMatch> good_matches;
                /*
                std::vector< std::vector<cv::DMatch> > knn_matches;
                cout<<" is good?"<<endl;
                if(descriptors_prev_.type()!=CV_32F) {
                descriptors_prev_.convertTo(descriptors_prev_, CV_32F);
                }

                if(descriptors_curr_.type()!=CV_32F) {
                descriptors_curr_.convertTo(descriptors_curr_, CV_32F);
                }
                matcher->knnMatch(descriptors_prev_,descriptors_curr_, knn_matches,2);
                const float ratio_thres = 0.7f;

                cout<<" is good?"<<endl;
                for(size_t i =0; i<knn_matches.size(); i++)
                {
                    if(knn_matches[i][0].distance < ratio_thres* knn_matches[i][1].distance)
                        good_matches.push_back(knn_matches[i][0]);
                }
                cout<<" is good?"<<endl;
                */

                matcher_flann_.match(descriptors_prev_,
                                     descriptors_curr_,
                                     good_matches);
                /*

                cout<<" Okay matches:"<<good_matches.size()<<endl;
                cout<<" size of prev:"<<keypoints_prev_.size()<<endl;
                cout<<" size of prev:"<<keypoints_curr_.size()<<endl;

                Mat img_matches;
                cv::drawMatches(curr_->color_, keypoints_curr_,
                                ref_->color_, keypoints_prev_,
                                good_matches, img_matches,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                cout<<" is good?"<<endl;
                cv::imshow("Draw Matches", img_matches);
                cv::waitKey(0);
                */
                vector<cv::Point2d> pts1(good_matches.size());
                vector<cv::Point2d> pts2(good_matches.size());
                for(int i=0;i < good_matches.size(); i++)
                {
                    pts1.push_back(keypoints_prev_[good_matches[i].queryIdx].pt);
                    pts2.push_back(keypoints_curr_[good_matches[i].trainIdx].pt);
                }


                Mat R, t;
                double lambda;
                Mat inlierMask;
                Mat E = cv::findEssentialMat(pts1,pts2,curr_->camera_->mK,cv::RANSAC,0.999,1.0,inlierMask);
                vector<Eigen::Vector3d> vP3D;
                vector<bool> vbTriangulated;

                if(FindMotionFromEssential(E, curr_->camera_->mK,
                                           pts1, pts2,
                                           inlierMask, curr_->T_c_w_,
                                           vP3D, vbTriangulated,
                                           1.0, 50))
                {
                    std::cout<<"Pose = "<<curr_->T_c_w_.rotation_matrix()<<"\n"<<curr_->T_c_w_.translation()<<std::endl;
                }
                cout<<"Ok???"<<endl;


                ref_ = std::move(curr_);
                keypoints_prev_ = std::move(keypoints_curr_);
                descriptors_prev_ = std::move(descriptors_curr_);

                return false;

            }


        }
        addKeyFrame();      // the first frame is a key-frame

        descriptors_prev_ = std::move(descriptors_curr_);
        break;
    }
    case OK:
    {
        curr_ = frame;
        curr_->T_c_w_ = ref_->T_c_w_;
        extractKeyPoints();
        computeDescriptors();               // Constant Velocity Model Tracker
        featureMatching();                  // Constant Velocity Model Tracker
        poseEstimationPnP();                // Triangulation (Essential matrix Decomp)
        if ( checkEstimatedPose() == true ) // a good estimation
        {
            curr_->T_c_w_ = T_c_w_estimated_;
            optimizeMap();
            num_lost_ = 0;
            if ( checkKeyFrame() == true ) // is a key-frame
            {
                addKeyFrame();
            }
        }
        else // bad estimation due to various reasons
        {
            num_lost_++;
            if ( num_lost_ > max_num_lost_ )
            {
                state_ = LOST;
            }
            return false;
        }
        break;
    }
    case LOST:
    {
        cout<<"vo has lost."<<endl;
        break;
    }
    }

    return true;
}

void VisualOdometry::extractKeyPoints()
{
    boost::timer timer;
    orb_->detect ( curr_->color_, keypoints_curr_ );
    cout<<"extract keypoints cost time: "<<timer.elapsed() <<endl;
}

void VisualOdometry::computeDescriptors()
{
    boost::timer timer;
    orb_->compute ( curr_->color_, keypoints_curr_, descriptors_curr_ );
    cout<<"descriptor computation cost time: "<<timer.elapsed() <<endl;
}

void VisualOdometry::featureMatching()
{
    boost::timer timer;
    vector<cv::DMatch> matches;
    // select the candidates in map
    Mat desp_map;
    vector<MapPoint::Ptr> candidate;
    for ( auto& allpoints: map_->map_points_ )
    {
        MapPoint::Ptr& p = allpoints.second;
        // check if p in curr frame image
        if ( curr_->isInFrame(p->pos_) )
        {
            // add to candidate
            p->visible_times_++;
            candidate.push_back( p );
            desp_map.push_back( p->descriptor_ );
        }
    }

    matcher_flann_.match ( desp_map, descriptors_curr_, matches );
    // select the best matches
    float min_dis = std::min_element (
                        matches.begin(), matches.end(),
                        [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
    {
        return m1.distance < m2.distance;
    } )->distance;

    match_3dpts_.clear();
    match_2dkp_index_.clear();
    for ( cv::DMatch& m : matches )
    {
        if ( m.distance < max<float> ( min_dis*match_ratio_, 30.0 ) )
        {
            match_3dpts_.push_back( candidate[m.queryIdx] );
            match_2dkp_index_.push_back( m.trainIdx );
        }
    }
    cout<<"good matches: "<<match_3dpts_.size() <<endl;
    cout<<"match cost time: "<<timer.elapsed() <<endl;
}

void VisualOdometry::poseEstimationPnP()
{
    // construct the 3d 2d observations
    vector<cv::Point3f> pts3d;
    vector<cv::Point2f> pts2d;

    for ( int index:match_2dkp_index_ )
    {
        pts2d.push_back ( keypoints_curr_[index].pt );
    }
    for ( MapPoint::Ptr pt:match_3dpts_ )
    {
        pts3d.push_back( pt->getPositionCV() );
    }

    Mat K = ( cv::Mat_<double> ( 3,3 ) <<
              ref_->camera_->fx_, 0, ref_->camera_->cx_,
              0, ref_->camera_->fy_, ref_->camera_->cy_,
              0,0,1
            );
    Mat rvec, tvec, inliers;
    cv::solvePnPRansac ( pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );
    num_inliers_ = inliers.rows;
    cout<<"pnp inliers: "<<num_inliers_<<endl;
    T_c_w_estimated_ = SE3 (
                           SO3 ( rvec.at<double> ( 0,0 ), rvec.at<double> ( 1,0 ), rvec.at<double> ( 2,0 ) ),
                           Vector3d ( tvec.at<double> ( 0,0 ), tvec.at<double> ( 1,0 ), tvec.at<double> ( 2,0 ) )
                       );

    // using bundle adjustment to optimize the pose
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> Block;
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block* solver_ptr = new Block ( linearSolver );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setId ( 0 );
    pose->setEstimate ( g2o::SE3Quat (
        T_c_w_estimated_.rotation_matrix(), T_c_w_estimated_.translation()
    ));
    optimizer.addVertex ( pose );

    // edges
    for ( int i=0; i<inliers.rows; i++ )
    {
        int index = inliers.at<int> ( i,0 );
        // 3D -> 2D projection
        EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
        edge->setId ( i );
        edge->setVertex ( 0, pose );
        edge->camera_ = curr_->camera_.get();
        edge->point_ = Vector3d ( pts3d[index].x, pts3d[index].y, pts3d[index].z );
        edge->setMeasurement ( Vector2d ( pts2d[index].x, pts2d[index].y ) );
        edge->setInformation ( Eigen::Matrix2d::Identity() );
        optimizer.addEdge ( edge );
        // set the inlier map points
        match_3dpts_[index]->matched_times_++;
    }

    optimizer.initializeOptimization();
    optimizer.optimize ( 10 );

    T_c_w_estimated_ = SE3 (
        pose->estimate().rotation(),
        pose->estimate().translation()
    );

    cout<<"T_c_w_estimated_: "<<endl<<T_c_w_estimated_.matrix()<<endl;
}

bool VisualOdometry::checkEstimatedPose()
{
    // check if the estimated pose is good
    if ( num_inliers_ < min_inliers_ )
    {
        cout<<"reject because inlier is too small: "<<num_inliers_<<endl;
        return false;
    }
    // if the motion is too large, it is probably wrong
    SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    if ( d.norm() > 5.0 )
    {
        cout<<"reject because motion is too large: "<<d.norm() <<endl;
        return false;
    }
    return true;
}

bool VisualOdometry::checkKeyFrame()
{
    SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
        return true;
    return false;
}

void VisualOdometry::addKeyFrame()
{
    if ( map_->keyframes_.empty() )
    {
        // first key-frame, add all 3d points into map
        for ( size_t i=0; i<keypoints_curr_.size(); i++ )
        {
            double d = curr_->findDepth ( keypoints_curr_[i] );
            if ( d < 0 )
                continue;
            Vector3d p_world = ref_->camera_->pixel2world (
                Vector2d ( keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y ), curr_->T_c_w_, d
            );
            Vector3d n = p_world - ref_->getCamCenter();
            n.normalize();
            MapPoint::Ptr map_point = MapPoint::createMapPoint(
                p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
            );
            map_->insertMapPoint( map_point );
        }
    }

    map_->insertKeyFrame ( curr_ );
    ref_ = curr_;
}

void VisualOdometry::addMapPoints()
{
    // add the new map points into map
    vector<bool> matched(keypoints_curr_.size(), false);
    for ( int index:match_2dkp_index_ )
        matched[index] = true;
    for ( int i=0; i<keypoints_curr_.size(); i++ )
    {
        if ( matched[i] == true )
            continue;
        double d = ref_->findDepth ( keypoints_curr_[i] );
        if ( d<0 )
            continue;
        Vector3d p_world = ref_->camera_->pixel2world (
            Vector2d ( keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y ),
            curr_->T_c_w_, d
        );
        Vector3d n = p_world - ref_->getCamCenter();
        n.normalize();
        MapPoint::Ptr map_point = MapPoint::createMapPoint(
            p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
        );
        map_->insertMapPoint( map_point );
    }
}

void VisualOdometry::optimizeMap()
{
    // remove the hardly seen and no visible points
    for ( auto iter = map_->map_points_.begin(); iter != map_->map_points_.end(); )
    {
        if ( !curr_->isInFrame(iter->second->pos_) )
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }
        float match_ratio = float(iter->second->matched_times_)/iter->second->visible_times_;
        if ( match_ratio < map_point_erase_ratio_ )
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }

        double angle = getViewAngle( curr_, iter->second );
        if ( angle > M_PI/6. )
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }
        if ( iter->second->good_ == false )
        {
            // TODO try triangulate this map point
        }
        iter++;
    }

    if ( match_2dkp_index_.size()<100 )
        addMapPoints();
    if ( map_->map_points_.size() > 1000 )
    {
        // TODO map is too large, remove some one
        map_point_erase_ratio_ += 0.05;
    }
    else
        map_point_erase_ratio_ = 0.1;
    cout<<"map points: "<<map_->map_points_.size()<<endl;
}

double VisualOdometry::getViewAngle ( Frame::Ptr frame, MapPoint::Ptr point )
{
    Vector3d n = point->pos_ - frame->getCamCenter();
    n.normalize();
    return acos( n.transpose()*point->norm_ );
}

bool VisualOdometry::FindMotionFromEssential(const Mat& _E, const Mat& _K,
                                 vector<cv::Point2d> pts1, vector<cv::Point2d> pts2,
                                 const Mat& inlierMask, SE3 &pose,
                                 vector<Vector3d> &vP3D,vector<bool> vbTriangulated,
                                 float minParallax, int minTriangulated)
{
    int N=0;
    for(size_t i=0,
        iend = inlierMask.cols>inlierMask.rows?inlierMask.cols:inlierMask.rows;
        i<iend;i++)
        N++;

    //Cast cv Mat to eigen Matrix
    Eigen::MatrixXd E(3,3);
    Eigen::MatrixXd K(3,3);
    Eigen::MatrixXd Rz1(3,3);
    Eigen::MatrixXd Rz2(3,3);
    cv::cv2eigen(_E,E);
    cv::cv2eigen(_K,K);
    Rz1 <<  0, -1,  0,
            1,  0,  0,
            0,  0,  1;
    Rz2 = Rz1.transpose();


    // Decompose E
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(E, ComputeThinU|ComputeThinV);
    Eigen::MatrixXd U =svd.matrixU();
    Eigen::MatrixXd V =svd.matrixV();
    Eigen::MatrixXd S = svd.singularValues().asDiagonal();

    // Compute Candidates
    Vector3d t1(U(0,2),U(1,2),U(2,2));
    t1 = t1/t1.norm();
    Vector3d t2 = -t1;

    Eigen::Matrix3d R1 = U*Rz1*V.transpose();
    if(R1.determinant()<0) R1=-R1;
    Eigen::Matrix3d R2 = U*Rz2*V.transpose();
    if(R2.determinant()<0) R2=-R2;


    vector<Eigen::Vector3d> vP3D1, vP3D2, vP3D3, vP3D4;
    vector<bool> vbGood1, vbGood2, vbGood3, vbGood4;
    float parallax1, parallax2, parallax3, parallax4;

    int nGood1 = countGoodDecompose(R1,t1,pts1,pts2,inlierMask,vP3D1, 4.0, K, vbGood1, parallax1);
    int nGood2 = countGoodDecompose(R1,t2,pts1,pts2,inlierMask,vP3D2, 4.0, K, vbGood2, parallax2);
    int nGood3 = countGoodDecompose(R2,t1,pts1,pts2,inlierMask,vP3D3, 4.0, K, vbGood3, parallax3);
    int nGood4 = countGoodDecompose(R2,t2,pts1,pts2,inlierMask,vP3D4, 4.0, K, vbGood4, parallax4);

    int maxGood = Max(nGood1, nGood2, nGood3, nGood4);

    int nMinGood = max(static_cast<int>(0.9*N), minTriangulated);

    int nsimilar = 0;
    if(nGood1 > .7 * maxGood) nsimilar++;
    if(nGood2 > .7 * maxGood) nsimilar++;
    if(nGood3 > .7 * maxGood) nsimilar++;
    if(nGood4 > .7 * maxGood) nsimilar++;

    if( maxGood>nMinGood || nsimilar> 1) return false;

    if(maxGood == nGood1 && parallax1 > minParallax)
    {
        vP3D = vP3D1;
        vbTriangulated = std::move(vbGood1);
        pose = SE3(R1,t1);
        pose.setRotationMatrix(R1);
        pose.translation() = t1;
    }
    else if(maxGood == nGood2 && parallax2 > minParallax)
    {
        vP3D = vP3D2;
        vbTriangulated = std::move(vbGood2);
        pose = SE3(R1,t2);
        pose.setRotationMatrix(R1);
        pose.translation() = t2;
    }
    else if(maxGood == nGood3 && parallax3 > minParallax)
    {
        vP3D = vP3D1;
        vbTriangulated = std::move(vbGood3);
        pose = SE3(R2,t1);
        pose.setRotationMatrix(R2);
        pose.translation() = t1;
    }
    else if(maxGood == nGood4 && parallax4 > minParallax)
    {
        vP3D = vP3D1;
        vbTriangulated = std::move(vbGood4);
        pose = SE3(R2,t2);
        pose.setRotationMatrix(R2);
        pose.translation() = t2;
    }
    cout<<"rotation in FindMotion = "<<pose.rotation_matrix()<<endl;
    cout<<"translation in FindMotion = "<<pose.translation()<<endl;

    return true;
}
    Vector3d VisualOdometry::Down(Eigen::Matrix3d& A)
    {
        return Vector3d(A(2,1), A(0,2),A(1,0));

    }
    Eigen::Matrix3d& VisualOdometry::Up(const Vector3d& a)
    {
        Eigen::Matrix3d A;
        A << 0, -a(2), a(1), a(2), 0, -a(0), -a(1), a(0), 0;
        return A;

    }
    Vector3f VisualOdometry::Down(Eigen::Matrix3f& A)
    {
        return Vector3f(A(2,1), A(0,2),A(1,0));

    }
    Eigen::Matrix3f& VisualOdometry::Up(const Vector3f& a)
    {
        Eigen::Matrix3f A;
        A << 0, -a(2), a(1), a(2), 0, -a(0), -a(1), a(0), 0;
        return A;

    }
    cv::Point3f& VisualOdometry::Homogenizing(const cv::Point2d& pt)
    {
        cv::Point3f tmp(pt.x,pt.y,1);
        return tmp;
    }
    int VisualOdometry::countGoodDecompose(const Eigen::Matrix3d& R, const Eigen::Vector3d t,
                           const vector<cv::Point2d> pts1,const vector<cv::Point2d> pts2,
                           const Mat& inliers, vector<Eigen::Vector3d> &vP3D,
                           float th2, const Eigen::Matrix3d& K,
                           vector<bool>& vbGood, float& parallax)
    {
        const float fx = K(0,0);
        const float fy = K(1,1);
        const float cx = K(0,2);
        const float cy = K(1,2);

        vbGood = vector<bool>(pts1.size(), false);
        vP3D.resize(pts1.size());

        vector<float> vCosParallax;
        vCosParallax.reserve(pts1.size());

        //Camera1 Projection Matrix K[I|0]
        Eigen::Matrix<double, 3, 4> P1 = Matrix<double, 3,4>::Zero();
        P1.block(0,0,3,3) = K;

        Eigen::Vector3d O1(0,0,0);

        Eigen::Matrix<double, 3, 4> P2= Matrix<double, 3,4>::Zero();
        P2.block(0,0,3,3) = R;
        P2.block(0,3,3,4) = t;
        P2 = K*P2;

        Eigen::Vector3d O2 = -R.transpose()*t;

        int nGood = 0;

        for(size_t i=0, iend=(inliers.cols>inliers.rows?inliers.cols:inliers.rows);i<iend; i++)
        {
            if(!inliers.at<bool>(i,0)) continue; // Pass outlier;

            const cv::Point2d p1 = pts1[i];
            const cv::Point2d p2 = pts2[i];
            Eigen::Vector3d p3dC1;
            Triangulate(p1,p2, P1,P2, p3dC1); // DLT method output is 3D point in Real space 3D

            if(!std::isfinite(p3dC1(0)) || !isfinite(p3dC1(1))|| !isfinite(p3dC1(2)))
            {
                vbGood[i] = false;
                continue;
            }

            // Check parallax
            Vector3d normal1 = p3dC1 -O1;
            float dist1 = normal1.norm();
            Vector3d normal2 = p3dC1 -O2;
            float dist2 = normal2.norm();

            float cosParallax = normal1.dot(normal2)/(dist1*dist2);

            // Check depth in front of first camera (only if enough paralla, as "infinite" points
            // can easily go to negative depth)
            if(p3dC1(2) <=0 && cosParallax<0.99998) continue;

            // Check depth in front of second camera (only if enough paralla, as "infinite" points
            // can easily go to negative depth)
            Vector3d p3dC2 = R*p3dC1 + t;
            if(p3dC2(2) <=0 && cosParallax<0.99998) continue;

            // Check reprojection Error in First image
            float im1x, im1y;
            float invZ1 = 1.0f/p3dC1(2);
            im1x = fx*p3dC1(0)*invZ1 +cx;
            im1y = fy*p3dC1(0)*invZ1 +cy;

            float squareError1 = (im1x-p1.x)*(im1x-p1.x) + (im1y-p1.y)*(im1y-p1.y);

            if(squareError1 > th2) continue;

            // Check reprojection Error in Second image
            float im2x, im2y;
            float invZ2 = 1.0f/p3dC2(2);
            im1x = fx*p3dC2(0)*invZ2 +cx;
            im1y = fy*p3dC2(0)*invZ2 +cy;

            float squareError2 = (im2x-p2.x)*(im2x-p2.x) + (im2y-p2.y)*(im2y-p2.y);

            if(squareError2 > th2) continue;

            // Summary
            vCosParallax.push_back(cosParallax);
            vP3D[i] = Vector3d(p3dC1(0), p3dC1(1), p3dC1(2));
            nGood++;

            if(cosParallax<0.99998)
            vbGood[i]=true;
        }

        if(nGood >0)
        {
            sort(vCosParallax.begin(), vCosParallax.end());

            size_t idx = min(50, int(vCosParallax.size()-1));
            parallax = acos(vCosParallax[idx]*180/CV_PI);
        }
        else
        {
            parallax = 0;
        }

        return nGood;
    }
    void VisualOdometry::Triangulate(const cv::Point2d p1,const cv::Point2d p2,
                                     Projection P1, Projection P2,
                                     Eigen::Vector3d& p3dC1)
    {
        Eigen::Matrix4d T;
        T.row(0) = p1.x *P1.row(2) - P1.row(0);
        T.row(1) = p1.y *P1.row(2) - P1.row(1);
        T.row(2) = p2.x *P2.row(2) - P2.row(0);
        T.row(3) = p2.y *P2.row(2) - P2.row(1);

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(T, ComputeFullU|ComputeFullV);
        Eigen::MatrixXd V =svd.matrixV();
        Vector4d temp =V.transpose().row(3).transpose();
        p3dC1 = Vector3d(temp(0)/temp(3),temp(1)/temp(3), temp(2)/temp(3));
    }
}
