/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, * but WITHOUT ANY WARRANTY; without even the implied warranty of
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
    surf_ = cv::xfeatures2d::SurfFeatureDetector::create();
    surf_desc_ = cv::xfeatures2d::SurfDescriptorExtractor::create();
    srand(0);
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
                ref_ = curr_;
                keypoints_prev_.clear();
                for(int i=0; i< keypoints_curr_.size();i++)
                    keypoints_prev_.push_back(keypoints_curr_[i]);
                descriptors_prev_ = descriptors_curr_.clone();
                return false;
            }
            else if(curr_ != nullptr && ref_ != nullptr)
            {
                vector<cv::DMatch> good_matches;
                cv::Ptr<cv::DescriptorMatcher> matcher =
                 cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
                std::vector<std::vector<cv::DMatch>> knn_matches;
                matcher->knnMatch(descriptors_prev_,descriptors_curr_, knn_matches, 2);

                const float ratio_thresh = .7f;
                for( size_t i =0; i<knn_matches.size(); i++)
                {
                    if(knn_matches[i][0].distance<ratio_thresh * knn_matches[i][1].distance)
                    {
                        good_matches.push_back(knn_matches[i][0]);
                    }
                }

                /*
                Mat img_matches;
                cv::drawMatches(ref_->color_, keypoints_prev_,
                                curr_->color_, keypoints_curr_,
                                good_matches, img_matches,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                cv::imshow("Draw Matches", img_matches);
                */

                Mat src = ref_->color_.clone();
                size_t sz = std::min(keypoints_curr_.size(), keypoints_prev_.size());
                sz = good_matches.size();
                cv::RNG rng(0);
                for(size_t t = 0; t < sz; t++)
                {
                    cv::arrowedLine(src, keypoints_prev_[good_matches[t].queryIdx].pt,
                                  keypoints_curr_[good_matches[t].trainIdx].pt,
                                  cv::Scalar(rng(256),rng(256),rng(256)), 2);
                }
                cv::imshow("features", src);

                vector<cv::Point2f> pts1;
                vector<cv::Point2f> pts2;
                pts1.reserve(good_matches.size());
                pts2.reserve(good_matches.size());
                for(size_t i=0;i < good_matches.size(); i++)
                {
                    const float x1 = keypoints_prev_[good_matches[i].queryIdx].pt.x;
                    const float y1 = keypoints_prev_[good_matches[i].queryIdx].pt.y;
                    const float x2 = keypoints_curr_[good_matches[i].trainIdx].pt.x;
                    const float y2 = keypoints_curr_[good_matches[i].trainIdx].pt.y;

                    pts1.emplace_back(x1,y1);
                    pts2.emplace_back(x2,y2);
                }

                double lambda;
                Mat inlierMask;
                //Mat E = cv::findEssentialMat(pts1,pts2,curr_->camera_->mK,cv::RANSAC,0.999,1.0,inlierMask);
                vector<bool> vbMatchesInliersF;
                float score=0;
                Mat F;
                //FindFundamental(vbMatchesInliersF,score,F, pts1,pts2);
                F = cv::findFundamentalMat(pts1,pts2,inlierMask,cv::RANSAC);
                F.convertTo(F,CV_32F);

                //Mat lines;
                drawepipolarlines("Epipolar lines", F, ref_->color_,curr_->color_,
                                  pts1,pts2);

                //vector<Eigen::Vector3d> vP3D;
                vector<cv::Point3f> vP3D;
                vector<bool> vbTriangulated;
                if(FindMotion(F, curr_->camera_->mK,
                                           pts1, pts2,
                                           inlierMask, curr_->T_c_w_, // Output pose
                                           vP3D, vbTriangulated, // Output Point, boolean
                                           1.0, 50))
                {
                    std::cout<<"Pose = "<<curr_->T_c_w_.rotation_matrix()<<"\n"<<curr_->T_c_w_.translation()<<std::endl;
                }
                else
                {
                    return false;
                }


                ref_ = curr_;
                keypoints_prev_.clear();
                for(int i=0; i< keypoints_curr_.size();i++)
                    keypoints_prev_.push_back(keypoints_curr_[i]);
                descriptors_prev_ = descriptors_curr_.clone();

            }


        }
        addKeyFrame();      // the first frame is a key-frame

        ref_ = curr_;
        keypoints_prev_.clear();
        for(int i=0; i< keypoints_curr_.size();i++)
            keypoints_prev_.push_back(keypoints_curr_[i]);
        descriptors_prev_ = descriptors_curr_.clone();
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
    //surf_->detect(curr_->color_, keypoints_curr_);
    cout<<"extract keypoints cost time: "<<timer.elapsed() <<endl;
}

void VisualOdometry::computeDescriptors()
{
    boost::timer timer;
    orb_->compute ( curr_->color_, keypoints_curr_, descriptors_curr_);
    //surf_desc_->compute(curr_->color_, keypoints_curr_,descriptors_curr_);
    descriptors_curr_.convertTo(descriptors_curr_, CV_32F);
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

//ReconstructF
bool VisualOdometry::FindMotion(const Mat& F, const Mat& K,
                                 const vector<cv::Point2f>& pts1,const vector<cv::Point2f>& pts2,
                                 const Mat& inlierMask, SE3 &pose,
                                 //vector<Vector3d> &vP3D,
                                 vector<cv::Point3f> &vP3D,
                                 vector<bool> vbTriangulated,
                                 float minParallax, int minTriangulated)
{
    int N=0;
    for(size_t i=0,
        iend = inlierMask.cols>inlierMask.rows?inlierMask.cols:inlierMask.rows;
        i<iend;i++)
        {
            if(inlierMask.at<bool>(i,0))
                N++;
        }

    //Cast cv-Mat to eigen-Matrix
    cv::Mat E = K.t()*F*K;
    /*
    cv::Mat Rz1 = cv::Mat::eye(3,3,CV_32F);
    Rz1.at<float>(0,1) = -1;
    Rz1.at<float>(1,0) = 1;
    Rz1.at<float>(2,2) = 1;
    */

    Mat R1, R2, t1, t2;
    cv::decomposeEssentialMat(E, R1, R2, t1);
    t2=-t1;




    //vector<Eigen::Vector3d> vP3D1, vP3D2, vP3D3, vP3D4;
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    vector<bool> vbGood1, vbGood2, vbGood3, vbGood4;
    float parallax1, parallax2, parallax3, parallax4;

    float sigma2=2.0*2.0;
    int nGood1 = countGoodDecompose(R1,t1,pts1,pts2,inlierMask,vP3D1, 4.0*sigma2, K, vbGood1, parallax1);
    int nGood2 = countGoodDecompose(R1,t2,pts1,pts2,inlierMask,vP3D2, 4.0*sigma2, K, vbGood2, parallax2);
    int nGood3 = countGoodDecompose(R2,t1,pts1,pts2,inlierMask,vP3D3, 4.0*sigma2, K, vbGood3, parallax3);
    int nGood4 = countGoodDecompose(R2,t2,pts1,pts2,inlierMask,vP3D4, 4.0*sigma2, K, vbGood4, parallax4);

    cout<< "ngood1= " << nGood1
        << "\nngood2= " << nGood2
        << "\nngood3= " << nGood3
        << "\nngood4= " << nGood4<<endl;



    int maxGood = max(nGood1, max(nGood2, max(nGood3, nGood4)));
    if(maxGood >0)
    {
        cout<<"maxGood = "<<maxGood<<endl;
    }
        cout<<"maxGood = "<<maxGood<<endl;

    int nMinGood = max(static_cast<int>(0.9*N), minTriangulated);

    int nsimilar = 0;
    if(nGood1 > .7 * maxGood) nsimilar++;
    if(nGood2 > .7 * maxGood) nsimilar++;
    if(nGood3 > .7 * maxGood) nsimilar++;
    if(nGood4 > .7 * maxGood) nsimilar++;

    // If there is not a clear winner (nsimilar > 1)
    // or not enough triangulated points (maxGood>nMinGood)
    // reject initialization
    if( maxGood>nMinGood || nsimilar> 1)
    {
        return false;
    }
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    if(maxGood == nGood1 && parallax1 > minParallax)
    {
        vP3D = vP3D1;
        vbTriangulated = std::move(vbGood1);
        cv::cv2eigen(R1, R);
        cv::cv2eigen(t1, t);
    }
    else if(maxGood == nGood2 && parallax2 > minParallax)
    {
        vP3D = vP3D2;
        vbTriangulated = std::move(vbGood2);
        cv::cv2eigen(R1, R);
        cv::cv2eigen(t2, t);
    }
    else if(maxGood == nGood3 && parallax3 > minParallax)
    {
        vP3D = vP3D1;
        vbTriangulated = std::move(vbGood3);
        cv::cv2eigen(R2, R);
        cv::cv2eigen(t1, t);
    }
    else if(maxGood == nGood4 && parallax4 > minParallax)
    {
        vP3D = vP3D1;
        vbTriangulated = std::move(vbGood4);
        cv::cv2eigen(R2, R);
        cv::cv2eigen(t2, t);
    }
    else
    {
        return false;
    }
    pose = SE3(R,t);

    return true;
}
int VisualOdometry::countGoodDecompose(const cv::Mat& R,
                        const cv::Mat& t,
                        const vector<cv::Point2f>& pts1,
                        const vector<cv::Point2f>& pts2,
                        const Mat& inliers,
                        //vector<Eigen::Vector3d> &vP3D,
                        vector<cv::Point3f> &vP3D,
                        const float& th2,
                        const cv::Mat& K,
                        vector<bool>& vbGood, float& parallax)
{
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);
    vector<float> vCosParallax;
    vCosParallax.reserve(pts1.size()); //Camera1 Projection Matrix K[I|0]
    vP3D.resize(pts1.size());
    vbGood = vector<bool>(pts1.size(), false);
    Mat P1 = cv::Mat::zeros(3,4,CV_32F);
    Mat P2 = cv::Mat::zeros(3,4,CV_32F);

    K.copyTo(P1.rowRange(0,3).colRange(0,3));
    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    K.convertTo(K, CV_32F);
    P2 = K*P2;
    cv::Mat O2 = -R.t()*t;
    O2.convertTo(O2, CV_32F);


    int nGood = 0;

    for(size_t i=0, iend=(inliers.cols>inliers.rows?inliers.cols:inliers.rows);i<iend; i++)
    {
        if(!inliers.at<bool>(i,0)) continue; // Pass outlier;

        const cv::Point2f p1 = pts1[i];
        const cv::Point2f p2 = pts2[i];
        Eigen::Vector3d p3dC1;
        cv::Mat cvp3dC1;

        //Triangulate(p1,p2, P1,P2, p3dC1); // DLT method output is 3D point in Real space 3D
        Triangulate(p1,p2, P1,P2,cvp3dC1 );

        if(!isfinite(cvp3dC1.at<float>(0)) || !isfinite(cvp3dC1.at<float>(1)) || !isfinite(cvp3dC1.at<float>(2)))
        {
            vbGood[i]=false;
            continue;
        }

        // Check parallax
        cv::Mat normal1 = cvp3dC1 - O1;
        float dist1 = cv::norm(normal1);

        cv::Mat normal2 = cvp3dC1 - O2;
        float dist2 = cv::norm(normal2);

        float cosParallax = normal1.dot(normal2)/(dist1*dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if(cvp3dC1.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R*cvp3dC1+t;

        if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check reprojection error in first image
        float im1x, im1y;
        float invZ1 = 1.0/cvp3dC1.at<float>(2);
        im1x = fx*cvp3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*cvp3dC1.at<float>(1)*invZ1+cy;

        float squareError1 = (im1x-p1.x)*(im1x-p1.x)+(im1y-p1.y)*(im1y-p1.y);

        if(squareError1>th2)
            continue;

        // Check reprojection error in second image
        float im2x, im2y;
        float invZ2 = 1.0/p3dC2.at<float>(2);
        im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
        im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

        float squareError2 = (im2x-p2.x)*(im2x-p2.x)+(im2y-p2.y)*(im2y-p2.y);

        if(squareError2>th2)
            continue;

        vCosParallax.push_back(cosParallax);
        vP3D[i] = cv::Point3f(cvp3dC1.at<float>(0),cvp3dC1.at<float>(1),cvp3dC1.at<float>(2));
        nGood++;

        if(cosParallax<0.99998)
            vbGood[i]=true;
    }

    if(nGood>0)
    {
        sort(vCosParallax.begin(),vCosParallax.end());

        size_t idx = min(50,int(vCosParallax.size()-1));
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;
    return nGood;
}
    void VisualOdometry::Triangulate(const cv::Point2f p1,const cv::Point2f p2,
                                     Projection P1, Projection P2,
                                     Eigen::Vector3d& p3dC1)
    {
        Eigen::Matrix4d T;
        T.row(0) = p1.x * P1.row(2) - P1.row(0);
        T.row(1) = p1.y * P1.row(2) - P1.row(1);
        T.row(2) = p2.x * P2.row(2) - P2.row(0);
        T.row(3) = p2.y * P2.row(2) - P2.row(1);


        Eigen::JacobiSVD<Eigen::MatrixXd> svd(T, ComputeFullU|ComputeFullV);
        const MatrixXd V =svd.matrixV();
        const Vector4d temp (V(3,0),V(3,1),V(3,2),V(3,3));
        p3dC1 = Vector3d(temp(0)/temp(3),temp(1)/temp(3), temp(2)/temp(3));
    }

void VisualOdometry::FindFundamental(std::vector<bool> &vbMatchesInliers, float &score,
                        cv::Mat &F21, std::vector<cv::Point2f> &src, std::vector<cv::Point2f> &dst)
{
        const int N = src.size();
        vector<vector<size_t>> mvSets;
        const int maxIteration = 500;
        vector<size_t> vAllindices;
        vAllindices.reserve(N);
        vector<size_t> vAvailableIndices;

        score = 0;
        for(size_t i=0; i<N; i++)
        {
            vAllindices.push_back(i);
        }
        mvSets = vector<vector<size_t>> (maxIteration, vector<size_t>(8,0));
        const int min =0;
        const int max  = vAvailableIndices.size()-1;
        const int d = max - min + 1;
        for(size_t i=0; i<maxIteration; i++)
        {
            vAvailableIndices = vAllindices;

            for(size_t j=0; j<8; j++)
            {
                int randi = int(((double)rand()/((double)RAND_MAX + 1.0)) * d)+min;
                int idx = vAvailableIndices[randi];

                mvSets[i][j] = idx;

                vAvailableIndices[randi] = vAvailableIndices.back();
                vAvailableIndices.pop_back();
            }
        }

        // Launch threads to compute in parallal a fundamental matrix and a homography
        //const int N = vbMatchesInliers.size();

        // Normalize Coordinates
        vector<cv::Point2f> vPn1, vPn2;
        cv::Mat T1, T2;
        Normalize(src, vPn1, T1);
        Normalize(dst, vPn2, T2);
        cv::Mat T2t = T2.t();

        // Best Results variables
        score = 0.0f;
        vbMatchesInliers = vector<bool>(N, false); //


        // Iteration variables
        vector<cv::Point2f> vPn1i(8);
        vector<cv::Point2f> vPn2i(8);
        cv::Mat F21i;
        vector<bool> vbCurrentInliers(N, false);
        float currentScore;

        // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<maxIteration; it++)
    {
        // Select a minimum set
        for(int j=0; j<8; j++)
        {
            int idx = mvSets[it][j];

            vPn1i[j] = vPn1[j]; // i don't input match info just point pair set
            vPn2i[j] = vPn2[j];
        }

        cv::Mat Fn = ComputeF21(vPn1i,vPn2i);
        F21i = T2t*Fn*T1;

        currentScore = CheckFundamental(F21i, vbCurrentInliers,src, dst, 1.0);//msigma = 1.0

        if(currentScore>score)
        {
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }


}
void VisualOdometry::Normalize(const vector<cv::Point2f>& pts, vector<cv::Point2f> &res, Mat &T)
{
        float meanX = 0;
        float meanY = 0;
        const int N = pts.size();

        res.resize(N);

        for(size_t i = 0; i < N; i++)
        {
            meanX += pts[i].x;
            meanX += pts[i].y;
        }
        meanX /= N;
        meanY /= N;

        float meanDevX = 0;
        float meanDevY = 0;

        for(size_t i = 0; i< N; i++)
        {
            res[i].x = pts[i].x - meanX;
            res[i].y = pts[i].y - meanY;

            meanDevX += fabs(res[i].x);
            meanDevY += fabs(res[i].y);
        }
        meanDevX /= N;
        meanDevY /= N;


        float sX = 1.0/meanDevX;
        float sY = 1.0/meanDevY;

        for(size_t i = 0; i < N; i++)
        {
            res[i].x *= sX;
            res[i].y *= sY;
        }

        T = Mat::eye(3,3,CV_32F);
        T.at<float>(0,0) = sX;
        T.at<float>(1,1) = sY;
        T.at<float>(0,2) = -meanX*sX;
        T.at<float>(1,2) = -meanY*sY;
}

cv::Mat VisualOdometry::ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(N, 9, CV_32F);

    for(size_t i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<float>(2)=0;

    return  u*cv::Mat::diag(w)*vt;
}
float VisualOdometry::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers,
                                       const vector<cv::Point2f> &mvKeys1, const vector<cv::Point2f> &mvKeys2,
                                        float sigma)
{
    const int N = mvKeys1.size();

    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 3.841;
    const float thScore = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);

    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::Point2f &kp1 = mvKeys1[i];
        const cv::Point2f &kp2 = mvKeys2[i];

        const float u1 = kp1.x;
        const float v1 = kp1.y;
        const float u2 = kp2.x;
        const float v2 = kp2.y;

        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)

        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

        const float num2 = a2*u2+b2*v2+c2;

        const float squareDist1 = num2*num2/(a2*a2+b2*b2);

        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)

        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        const float num1 = a1*u1+b1*v1+c1;

        const float squareDist2 = num1*num1/(a1*a1+b1*b1);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}
Vector3d VisualOdometry::Down(Eigen::Matrix3d M)
{
    return Vector3d(M(2,1),M(0,2),M(2,0));
}
void VisualOdometry::drawepipolarlines(const std::string& title, const cv::Mat &F,
                const cv::Mat& img1, const cv::Mat& img2,
                const std::vector<cv::Point2f> points1,
                const std::vector<cv::Point2f> points2,
                const float inlierDistance)
{
  CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());
  cv::Mat outImg(img1.rows, img1.cols*2, CV_8UC3);
  cv::Rect rect1(0,0, img1.cols, img1.rows);
  cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);
  /*
   * Allow color drawing
   */
  if (img1.type() == CV_8U)
  {
    cv::cvtColor(img1, outImg(rect1), cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, outImg(rect2), cv::COLOR_GRAY2BGR);
  }
  else
  {
    img1.copyTo(outImg(rect1));
    img2.copyTo(outImg(rect2));
  }
  std::vector<cv::Point3f> epilines1, epilines2;
  cv::computeCorrespondEpilines(points1, 1, F, epilines1); //Index starts with 1
  cv::computeCorrespondEpilines(points2, 2, F, epilines2);

  CV_Assert(points1.size() == points2.size() &&
        points2.size() == epilines1.size() &&
        epilines1.size() == epilines2.size());

  cv::RNG rng(0);
  for(size_t i=0; i<points1.size(); i++)
  {
    if(inlierDistance > 0)
    {
      if(distancePointLine(points1[i], epilines2[i]) > inlierDistance||
        distancePointLine(points2[i], epilines1[i]) > inlierDistance)
      {
        //The point match is no inlier
        continue;
      }
    }
    /*
     * Epipolar lines of the 1st point set are drawn in the 2nd image and vice-versa
     */
    cv::Scalar color(rng(256),rng(256),rng(256));

    cv::line(outImg(rect2),
      cv::Point(0,-epilines1[i].z/epilines1[i].y),
      cv::Point(img1.cols,-(epilines1[i].z+epilines1[i].x*img1.cols)/epilines1[i].y),
      color);
    cv::circle(outImg(rect1), points1[i], 3, color, -1);

    cv::line(outImg(rect1),
      cv::Point(0,-epilines2[i].z/epilines2[i].y),
      cv::Point(img2.cols,-(epilines2[i].z+epilines2[i].x*img2.cols)/epilines2[i].y),
      color);
    cv::circle(outImg(rect2), points2[i], 3, color, -1);
  }
  cv::imshow(title, outImg);
}

float VisualOdometry::distancePointLine(const cv::Point2f& point, const cv::Point3f& line)
{
  //Line is given as a*x + b*y + c = 0
  return std::fabs(line.x*point.x + line.y*point.y + line.z)
      / std::sqrt(line.x*line.x+line.y*line.y);
}
void VisualOdometry::Triangulate(const cv::Point2f &p1, const cv::Point2f &p2, const cv::Mat &P1, const cv::Mat &P2, Mat &x3D)
{
    cv::Mat A(4,4,CV_32F);
    A.row(0) = p1.x*P1.row(2)-P1.row(0);
    A.row(1) = p1.y*P1.row(2)-P1.row(1);
    A.row(2) = p2.x*P2.row(2)-P2.row(0);
    A.row(3) = p2.y*P2.row(2)-P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
    x3D.convertTo(x3D, CV_32F);
}

}
