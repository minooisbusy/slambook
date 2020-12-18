#ifndef KEYFRAME_H
#define KEYFRMAE_H
#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"
#include "myslam/mappoint.h"
namespace myslam
{
class Map;
Class MapPoint;
class KeyFrame
{
  KeyFrame(Frame &F, Map* pMap);

}
}
#endif