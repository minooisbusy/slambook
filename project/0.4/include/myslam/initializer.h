#ifndef INITIALIZER_H
#define INITIALIZER_H
#include "myslam/common_include.h"
namespace myslam
{
class Initializer
{
  typedef std::pair<int, int> Match;
  std::shared_ptr<Initializer> Ptr;
  std::vector<cv::KeyPoint> mvPrevKeys;
  std::vector<cv::KeyPoint> mvCurrKeys;
  std::vector<Match> mvMatches12;
  std::vector<bool> mvbMatched1; // 굳이 필요한 이유가 있는가?
  cv::Mat mK;
  float mSigma, mSigma2;

  // Ransac max iterations
  int mMaxIterations;

  // Ransac sets
  vector<vector<size_t> > mvSets;

  public:
  bool Initialize();
  void ComputeEssential();


};
}

#endif
