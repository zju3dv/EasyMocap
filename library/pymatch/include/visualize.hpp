/*** 
 * @Date: 2020-09-12 19:37:01
 * @Author: Qing Shuai
 * @LastEditors: Qing Shuai
 * @LastEditTime: 2020-09-12 19:37:46
 * @FilePath: /MatchLR/include/visualize.hpp
 */
#pragma once
#ifdef _USE_OPENCV_
#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"
#endif

namespace match
{


#ifdef _USE_OPENCV_
cv::Mat eigen2mat(Mat Z){
    cv::Mat showi, showd, showrgb;
    auto Zmin = Z.minCoeff();
    auto Zmax = Z.maxCoeff();
    Z = ((Z.array() - Zmin)/(Zmax - Zmin)).matrix();
    cv::eigen2cv(Z, showd);
    showd.convertTo(showi, CV_8UC1, 255);
    cv::applyColorMap(showi, showrgb, cv::COLORMAP_JET);
    while(showrgb.rows < 600){
        cv::resize(showrgb, showrgb, cv::Size(), 2, 2, cv::INTER_NEAREST);
    }
    return showrgb;
}
#endif

} // namespace match
