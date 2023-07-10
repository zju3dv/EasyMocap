/*** 
 * @Date: 2020-09-19 16:10:21
 * @Author: Qing Shuai
 * @LastEditors: Qing Shuai
 * @LastEditTime: 2020-09-24 21:09:58
 * @FilePath: /MatchLR/include/match/base.h
 */
#pragma once
#include <vector>
#include "Eigen/Dense"
#include <unordered_map>
#include <string>

namespace match
{
    typedef float Type;
    typedef Eigen::Matrix<Type, -1, -1> Mat;
    typedef Eigen::Array<Type, -1, -1> Array;
    template <typename T>
    using Vec=std::vector<T>;
    typedef std::vector<int> List;
    typedef std::vector<List> ListList;
    struct MatchInfo
    {
        int maxIter = 100;
        float alpha = 200;
        float beta = 0.1;
        float tol = 1e-3;
        float w_sparse = 0.1;
        float w_rank = 50;
    };
    typedef std::unordered_map<std::string, float> Control;

    void print(Vec<int>& lists, std::string name){
        std::cout << name << ": [";
        for(auto i:lists){
            std::cout << i << ", ";
        }
        std::cout << "]" << std::endl;
    }
} // namespace match
