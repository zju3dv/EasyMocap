/*** 
 * @Date: 2020-09-12 19:01:56
 * @Author: Qing Shuai
 * @LastEditors: Qing Shuai
 * @LastEditTime: 2022-07-29 22:38:40
 * @FilePath: /EasyMocapPublic/library/pymatch/include/matchSVT.hpp
 */
#pragma once
#include "base.h"
#include "projfunc.hpp"
#include "Timer.hpp"
#include "visualize.hpp"

namespace match
{
    struct Config
    {
        bool debug;
        int max_iter;
        float tol;
        float w_rank;
        float w_sparse;
        Config(Control& control){
            debug = (control["debug"] > 0.);
            max_iter = int(control["maxIter"]);
            tol = control["tol"];
            w_rank = control["w_rank"];
            w_sparse = control["w_sparse"];
        }
    };

    // dimGroups: [0, nF1, nF1 + nF2, ...]
    ListList getBlocksFromDimGroups(const List& dimGroups){
        ListList block;
        for(int i=0;i<dimGroups.size() - 1;i++){
            // 这个视角没有找到人的情况
            if(dimGroups[i] == dimGroups[i+1])continue;
            for(int j=0;j<dimGroups.size()-1;j++){
                if(i==j)continue;
                if(dimGroups[j] == dimGroups[j+1])continue;
                block.push_back({dimGroups[i], dimGroups[i+1] - dimGroups[i], 
                    dimGroups[j], dimGroups[j+1] - dimGroups[j]});
            }
        }
        return block;
    }
    // matchSVT with constraint and observation
    // M_aff: (N, N): affinity matrix
    // M_constr: =0, when (i, j) cannot be the same person
    //      if not consider this, set to 1(N, N)
    // M_obs: =0, when (i, j) cannot be observed
    //      if not consider this, set to 1(N, N)
    Mat matchSVT(Mat M_aff, List dimGroups, Mat M_constr, Mat M_obs, Control control)
    {
        bool debug = (control["debug"] > 0.);
        int max_iter = int(control["maxIter"]);
        float tol = control["tol"];

        int N = M_aff.rows();
        auto dual_blocks = getBlocksFromDimGroups(dimGroups);
        // 对角线约束
        for(int i=0;i<dimGroups.size() - 1;i++){
            M_constr.block(dimGroups[i], dimGroups[i], dimGroups[i+1] - dimGroups[i], dimGroups[i+1]-dimGroups[i]).setZero();
        }
        M_constr.diagonal().setOnes();
        // 将affinity乘一下constraint，保证满足约束
        M_aff = (M_aff.array() * M_constr.array()).matrix();
        // check一下所有区块，如果最大值和最小值差异过小的，直接认为是错误观测
        for (auto block : dual_blocks)
        {
            Mat mat = M_aff.block(block[0], block[2], block[1], block[3]);
            if(debug){
                std::cout << "(" << block[0] << ", " << block[2] << "), ";
                std::cout << "min: " << mat.minCoeff() << ", max: " << mat.maxCoeff() << std::endl;
            }
            if(mat.minCoeff() > 0.7 && block[1] > 1 && block[3] > 1){
                // 如果大于0.9,说明区分度不够高啊，认为观测是虚假的
                M_obs.block(block[0], block[2], block[1], block[3]).setZero();
            }
        }
        // set the diag of M_aff to zeros
        M_aff.diagonal().setConstant(0);
        Mat X = M_aff;
        Mat Y = Mat::Zero(N, N);
        Mat Q = M_aff;
        Mat W = (control["w_sparse"] - M_aff.array()).matrix();
        float mu = 64;
        Timer timer;
        timer.tic();
        for (int iter = 0; iter < max_iter; iter++)
        {
            Mat X0 = X;
            // update Q with SVT
            Q = 1.0 / mu * Y + X;
            Eigen::BDCSVD<Mat> UDV(Q.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV));
            Array Ds(Dsoft(UDV.singularValues(), control["w_rank"] / mu));
            Mat Qnew(UDV.matrixU() * Ds.matrix().asDiagonal() * UDV.matrixV().adjoint());
            Q = Qnew;
            // update X
            X = Q - (M_obs.array() * W.array() + Y.array()).matrix() / mu;
            X = (X.array() * M_constr.array()).matrix();
            // set the diagonal
            X.diagonal().setOnes();
            // 注意这个min,max
            X = X.cwiseMin(1.f).cwiseMax(0.f);
            #pragma omp parallel for
            for(int i=0;i<dual_blocks.size();i++)
            {
                auto& block = dual_blocks[i];
                X.block(block[0], block[2], block[1], block[3]) = myproj2dpam(X.block(block[0], block[2], block[1], block[3]), 1e-2);
            }
            X = (X + X.transpose().eval()) / 2;
            Y = Y + mu * (X - Q);
            float pRes = (X - Q).norm() / N;
            float dRes = mu * (X - X0).norm() / N;
            if(debug){
#ifdef _USE_OPENCV_
                cv::imshow("Q", eigen2mat(Q));
                cv::imshow("X", eigen2mat(X));
                cv::waitKey(100);
#endif
                std::cout << "Iter " << iter << ", Res = (" << pRes << ", " << dRes << "), mu = " << mu << std::endl;
            }
            if (pRes < tol && dRes < tol)
            {
                std::cout << "End " << iter << ", Res = (" << pRes << ", " << dRes << "), mu = " << mu << std::endl;
                break;
            }

            if (pRes > 10 * dRes)
            {
                mu *= 2;
            }
            else if (dRes > 10 * pRes)
            {
                mu /= 2;
            }
        }
        if(debug){
#ifdef _USE_OPENCV_
            timer.toc("solving svt");
            cv::imshow("X", eigen2mat(X));
            cv::imshow("Q", eigen2mat(Q));
            cv::waitKey(0);
#endif
        }
        return X;
    }

    Mat matchALS(Mat M_aff, List dimGroups, Mat M_constr, Mat M_obs, Control control)
    {
        // This function is to solve
        // min <W, X> + alpha||x||_* + beta||x||_1, st. X \in C
        // <beta - W, AB^T> + alpha/2||A||^2 + alpha/2||B||^2
        // st AB^T = Z, Z\in \Omega
        const Config cfg(control);
        Mat W = (M_aff + M_aff.transpose())/2;
        // set the diag of W to zeros
        for(int i=0;i<W.rows();i++){
            W(i, i) = 0;
        }
        Mat X = W;
        Mat Z = W;
        Mat Y = W;
        Y.setZero();
        int mu = 64;
        int n = X.rows();
        int maxRank = 0;
        for(size_t i=0;i<dimGroups.size() - 1;i++){
            if(dimGroups[i+1]-dimGroups[i] > maxRank){
                maxRank = dimGroups[i+1]-dimGroups[i];
            }
        }
        std::cout << "[matchALS] set the max rank = " << maxRank  << std::endl;
        Mat eyeRank = Mat::Identity(maxRank, maxRank);
        // initial value
        Mat A = Mat::Random(n, maxRank);
        Mat B;
        for(int iter=0;iter<cfg.max_iter;iter++){
            Mat X0 = X;
            X = Z - (((Y - W).array() + cfg.w_sparse)/mu).matrix();
            B = ((A.transpose() * A + cfg.w_rank/mu * eyeRank).ldlt().solve(A.transpose() * X)).transpose();
            A = ((B.transpose() * B + cfg.w_rank/mu * eyeRank).ldlt().solve(B.transpose() * X.transpose())).transpose();

            X = A * B.transpose();
            Z = X + Y/mu;
            for(int i=0;i<dimGroups.size() - 1;i++){
                int start = dimGroups[i];
                int end = dimGroups[i+1];
                Z.block(start, start, end-start, end-start).setIdentity();
            }
            // 注意这个min,max
            Z = Z.cwiseMin(1.f).cwiseMax(0.f);
            Y = Y + mu*(X - Z);
            
            float pRes = (X - Z).norm()/n;
            float dRes = mu*(X - X0).norm()/n;

            if(cfg.debug){
#ifdef _USE_OPENCV_
                cv::imshow("Z", eigen2mat(Z));
                cv::imshow("X", eigen2mat(X));
                cv::waitKey(10);
#endif
                std::cout << "Iter " << iter << ", Res = (" << pRes << ", " << dRes << "), mu = " << mu << std::endl;
            }

            if (pRes < cfg.tol && dRes < cfg.tol)
            {
                std::cout << "End " << iter << ", Res = (" << pRes << ", " << dRes << "), mu = " << mu << std::endl;
                break;
            }
            
            if(pRes > 10*dRes){
                mu *= 2;
            }else if(dRes > 10*pRes){
                mu /= 2;
            }
        }
        X = (X + X.transpose()) / 2;
        return X;
    }

    Vec<int> getViewsFromDim(List& dimGroups){
        Vec<int> lists(dimGroups.back(), -1);
        for(int i=0;i<dimGroups.size() - 1;i++){
            for(int c=dimGroups[i];c<dimGroups[i+1];c++){
                lists[c] = i;
            }
        }
        return lists;
    }

    Vec<int> getDimsFromViews(List& views){
        Vec<int> dims = {0};
        int startview = 0;
        for(int i=0;i<views.size();i++){
            if(views[i] != startview){
                dims.push_back(i);
                startview = views[i];
            }
        }
        dims.push_back(views.size());
        return dims;
    }
} // namespace match
