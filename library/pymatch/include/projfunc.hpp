#pragma once
#include <vector>
#include <iostream>
#include "base.h"

namespace match
{
    Mat proj2pav(Mat y);
    Mat projR(Mat X);
    Mat projC(Mat X);

    Mat myproj2dpam(Mat Y, float tol = 1e-4, bool debug = false)
    {
        Mat X0 = Y;
        Mat X = Y;
        Mat I2 = X;
        I2.setZero();
        Mat X1, I1, X2;
        for (int iter = 0; iter < 10; iter++)
        {
            X1 = projR(X0 + I2);
            I1 = X1 - (X0 + I2);
            X2 = projC(X0 + I1);
            I2 = X2 - (X0 + I1);
            float chg = (X2 - X).array().abs().sum() / (X.rows() * X.cols());
            X = X2;
            if (chg < tol)
            {
                return X;
            }
        }
        return X;
    }

    Mat projR(Mat X)
    {
        int n = X.cols();
        // std::cout << "before projR: " << X << std::endl;
        for (int i = 0; i < X.rows(); i++)
        {
            Mat x = proj2pav(X.block(i, 0, 1, n).transpose());
            X.block(i, 0, 1, n) = x.transpose();
        }
        // std::cout << "after projR: " << X << std::endl;
        return X;
    }

    Mat projC(Mat X)
    {
        int n = X.rows();
        // std::cout << "before projC: " << X << std::endl;
        for (int j = 0; j < X.cols(); j++)
        {
            Mat x = proj2pav(X.block(0, j, n, 1));
            X.block(0, j, n, 1) = x;
        }
        // std::cout << "after projC: " << X << std::endl;
        return X;
    }

    Mat proj2pav(Mat y)
    {
        y = y.cwiseMax(0.f);
        Mat x = y;
        x.setZero();
        if (y.sum() < 1)
        {
            x = y;
        }
        else
        {
            std::vector<float> u, sv;
            for (int i = 0; i < y.rows(); i++)
            {
                u.push_back(y(i, 0));
            }
            // 排序
            std::sort(u.begin(), u.end(), std::greater<float>());
            float usum = 0;
            for (int i = 0; i < u.size(); i++)
            {
                usum += u[i];
                sv.push_back(usum);
            }
            int rho = 0;
            for (int i = 0; i < u.size(); i++)
            {
                if (u[i] > (sv[i] - 1) / (i + 1))
                {
                    rho = i;
                }
            }
            float theta = std::max(0.f, (sv[rho] - 1) / (rho + 1));
            x = (y.array() - theta).matrix();
            x = x.cwiseMax(0.f);
        }
        return x;
    }

    Mat proj2pavC(Mat y)
    {
        // y: N, 1
        // y[y<0] = 0
        int n = y.rows();
        y = y.cwiseMax(0.f);
        Mat x = y;
        if (y.sum() < 1)
        {
            x = y;
        }
        else
        {
            std::vector<float> u;
            for (int i = 0; i < y.rows(); i++)
            {
                u.push_back(y(i, 0));
            }
            // 排序
            std::sort(u.begin(), u.end(), std::greater<float>());
            float tmpsum = 0;
            bool bget = false;
            float tmax;
            for (int ii = 0; ii < n - 1; ii++)
            {
                tmpsum += u[ii];
                tmax = (tmpsum - 1) / (ii + 1);
                if (tmax >= u[ii + 1])
                {
                    bget = true;
                    break;
                }
            }
            if (!bget)
            {
                tmax = (tmpsum + u[n - 1] - 1) / n;
            }
            x = (y.array() - tmax).matrix();
            x = x.cwiseMax(0.f);
        }
        return x;
    }

    int proj201(Mat &z)
    {
        z = z.cwiseMin(1.f).cwiseMax(0.f);
        return 0;
    }

    Mat proj2kav_(Mat x0, Mat A, Mat b)
    {
        // to solve:
        // min 1/2||x - x_0||_F^2 + ||z||_1
        // s.t. Ax = b, x-z=0, x>=0, x<=1
        // convert to L(x, y) = 1/2||x - x_0||_F^2 + y^T(Ax - b)
        // x = (I + \rho A^T @A)^-1 @ (x_0 - A^T@y + \rho A^T@b)
        // y = y + \rho *(Ax - b)
        int n = x0.rows();
        Mat I(n, n);
        I.setIdentity();
        Mat X = x0;
        Mat y = b;
        float rho = 2;
        y.setZero();
        float tol = 1e-4;
        Mat Y, B, Z, c;
        for (int iter = 0; iter < 100; iter++)
        {
            Mat X0 = X;
            // (x - x_0) + A^Ty + \rho A^T(Ax + By -c)
            X = (I + rho * A.transpose() * A).ldlt().solve(x0 - A.transpose() * y + rho * A.transpose() * b);
            y = y + rho * (A * X - b);

            Y = Y + rho * (A * X + B * Z - c);
            float pRes = (A * X + B * Z - c).norm() / n;
            float dRes = rho * (X - X0).norm() / n;
            // std::cout << "  Iter " << iter << ", Res = (" << pRes << ", " << dRes << "), rho = " << rho << std::endl;

            if (pRes < tol && dRes < tol)
                break;
            if (pRes > 10 * dRes)
            {
                rho *= 2;
            }
            else if (dRes > 10 * pRes)
            {
                rho /= 2;
            }
        }
        return X;
    }

    Mat softthres(Mat b, float thres)
    {
        // TODO:vector
        for (int i = 0; i < b.rows(); i++)
        {
            if (b(i, 0) < -thres)
            {
                b(i, 0) += thres;
            }
            else if (b(i, 0) > thres)
            {
                b(i, 0) -= thres;
            }
            else
            {
                b(i, 0) = 0;
            }
        }
        return b;
    }

    Array Dsoft(const Array &d, float penalty)
    {
        // inverts the singular values
        // takes advantage of the fact that singular values are never negative
        Array di(d.rows(), d.cols());
        int maxRank = 0;
        for (int j = 0; j < d.size(); ++j)
        {
            double penalized = d(j, 0) - penalty;
            if (penalized < 0)
            {
                di(j, 0) = 0;
            }
            else
            {
                di(j, 0) = penalized;
                maxRank++;
            }
        }
        // std::cout << "max rank: " << maxRank << std::endl;
        return di;
    }

    Mat _proj2kav(Mat x0, Mat A, Mat b, Mat weight)
    {
        // to solve:
        // min 1/2||x - x_0||_F^2 + 1/2\lambda||Ax - b||_F^2 + \alpha||z||_1
        // s.t. x=z, z \in {z| z>=0, z<=1|
        // convert to L(x, y) = 1/2||x - x_0||_F^2 + 1/2\lambda||Ax - b||_F^2 + \alpha||z||_1
        //       + <y, x - z> + \rho/2||x - z||_F^2
        // update x:
        //     x = (1/rho + I + lambda/rhoA^TA)^-1 @ (1/rho x_0 + lambda/rho A^Tb + y)
        // update z:
        //     z = softthres(x + 1/rho y, lambda/rho)
        // update y:
        //     y = y + \rho *(x - z)
        int n = x0.rows();
        Mat I(n, n);
        I.setIdentity();
        Mat X = x0;
        Mat Y = X;
        Mat Z = Y;
        Y.setZero();
        float rho = 64;
        // weight
        float w_init = 1;
        float w_paf = 1e-1;
        float w_Ax = 100;
        float w_l1 = 1e-1;
        float tol = 1e-4;
        std::cout << "x0: " << x0 << std::endl;
        std::cout << "paf: " << weight << std::endl;
        for (int iter = 0; iter < 100; iter++)
        {
            Mat X0 = X;
            // update X
            X = ((rho + w_init) * I + w_Ax * A.transpose() * A).ldlt().solve(x0 + w_Ax * A.transpose() * b + rho * Z - Y + w_paf * weight);
            // update Z
            Z = softthres(X + 1 / rho * Y, w_l1 / rho);
            // projection Z
            Z = Z.cwiseMin(1.f).cwiseMax(0.f);
            // update Y
            Y = Y + rho * (X - Z);
            // convergence
            float pRes = (X - Z).norm() / n;
            float dRes = rho * (X - X0).norm() / n;
            std::cout << "  proj2kav Iter " << iter << ", Res = (" << pRes << ", " << dRes << "), rho = " << rho << std::endl;
            std::cout << " init= " << w_init * 0.5 * (X - x0).norm() / n
                      << ", equ= " << 0.5 * w_Ax * (A * X - b).norm() / n
                      << ", paf=" << -w_paf * weight.transpose() * X
                      << ", l1= " << 1.0 * (X.array() > 0).count() / n << std::endl;

            if (pRes < tol && dRes < tol)
                break;
            if (pRes > 10 * dRes)
            {
                rho *= 2;
            }
            else if (dRes > 10 * pRes)
            {
                rho /= 2;
            }
        }
        return X;
    }

    Mat proj2kav(Mat x0, Mat A, Mat b, Mat paf)
    {
        // reduce this problem

        std::vector<int> indices;
        // here we directly set the non-zero entries
        for (int j = 0; j < A.cols(); j++)
        {
            if (A(A.rows() - 1, j) != 0)
            {
                indices.push_back(j);
            }
        }
        // just use the last row
        int n = indices.size();
        Mat Areduce(1, n);
        Areduce.setOnes();
        Mat x0reduce(n, 1), pafreduce(n, 1);
        for (int i = 0; i < n; i++)
        {
            x0reduce(i, 0) = x0(indices[i], 0);
            pafreduce(i, 0) = paf(indices[i], 0);
        }
        Mat breduce(1, 1);
        breduce(0, 0) = b(b.rows() - 1, 0);

        Mat xreduce = _proj2kav(x0reduce, Areduce, breduce, pafreduce);
        Mat X(x0.rows(), 1);
        X.setOnes();
        for (int i = 0; i < n; i++)
        {
            X(indices[i], 0) = xreduce(i, 0);
        }
        return X;
    }

} // namespace match
