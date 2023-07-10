/*** 
 * @Date: 2020-09-18 14:05:37
 * @Author: Qing Shuai
 * @LastEditors: Qing Shuai
 * @LastEditTime: 2021-07-24 14:50:42
 * @FilePath: /EasyMocap/library/pymatch/python/pymatchlr.cpp
 */
/*
 * @Date: 2020-06-29 10:51:28
 * @LastEditors: Qing Shuai
 * @LastEditTime: 2020-07-12 17:11:43
 * @Author: Qing Shuai
 * @Mail: s_q@zju.edu.cn
 */ 
#include <iostream>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include "pybind11/eigen.h"
#include "matchSVT.hpp"

#define myprint(x) std::cout << #x << ": " << std::endl << x.transpose() << std::endl;
#define printshape(x) std::cout << #x << ": (" << x.rows() << ", " << x.cols() << ")" << std::endl;

namespace py = pybind11;

PYBIND11_MODULE(pymatchlr, m) {
    m.def("matchSVT", &match::matchSVT, "SVT for matching", 
        py::arg("affinity"), py::arg("dimGroups"), py::arg("constraint"), py::arg("observe"), py::arg("debug"));
    m.def("matchALS", &match::matchALS, "ALS for matching", 
        py::arg("affinity"), py::arg("dimGroups"), py::arg("constraint"), py::arg("observe"), py::arg("debug"));
    m.attr("__version__") = "0.1.0";
}