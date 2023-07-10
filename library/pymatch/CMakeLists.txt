cmake_minimum_required(VERSION 3.12.0)
project(MATCHLR VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "-pthread -O3 -fPIC")

add_definitions(-DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

set(Eigen3_DIR "${PROJECT_SOURCE_DIR}/../../3rdparty/eigen-3.3.7/share/eigen3/cmake")
set(PYBIND11_DIR "${PROJECT_SOURCE_DIR}/../../3rdparty/pybind11")

find_package(Eigen3 3.3.7 REQUIRED)
message(STATUS "Eigen3 path is ${EIGEN3_INCLUDE_DIR}")

set(USE_OPENCV 0)
if(USE_OPENCV)
    # OpenCV
    find_package(OpenCV REQUIRED)
    add_definitions(-D_USE_OPENCV_)
endif(USE_OPENCV)

IF (WIN32)
	MESSAGE(STATUS "I don't test on Windows")
ELSEIF (APPLE)
	MESSAGE(STATUS "Not use openmp")
ELSEIF (UNIX)
  find_package(OpenMP REQUIRED)
  set(OTHER_LIBS ${OTHER_LIBS} OpenMP::OpenMP_CXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
ENDIF ()

set(PUBLIC_INCLUDE ${PROJECT_SOURCE_DIR}
                   ${PROJECT_SOURCE_DIR}/include
                   ${EIGEN3_INCLUDE_DIR}
)

set(OTHER_SRCS "")


if(USE_OPENCV)
    set(PUBLIC_INCLUDE ${PUBLIC_INCLUDE} ${OpenCV_INCLUDE_DIRS})
    set(OTHER_LIBS ${OTHER_LIBS} ${OpenCV_LIBRARIES})
endif(USE_OPENCV)

# add_subdirectory(test)
add_subdirectory(${PYBIND11_DIR} "PYBIND11_out")
add_subdirectory(python)
# add_subdirectory(src)
