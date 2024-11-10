cmake_minimum_required(VERSION 3.10)

# 项目名称
project(Demo)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 设置 OpenCV 路径
set(OpenCV_DIR "/home/lw/lib/opencv4.10.0_install/lib/cmake/opencv4")

# 查找 OpenCV 包
find_package(OpenCV REQUIRED)

# 包含 OpenCV 头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加源文件
add_executable(main main.cpp)

# 链接 OpenCV 库
target_link_libraries(main ${OpenCV_LIBS})

message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
