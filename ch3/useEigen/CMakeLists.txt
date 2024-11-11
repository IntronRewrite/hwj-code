cmake_minimum_required(VERSION 3.12)
project(useEigen)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-g")
set(CMAKE_CXX_STANDARD 11)

# 查找Eigen库
find_package(Eigen3 3.3 REQUIRED)

# 添加Eigen头文件
include_directories(${EIGEN3_INCLUDE_DIR})

# 添加标准库路径
include_directories( 
    "/usr/include" 
    "/usr/include/c++/11"
    "/usr/include/x86_64-linux-gnu/c++/11" # 如果有这个路径，也可以添加
)

# 添加可执行文件
add_executable(eigenMatrix eigenMatrix.cpp)
add_executable(test test.cpp)

# 链接Eigen库
target_link_libraries(eigenMatrix Eigen3::Eigen)
target_link_libraries(test Eigen3::Eigen)