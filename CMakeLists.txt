cmake_minimum_required( VERSION 2.8 )
project( get_depth )

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# opencv 
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


# pcl 
FIND_PACKAGE( PCL REQUIRED COMPONENTS common io visualization filters )
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

add_executable( get_depth  get_depth.cpp )
target_link_libraries( get_depth ${OpenCV_LIBS} ${PCL_LIBRARIES} )
