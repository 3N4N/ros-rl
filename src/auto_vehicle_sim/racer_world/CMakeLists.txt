cmake_minimum_required(VERSION 3.12)
project(racer_world)

## Find catkin macros and libraries
find_package(catkin REQUIRED)

find_package(gazebo REQUIRED)
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})

find_package(Qt5 REQUIRED
    Core
    PrintSupport
    Gui
    Widgets
)
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})

# New in gazebo6: c++11 flags are now required
# for all downstream software to compile against gazebo.
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")

add_library(contact SHARED src/wall_contact.cc)
target_link_libraries(contact ${GAZEBO_LIBRARIES})
