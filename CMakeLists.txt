cmake_minimum_required(VERSION 3.0.2)
project(search_navigation_challenge)

find_package(catkin REQUIRED COMPONENTS
  rospy
  husarion_ros
  rosbot_description
  std_msgs
  move_base_msgs
  actionlib_msgs
)

################################################
## Declare ROS messages, services and actions ##
################################################
## Generate actions in the 'action' folder
add_action_files(
  FILES
  Explore.action
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  move_base_msgs
  std_msgs
  actionlib_msgs
)

catkin_package(
  CATKIN_DEPENDS
  rospy
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

catkin_install_python(PROGRAMS
  src/tracker.py
  src/controller.py
  src/vision.py
  src/explorer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY .rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#############
## Testing ##
#############