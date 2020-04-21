for b in data/arl/arl_husky_*.bag; do roslaunch launch/kitti-sort_ros-starmap.launch bagfile_basename:=$(pwd)/${b/.bag/}; done
