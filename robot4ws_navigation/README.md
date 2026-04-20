# robot4ws_navigation

trying to implement some easy path planner and follower. Map is given by external source for now (for drone scenario)

based on navigation2 stuck: [documentation](https://docs.nav2.org/) / [git repo](https://github.com/ros-navigation/navigation2)

## Global map assembly
 - NOTE: this might be bullshit. Need to figure out how to manage separate map source (drone) and user (rover)
 - ``odom``->``base_link`` transform (tf2) given by external source (odometry plugin, already in Archimede XACRO)
 - TODO: ``map`` -> ``odom`` transform (tf2), to figure out yet
 - local map published by the drone on the ``/map`` topic
 - costmap_2d reads ``/map``

## Dependencies
```bash
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
```
