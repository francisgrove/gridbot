a ROS2 package for Alphabot2 that allows it to navigate a user-defined route on a grid.

## Prerequisites:
1. Make sure [ROS2 kilted is installed and you have built a workspace](https://docs.ros.org/en/kilted/Installation.html):
2. Run rosdep for the `gridbot` package:
```bash
rosdep install --from-paths src --ignore-src -r -y
```
**If running the node on a physical robot:**
Build the [camera_ros package (Pi)](https://github.com/christianrauch/camera_ros) in a separate workspace.

## Installation:

1. Clone this repository into your ros2 workspace's `src` folder: 
```bash
git clone https://github.com/wurlmon/gridbot.git
```

2. Build the `gridbot_interfaces` package first:
```bash
colcon build --packages-select gridbot_interfaces
```

3. Build the `gridbot` package:
```bash
colcon build --packages-select gridbot
```

You can add `--symlink-install` to make python edits to the python files without having to recompile:
```bash
colcon build --packages-select gridbot --symlink-install
```

## Running

Each node can run independently, like so:
```bash
ros2 run gridbot motor_driver
```

To use it with a YAML parameter file:
```bash
ros2 run gridbot motor_driver  --ros-args --params-file /home/$USER/ros2_ws/src/gridbot/config/motor_driver_params.yaml -r __ns:=/gridbot
```

(the namespace parameter `__ns` is set as I've had issues with running the yaml files without it, as they require the structure at the top level to be: `/namespace/node:`)

## Generating a map

Before running any of the nodes independently, please first generate the gridline's map via:
```bash
ros2 run gridbot  map_generator  --ros-args --params-file /home/$USER/ros2_ws/src/gridbot/config/map_generator_params.yaml -r __ns:=/gridbot
```

`map_generator_params` features an exemplary grid scheme. If you wish to make your own, consider editing following parameters of `map_generator_params.yaml` :
* `graph_width` - width (number of columns) of the grid.
* `graph_height` - height (number of rows) of the grid.
* `removed_nodes` - nodes that should be removed from the graph. The program first generates the whole map, and only later removes these nodes.
* `edges` - Edges, along with weights, that should run through the grid. The node will reject any edges that connect to an nonexistent node.

**note**: `line_color` is specified in BGR, not RGB (meaning that e.g. `[0,0,255]` is red)

## Launching the package

1. Simulation

You can launch the whole package in a Gazebo simulation via 
```bash
ros2 launch gridbot sim.launch.py
```
along with:
* `gui:=` (either `true` or `false`) to enable/disable the Gazebo GUI 
* `rosout_level:=` (either `info`, `debug`, `warn`, `error`) - sets the rosout level for all nodes used in the launch file
* `foxglove:=` (either `true` or `false`) to enable/disable a Foxglove server.
Please note that running this file will generate the map defined in `map_generator_params.yaml` automatically, and will not check for an existing output from it.

2. Physical

You can also run the package on a Alphabot2 with RPI via:
```bash
ros2 launch gridbot robot_nodes.launch.py
```
along with:
* `motors:=` (either (`true` or `false`) to enable/disable the Motor Driver node.

(note you need to have the `camera_ros` package, as said above).

This will run the minimal code needed on the robot itself.

Then, on your PC run:
```bash
ros2 launch gridbot pc_nodes.launch.py
```
### Using Foxglove

If you're using Foxglove, you can import the panel template from `foxglove/gridbot_foxglove_template_.json`.

## Moving the robot

(This section is written following the `route_navigator.py` node)

The robot listens to directions sent to the `gridbot/user_cmd` topic that expects String type messages (e.g. "A3 B2 C1").

It will generate intermediate nodes between these nodes via A* and convert them to directions suitable for a Diff-drive robot (Move, Turn Left/Right).

**Moving:**
1. The robot listens to `line_offset` topic for the offset from the grid line's center, and uses a PID controller to correct its course.
2. The robot listens to a `aruco` topic. Whenever the robot stands on a tag, it will receive a message to this topic with the tag's ID and direction, relative to the robot's orientation. 

**Turning:**
1. The robot listens to the `line_turning` topic. Anytime the line observations change (whether a line is visible on the left, central or right side of the image), a new message is sent to this topic. When a new line is centered (new meaning, we previously lost a line) the robot considers itself to have made a proper turn).
