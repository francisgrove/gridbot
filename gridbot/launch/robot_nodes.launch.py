import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_name = "gridbot"
    pkg_share = get_package_share_directory(pkg_name)
    
    motor_driver_config = os.path.join(pkg_share, "config", "motor_driver_params.yaml")

    # flip, as the robot's camera is physically upside-down
    camera_node = Node(
        package="camera_ros",
        executable="camera_node",
        name="camera",
        namespace=pkg_name,
        parameters=[{
            "orientation": 180
        }]
    )

    motor_driver = Node(
        package="gridbot",
        executable="motor_driver",
        name="motor_driver",
        namespace=pkg_name,
        output="screen",
        parameters=[motor_driver_config],
    )

    return LaunchDescription(
        [
            camera_node,
            motor_driver,
        ]
    )
