import os

from launch import LaunchDescription

from launch.actions import (
    DeclareLaunchArgument,
)
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition


def generate_launch_description():
    pkg_name = "gridbot"
    pkg_share = get_package_share_directory(pkg_name)

    motors_arg = DeclareLaunchArgument(
        "motors",
        default_value="true",
        description="Use Motor Driver node. (useful to turn off when testing)",
    )

    camera_node = Node(
        package="camera_ros",
        executable="camera_node",
        name="camera",
        parameters=[{"orientation": 180, "width": 640, "height": 480}],
        namespace=pkg_name,
        remappings=[
            ("image_raw", "cam_image_raw"),
            ("camera_info", "camera_info"),
        ],
    )

    motor_driver_config = os.path.join(pkg_share, "config", "motor_driver_params.yaml")

    motor_driver = Node(
        package="gridbot",
        executable="motor_driver",
        name="motor_driver",
        namespace=pkg_name,
        output="screen",
        parameters=[motor_driver_config],
        condition=IfCondition(LaunchConfiguration("motors")),
    )
    
    camera_controller_config = os.path.join(
        pkg_share,
        "config",
        "camera_controller_params.yaml",
    )

    camera_controller = Node(
        package="gridbot",
        executable="camera_controller",
        name="camera_controller",
        namespace=pkg_name,
        output="screen",
        parameters=[camera_controller_config],
    )

    distance_measurer_config = os.path.join(
        pkg_share,
        "config",
        "distance_measurer_params.yaml",
    )

    distance_measurer = Node(
        package="gridbot",
        executable="distance_measurer",
        name="distance_measurer",
        namespace=pkg_name,
        output="screen",
        parameters=[distance_measurer_config],
    )

    return LaunchDescription(
        [
            # args
            motors_arg,
            # services
            camera_controller,
            distance_measurer,
            # camera
            camera_node,
            # nodes
            motor_driver,
        ]
    )
