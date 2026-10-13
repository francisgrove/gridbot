import os

from launch import LaunchDescription

from launch.actions import (
    DeclareLaunchArgument,
)
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition


def generate_launch_description():
    pkg_name = "gridbot"
    pkg_share = get_package_share_directory(pkg_name)

    motor_driver_config = os.path.join(pkg_share, "config", "motor_driver_params.yaml")

    camera_name = "camera"

    motors_arg = DeclareLaunchArgument(
        "motors",
        default_value="true",
        description="Use Motor Driver node. (useful to turn off when testing)",
    )

    camera_node = Node(
        package="camera_ros",
        executable="camera_node",
        name=camera_name,
        parameters=[{"orientation": 180, "width": 640, "height": 480}],
        namespace=pkg_name,
        remappings=[
            ("image_raw", "image_raw"),
            ("camera_info", "camera_info"),
        ],
    )

    composable_nodes = [
        ComposableNode(
            package="image_proc",
            plugin="image_proc::RectifyNode",
            name="rectify_node",
            namespace=pkg_name,
            remappings=[
                ("image", f"{camera_name}/image_raw"),
                ("image_rect", f"{camera_name}/image_rect"),
            ],
        ),
        ComposableNode(
            package="image_proc",
            plugin="image_proc::CropDecimateNode",
            name="crop_decimate_node",
            namespace=pkg_name,
            remappings=[
                ("in/image_raw", f"{camera_name}/image_rect"),
                ("in/camera_info", f"{camera_name}/camera_info"),
                ("out/image_raw", f"{camera_name}/image_downsized"),
                ("out/camera_info", f"{camera_name}/downsized_camera_info"),
            ],
            parameters=[
                {
                    "decimation_x": 2,
                    "decimation_y": 2,
                }
            ],
        ),
    ]

    container = ComposableNodeContainer(
        name="image_proc_container",
        package="rclcpp_components",
        executable="component_container",
        namespace=pkg_name,
        composable_node_descriptions=composable_nodes,
    )

    motor_driver = Node(
        package="gridbot",
        executable="motor_driver",
        name="motor_driver",
        namespace=pkg_name,
        output="screen",
        parameters=[motor_driver_config],
        condition=IfCondition(LaunchConfiguration("motors")),
<<<<<<< HEAD
        remappings=[("cmd_vel", "twist_mux/cmd_vel")],  # for twist_mux
=======
        # remappings=[("cmd_vel", "cmd_vel_out")],  # for twist_mux
>>>>>>> origin/new-driving
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

    return LaunchDescription(
        [
            motors_arg,
            camera_node,
            container,
            motor_driver,
            camera_controller
        ]
    )
