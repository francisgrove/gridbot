import os

from launch import LaunchDescription

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    pkg_name = "gridbot"
    pkg_share = get_package_share_directory(pkg_name)

    motor_driver_config = os.path.join(pkg_share, "config", "motor_driver_params.yaml")

    composable_nodes = [
        ComposableNode(
            package="camera_ros",
            plugin="camera::CameraNode",
            name="camera_node",
            namespace=pkg_name,
            parameters=[{"orientation": 180, "width": 640, "height": 480}],
            remappings=[
                ("image_raw", f"/{pkg_name}/image_raw"),
                ("camera_info", f"/{pkg_name}/camera_info"),
            ],
        ),
        ComposableNode(
            package="image_proc",
            plugin="image_proc::RectifyNode",
            name="rectify_node",
            namespace=pkg_name,
            remappings=[
                ("image", f"/{pkg_name}/image_raw"),
                ("camera_info", f"/{pkg_name}/camera_info"),
                ("image_rect", f"/{pkg_name}/image_rect"),
            ],
        ),
        ComposableNode(
            package="image_proc",
            plugin="image_proc::CropDecimateNode",
            name="crop_decimate_node",
            namespace=pkg_name,
            remappings=[
                ("image", f"/{pkg_name}/image_raw"),
                ("camera_info", f"/{pkg_name}/camera_info"),
                ("image_rect", f"/{pkg_name}/image_rect"),
            ],
            parameters={
                "decimation_x": 2,
                "decimation_y": 2,
            },  # pyright: ignore[reportArgumentType]
        ),
    ]

    container = ComposableNodeContainer(
        name="image_proc_container",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=composable_nodes,
    )  # pyright: ignore[reportCallIssue]

    # flip, as the robot's camera is physically upside-down

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
            container,
            motor_driver,
        ]
    )
