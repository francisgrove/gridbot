import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    DeclareLaunchArgument,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

pkg_name = "gridbot"
pkg_share = get_package_share_directory(pkg_name)


VALID_DECIMATIONS = {"1", "2", "4"}


def launch_setup(context):
    graph = LaunchConfiguration("graph").perform(context)
    graph_filepath = os.path.join(pkg_share, "config", f"{graph}.yaml")

    decimation = LaunchConfiguration("decimation").perform(context)
    if decimation not in VALID_DECIMATIONS:
        raise ValueError(
            f"Invalid decimation '{decimation}'. Must be one of: {', '.join(sorted(VALID_DECIMATIONS))}"
        )
    decimation = int(decimation)

    route_generator_config = os.path.join(
        pkg_share,
        "config",
        "route_generator_params.yaml",
    )

    route_generator = Node(
        package="gridbot",
        executable="route_generator",
        name="route_generator",
        namespace=pkg_name,
        output="screen",
        parameters=[route_generator_config, {"graph_filepath": graph_filepath}],
    )

    camera_name = "camera"

    composable_nodes = [
        ComposableNode(
            package="image_proc",
            plugin="image_proc::RectifyNode",
            name="rectify_node",
            namespace=pkg_name,
            remappings=[
                ("image", f"{camera_name}/cam_image_raw"),
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
                    "decimation_x": decimation,
                    "decimation_y": decimation,
                }
            ],
        ),
    ]

    image_proc_container = ComposableNodeContainer(
        name="image_proc_container",
        package="rclcpp_components",
        executable="component_container",
        namespace=pkg_name,
        composable_node_descriptions=composable_nodes,
    )

    return [route_generator, image_proc_container]


def generate_launch_description():

    foxglove_arg = DeclareLaunchArgument(
        "foxglove",
        default_value="true",
        description="Launch Foxglove server",
    )

    graph_arg = DeclareLaunchArgument(
        "graph",
        default_value="map_graph_8x8",
        description="Name of the graph config file (without .yaml extension)",
    )

    decimation_arg = DeclareLaunchArgument(
        "decimation",
        default_value="2",
        description="Image decimation factor (1, 2, or 4)",
    )

    grid_processor_config = os.path.join(
        pkg_share,
        "config",
        "grid_processor_params.yaml",
    )

    route_navigator_config = os.path.join(
        pkg_share,
        "config",
        "route_navigator_params.yaml",
    )

    grid_processor = Node(
        package="gridbot",
        executable="grid_processor",
        name="grid_processor",
        namespace=pkg_name,
        output="screen",
        parameters=[grid_processor_config],
        remappings=[
            ("image_raw", "image_downsized")
        ]
    )

    route_navigator = Node(
        package="gridbot",
        executable="route_navigator",
        name="route_navigator",
        namespace=pkg_name,
        output="screen",
        parameters=[route_navigator_config],
    )

    foxglove_pkg_share = get_package_share_directory("foxglove_bridge")
    foxglove_launch_path = os.path.join(
        foxglove_pkg_share,
        "launch",
        "foxglove_bridge_launch.xml",
    )

    foxglove_launch = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(foxglove_launch_path),
        condition=IfCondition(LaunchConfiguration("foxglove")),
    )

    return LaunchDescription(
        [
            # args
            foxglove_arg,
            graph_arg,
            decimation_arg,
            # services + camera's composable nodes
            OpaqueFunction(function=launch_setup),
            # nodes
            grid_processor,
            route_navigator,
            # rest
            foxglove_launch,
        ]
    )
