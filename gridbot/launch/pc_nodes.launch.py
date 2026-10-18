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
from launch_ros.actions import Node

pkg_name = "gridbot"
pkg_share = get_package_share_directory(pkg_name)


def launch_setup(context):
    graph = LaunchConfiguration("graph").perform(context)
    graph_filepath = os.path.join(pkg_share, "config", f"{graph}.yaml")

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

    return [route_generator]


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
            # services
            OpaqueFunction(function=launch_setup),
            # nodes
            grid_processor,
            route_navigator,
            # rest
            foxglove_launch,
        ]
    )
