import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    TimerAction,
)
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, FindExecutable


def launch_setup(context):
    pkg_name = "gridbot"
    pkg_share = get_package_share_directory(pkg_name)

    params = LaunchConfiguration("params").perform(context)
    graph_name = LaunchConfiguration("name").perform(context)
    overwrite = LaunchConfiguration("overwrite").perform(context)

    map_generator_config = os.path.join(pkg_share, "config", f"{params}.yaml")

    map_generator = Node(
        package="gridbot",
        executable="map_generator",
        name="map_generator",
        namespace=pkg_name,
        output="screen",
        parameters=[map_generator_config],
    )

    service_call = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    FindExecutable(name="ros2"),
                    "service", "call",
                    f"/{pkg_name}/map_generator/generate_map",
                    "gridbot_interfaces/srv/GenerateMap",
                    f"{{map_name: '{graph_name}', overwrite: {overwrite}}}",
                ],
                output="screen",
            ),
        ],
    )

    return [map_generator, service_call]


def generate_launch_description():
    params_arg = DeclareLaunchArgument(
        "params",
        default_value="map_generator_size_params",
        description="Name of the map generator parameter file (without .yaml extension)",
    )

    name_arg = DeclareLaunchArgument(
        "name",
        default_value="map_graph_8x8",
        description="Name for the generated graph (used as filename and map_name)",
    )

    overwrite_arg = DeclareLaunchArgument(
        "overwrite",
        default_value="false",
        description="Overwrite map graph if it already exists",
    )

    return LaunchDescription(
        [
            params_arg,
            name_arg,
            overwrite_arg,
            OpaqueFunction(function=launch_setup),
        ]
    )
