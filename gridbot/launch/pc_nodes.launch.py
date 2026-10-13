import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    DeclareLaunchArgument,
)
from launch.substitutions import LaunchConfiguration

from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch_ros.actions import Node
from launch.conditions import IfCondition

def generate_launch_description():

    rosout_level_arg = DeclareLaunchArgument(
        "rosout_level",
        default_value="debug",
        description="Rosout logger severity (error, debug, warn, info)",
        choices=["error", "warn", "debug", "info"],
    )

    rosout_level = LaunchConfiguration("rosout_level")

    foxglove_arg = DeclareLaunchArgument(
        "foxglove",
        default_value="true",
        description="Launch Foxglove server",
    )

    pkg_name = "gridbot"
    pkg_share = get_package_share_directory(pkg_name)

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
        arguments=[
            "--ros-args",
            "--log-level",
            [f"{pkg_name}.grid_processor:=", rosout_level],
        ],
    )

    route_navigator = Node(
        package="gridbot",
        executable="route_navigator",
        name="route_navigator",
        namespace=pkg_name,
        output="screen",
        parameters=[route_navigator_config],
        arguments=[
            "--ros-args",
            "--log-level",
            [f"{pkg_name}.route_navigator:=", rosout_level],
        ],
    )



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
        parameters=[route_generator_config],
        arguments=[
            "--ros-args",
            "--log-level",
            [f"{pkg_name}.route_generator:=", rosout_level],
        ],
    )

    route_navigator = Node(
        package="gridbot",
        executable="route_navigator",
        name="route_navigator",
        namespace=pkg_name,
        output="screen",
        parameters=[route_navigator_config],
        arguments=[
            "--ros-args",
            "--log-level",
            [f"{pkg_name}.route_navigator:=", rosout_level],
        ],
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

    
    twist_mux_config = os.path.join(
        pkg_share,
        "config",
        "twist_mux_params.yaml",
    )

    twist_mux = Node(
        package="twist_mux",
        executable="twist_mux",
        name="twist_mux",
        namespace=pkg_name,
        output="screen",
        parameters=[twist_mux_config],
        arguments=[
            "--ros-args",
            "--log-level",
            [f"{pkg_name}.twist_mux:=", rosout_level],
<<<<<<< HEAD
        ],
        remappings=[
            ("cmd_vel_out", "twist_mux/cmd_vel"),
=======
>>>>>>> origin/new-driving
        ]
    )

    return LaunchDescription(
        [
            rosout_level_arg,
            foxglove_arg,
            grid_processor,
            route_navigator,
            route_generator,
            twist_mux,
            foxglove_launch,
        ]
    )
