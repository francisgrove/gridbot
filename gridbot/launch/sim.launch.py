import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

pkg_name = "gridbot"
pkg_share = get_package_share_directory(pkg_name)


def generate_launch_description():

    gui_arg = DeclareLaunchArgument(
        "gui",
        default_value="true",
        description="Launch Gazebo GUI",
    )

    foxglove_arg = DeclareLaunchArgument(
        "foxglove",
        default_value="true",
        description="Launch Foxglove server",
    )

    graph_arg = DeclareLaunchArgument(
        "graph",
        description="Name of the graph config file (without .yaml extension), e.g. map_graph_8x8",
    )

    launch_dir = os.path.join(pkg_share, "launch")

    sim_base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_dir, "sim_base.launch.py")
        ),
        launch_arguments={
            "gui": LaunchConfiguration("gui"),
            "foxglove": LaunchConfiguration("foxglove"),
        }.items(),
    )

    sim_nodes = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_dir, "sim_nodes.launch.py")
        ),
        launch_arguments={
            "graph": LaunchConfiguration("graph"),
        }.items(),
    )

    return LaunchDescription(
        [
            gui_arg,
            foxglove_arg,
            graph_arg,
            sim_base,
            sim_nodes,
        ]
    )
