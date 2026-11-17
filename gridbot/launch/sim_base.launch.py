import os
import xacro
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    DeclareLaunchArgument,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition

pkg_name = "gridbot"
pkg_share = get_package_share_directory(pkg_name)


def generate_launch_description():
    robot_xacro = os.path.join(pkg_share, "description", "robot.urdf.xacro")
    robot_description = xacro.process_file(robot_xacro).toxml()

    world_path = os.path.join(pkg_share, "assets/worlds", "world.sdf")

    set_gz_model_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=os.path.join(pkg_share, "assets/models"),
    )

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

    gazebo_launch_path = os.path.join(
        get_package_share_directory("ros_gz_sim"),
        "launch",
        "gz_sim.launch.py",
    )

    gazebo_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_path),
        launch_arguments={
            "gz_args": [
                "-r ",
                world_path,
            ]
        }.items(),
        condition=IfCondition(LaunchConfiguration("gui")),
    )

    gazebo_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_path),
        launch_arguments={
            "gz_args": [
                "-sr ",
                world_path,
            ]
        }.items(),
        condition=UnlessCondition(LaunchConfiguration("gui")),
    )

    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        namespace=pkg_name,
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": True,
            }
        ],
    )

    joint_state_pub = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        namespace=pkg_name,
        output="screen",
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic",
            "robot_description",
            "-name",
            "road_bot",
            "-x",
            "-0.80",
            "-y",
            "0.82",
            "-z",
            "0.15",
            "-R",
            "0.00",
            "-P",
            "0.0",
            "-Y",
            "3.1415",
        ],
        namespace=pkg_name,
        output="screen",
    )

    bridge_config = os.path.join(
        pkg_share,
        "config",
        "ros_gz_bridge_params.yaml",
    )

    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        namespace=pkg_name,
        output="screen",
        parameters=[{"config_file": bridge_config}],
    )

    camera_compressor = Node(
        package="image_transport",
        executable="republish",
        name="camera_compressor",
        parameters=[{"in_transport": "raw", "out_transport": "compressed"}],
        remappings=[
            ("in", f"/{pkg_name}/camera/image_raw"),
            ("out/compressed", f"/{pkg_name}/camera/image_raw/compressed"),
        ],
        output="screen",
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
            gui_arg,
            foxglove_arg,
            set_gz_model_path,
            gazebo_headless,
            gazebo_gui,
            robot_state_pub,
            joint_state_pub,
            spawn_robot,
            ros_gz_bridge,
            camera_compressor,
            foxglove_launch,
        ]
    )
