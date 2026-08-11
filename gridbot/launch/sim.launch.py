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


def generate_launch_description():
    pkg_name = "gridbot"
    pkg_share = get_package_share_directory(pkg_name)

    robot_xacro = os.path.join(pkg_share, "description", "robot.urdf.xacro")
    robot_description = xacro.process_file(robot_xacro).toxml() # type: ignore

    world_path = os.path.join(pkg_share, "sim_assets/worlds", "world.sdf")

    set_gz_model_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=os.path.join(pkg_share, "sim_assets/models"),
    )

    gui_arg = DeclareLaunchArgument(
        "gui",
        default_value="true",
        description="Launch Gazebo GUI",
    )

    rosout_level_arg = DeclareLaunchArgument(
        "rosout_level",
        default_value="debug",
        description="Rosout logger severity (error, debug, warn, info)",
        choices=["error", "warn", "debug", "info"],
    )

    rosout_level = LaunchConfiguration("rosout_level")

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
            "-0.46",
            "-y",
            "0.6",
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

    map_generator_config = os.path.join(
        pkg_share,
        "config",
        "map_generator_params.yaml",
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
        namespace=pkg_name,
        arguments=["raw", "compressed"],
        remappings=[
            ("in", "camera/image_raw"),
            ("out/compressed", "camera/image_raw/compressed"),
        ],
        output="screen",
    )

    map_generator = Node(
        package="gridbot",
        executable="map_generator",
        name="map_generator",
        namespace=pkg_name,
        output="screen",
        parameters=[map_generator_config],
        arguments=[
            "--ros-args",
            "--log-level",
            [f"{pkg_name}.map_generator:=", rosout_level],
        ],
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

    foxglove_pkg_share = get_package_share_directory("foxglove_bridge")
    foxglove_launch_path = os.path.join(
        foxglove_pkg_share,
        "launch",
        "foxglove_bridge_launch.xml",
    )

    foxglove_launch = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(foxglove_launch_path)
    )

    return LaunchDescription(
        [
            gui_arg,
            rosout_level_arg,
            set_gz_model_path,
            gazebo_gui,
            gazebo_headless,
            camera_compressor,
            robot_state_pub,
            joint_state_pub,
            ros_gz_bridge,
            spawn_robot,
            map_generator,
            grid_processor,
            route_navigator,
            foxglove_launch,
        ]
    )
