import os
import xacro
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    DeclareLaunchArgument,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition

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

    graph_arg = DeclareLaunchArgument(
        "graph",
        description="Name of the graph config file (without .yaml extension), e.g. map_graph_8x8",
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
            "-1.35",
            "-y",
            "1.37",
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

    grid_processor_config = os.path.join(
        pkg_share,
        "config",
        "grid_processor_params.yaml",
    )

    grid_processor = Node(
        package="gridbot",
        executable="grid_processor",
        name="grid_processor",
        namespace=pkg_name,
        output="screen",
        parameters=[grid_processor_config],
    )

    route_navigator_config = os.path.join(
        pkg_share,
        "config",
        "route_navigator_params.yaml",
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
            gui_arg,
            foxglove_arg,
            graph_arg,
            set_gz_model_path,
            # services
            OpaqueFunction(function=launch_setup),
            # gazebo
            gazebo_headless,
            gazebo_gui,
            # states
            robot_state_pub,
            joint_state_pub,
            spawn_robot,
            ros_gz_bridge,
            # nodes
            grid_processor,
            route_navigator,
            # rest
            camera_compressor,
            foxglove_launch,
        ]
    )
