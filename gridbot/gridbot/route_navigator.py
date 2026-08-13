import rclpy
from rclpy.node import Node
from typing import Any, cast
from rcl_interfaces.msg import (
    ParameterDescriptor,
    FloatingPointRange,
    ListParametersResult,
)
from rclpy.timer import Timer
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64, Int8, String
from geometry_msgs.msg import Twist
import gridbot.gridbot_helpers as gridbot
from gridbot_interfaces.msg import SimpleAruco
import yaml

from pathlib import Path
from ament_index_python.packages import get_package_share_directory


import numpy as np
from itertools import groupby
import heapq
from collections import defaultdict


class RouteNavigator(Node):

    class RouteNode:
        """
        Describes a singular node in the robot's route
        """

        name: str
        priority: int
        directions: list[gridbot.RobotDirection]

        def __init__(self, name, priority, directions):
            self.name = name
            self.priority = priority
            self.directions = directions

        def __str__(self):
            return f"{self.name}: ({[dir.name for dir in self.directions]})"

        def __repr__(self):
            return f"{self.name}: ({[dir.name for dir in self.directions]})"

    line_offset_topic: str
    aruco_topic: str
    twist_topic: str
    line_turning_topic: str
    user_cmd_topic: str

    graph: dict[str, dict[str, int]]
    graph_width: int
    graph_height: int

    curr_pos: str

    curr_dir: gridbot.ArucoDirection
    frequency: float

    pause_time: float
    lost_timeout: float

    curr_error: float
    sum_error: float
    prev_error: float

    prev_time: int | None
    pid_output: float
    k_p: float
    k_i: float
    k_d: float

    linear_vel: float
    angular_vel: float

    linear_vel_multiplier: float
    angular_vel_multiplier: float

    robot_route: list[RouteNode] = []

    # state machine var
    curr_state: gridbot.RobotState

    pause_start: int

    node_reached: bool

    last_line_observation: gridbot.LineObservation
    center_found: bool

    clear_nodes: bool

    twist_timer: Timer

    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.prev_time = 0
        self.prev_error = 0.0

        self.sum_error = 0.0
        self.curr_error = 0.0

        self.pid_output = 0.0

        self.curr_state = gridbot.RobotState.IDLE

        self.last_line_observation = gridbot.LineObservation.NONE

        self.clear_nodes = False
        self.node_reached = False
        self.center_found = False

        self._setup_parameters()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.user_cmd_sub = self.create_subscription(
            msg_type=String,
            topic=self.user_cmd_topic,
            callback=self.user_cmd_callback,
            qos_profile=qos,
        )

        self.line_offset_sub = self.create_subscription(
            msg_type=Float64,
            topic=self.line_offset_topic,
            callback=self.line_offset_callback,
            qos_profile=qos,
        )

        self.aruco_sub = self.create_subscription(
            msg_type=SimpleAruco,
            topic=self.aruco_topic,
            callback=self.aruco_callback,
            qos_profile=qos,
        )

        self.line_turning_sub = self.create_subscription(
            msg_type=Int8,
            topic=self.line_turning_topic,
            callback=self.line_turning_callback,
            qos_profile=qos,
        )

        self.twist_pub = self.create_publisher(
            msg_type=Twist, topic=self.twist_topic, qos_profile=sensor_qos
        )

        self.twist_timer = self.create_timer(
            timer_period_sec=(self.frequency), callback=self.timer_callback
        )

        self.context.on_shutdown(self._cleanup)

    def user_cmd_callback(self, msg: String):

        self.get_logger().info(f"Received user command:\n{msg.data}")

        if msg.data == "STOP":
            self.get_logger().info(f"Stopping robot at first available pause.")
            self.clear_nodes = True
            return

        nodes_from_msg = [node for node in msg.data.split()]

        for node in nodes_from_msg:
            if node not in self.graph:
                self.get_logger().warn(
                    f"Node {node} given as in user command not in graph."
                )
                nodes_from_msg.remove(node)

        if not nodes_from_msg:
            self.get_logger().error("No valid nodes provided.")
            return

        start = self.curr_pos
        self.robot_route = self._generate_route(start, nodes_from_msg)

    def line_offset_callback(self, msg: Float64) -> None:
        error = float(msg.data)

        now = self.get_clock().now().nanoseconds

        if self.prev_time is None:
            self.prev_time = now
            self.prev_error = error

            return

        duration = now - self.prev_time
        dt = duration * 1e-9

        if dt < 1e-6:
            return

        self.sum_error += error * dt
        drv_error = (error - self.prev_error) / dt

        self.pid_output = (
            self.k_p * error + self.k_i * self.sum_error + self.k_d * drv_error
        )

        self.prev_error = error
        self.prev_time = now

    def aruco_callback(self, msg: SimpleAruco):
        if self.curr_state != gridbot.RobotState.MOVING:
            return

        id = int(msg.id)
        dir = int(msg.direction)

        self.get_logger().info(
            f"Received ArUco tag data:\nTag: {id} -> {gridbot.aruco_to_node(id, self.graph_height)}\nDirection: {dir} -> {gridbot.ArucoDirection(dir).name.capitalize()}"
        )

        expected_node = self.robot_route[0].name

        received_node = gridbot.aruco_to_node(id, self.graph_height)

        if received_node != expected_node:

            self.get_logger().info(f"Expected {expected_node}, but got {received_node}")
            # obtain all nodes that have priority
            priority_nodes = [
                node.name for node in self.robot_route if node.priority == 1
            ]

            new_route = self._generate_route(received_node, priority_nodes)

            self.robot_route = new_route

        node_name = gridbot.aruco_to_node(id, self.graph_height)
        dir_enum = gridbot.ArucoDirection(dir)

        if self.curr_pos != node_name:
            self.curr_pos = node_name
        if self.curr_dir != dir_enum:
            self.curr_dir = dir_enum

        self.node_reached = True
        return

    def line_turning_callback(self, msg: Int8) -> None:

        if self.curr_state not in [gridbot.RobotState.TURNING]:
            return

        flags = gridbot.LineObservation(msg.data)

        if self.last_line_observation is None:
            self.last_line_observation = flags
        elif self.last_line_observation == flags:
            return

        curr_center = flags & gridbot.LineObservation.CENTER

        self.get_logger().info(
            f"Road flags:\n [{int(bool(flags & gridbot.LineObservation.LEFT))} "
            f"{int(bool(flags & gridbot.LineObservation.CENTER))} "
            f"{int(bool(flags & gridbot.LineObservation.RIGHT))}]"
        )

        if (
            not curr_center
            and self.last_line_observation & gridbot.LineObservation.CENTER
        ):
            self.get_logger().debug("Center lost")
            self.center_found = False

        # Only count a line being found if previously the line wasn't in the center
        if (
            curr_center
            and not self.last_line_observation & gridbot.LineObservation.CENTER
        ):
            self.get_logger().debug("Center found")
            self.center_found = True

        self.last_line_observation = flags

    def timer_callback(self):

        twist_msg = Twist()

        if not self.robot_route:
            return

        new_state = None

        next_node = None
        next_dir = None

        if self.curr_state != gridbot.RobotState.PAUSED:

            next_node = self.robot_route[0] if len(self.robot_route) > 0 else None
            next_dir = (
                next_node.directions[0]
                if next_node is not None and len(next_node.directions) > 0
                else None
            )

            if next_node is None or next_dir is None:
                return

            match next_dir:
                case gridbot.RobotDirection.MOVE:
                    new_state = gridbot.RobotState.MOVING
                case (
                    gridbot.RobotDirection.TURN_LEFT | gridbot.RobotDirection.TURN_RIGHT
                ):
                    new_state = gridbot.RobotState.TURNING
                case _:
                    self.get_logger().error(f"Direction {next_dir} not supported.")

        if new_state is not None and self.curr_state != new_state:
            self.curr_state = new_state
            self.get_logger().info(
                f"Robot state set to: {self.curr_state.name.capitalize()}"
            )

        match self.curr_state:
            case gridbot.RobotState.MOVING:

                twist_msg.linear.x = float(
                    self.linear_vel * self.linear_vel_multiplier,
                )
                twist_msg.angular.z = float(
                    self.angular_vel * self.angular_vel_multiplier * self.pid_output,
                )

                if self.node_reached:
                    self.get_logger().info(
                        f"Node reached - Robot at {self.curr_pos}, looking {gridbot.ArucoDirection(self.curr_dir).name.capitalize()}"
                    )

                    self.node_reached = False
                    self.curr_state = gridbot.RobotState.PAUSED
                    self.pause_start = self.get_clock().now().nanoseconds
            case gridbot.RobotState.TURNING:
                twist_msg.linear.x = 0.0

                turn_dir = (
                    -1
                    if next_dir == gridbot.RobotDirection.TURN_RIGHT
                    else 1 if next_dir == gridbot.RobotDirection.TURN_LEFT else 0
                )

                twist_msg.angular.z = float(
                    self.angular_vel * self.angular_vel_multiplier * turn_dir
                )

                if self.center_found:
                    self.get_logger().info(
                        f"Centered at line - Robot at {self.curr_pos}, looking {gridbot.ArucoDirection(self.curr_dir).name.capitalize()}"
                    )
                    self.center_found = False
                    self.curr_state = gridbot.RobotState.PAUSED
                    self.pause_start = self.get_clock().now().nanoseconds

            case gridbot.RobotState.PAUSED:
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0

                time_passed = self.get_clock().now().nanoseconds - self.pause_start
                if time_passed > self.pause_time * 1e9:
                    self.get_logger().info(f"Unpaused after {time_passed/1e9} seconds.")

                    if self.clear_nodes:
                        self.get_logger().info("Cleared route.")
                        self.robot_route.clear()
                        self.clear_nodes = False

                    if (
                        len(self.robot_route) > 0
                        and len(self.robot_route[0].directions) > 1
                    ):
                        self.robot_route[0].directions.pop(0)
                    elif len(self.robot_route) > 0:
                        self.robot_route.pop(0)

                    if len(self.robot_route) > 0:
                        self.get_logger().info(
                            f"Next route plan: {self.robot_route[0].directions[0].name.capitalize()} for node {self.robot_route[0]}"
                        )
                    else:
                        self.get_logger().info(f"Finished route.")
                    self.curr_state = gridbot.RobotState.IDLE
            case gridbot.RobotState.IDLE:
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0
            case _:
                pass

        self.twist_pub.publish(twist_msg)

    def _setup_parameters(self):
        self.declare_parameter(
            name="line_offset_topic",
            value="line_offset",
            descriptor=ParameterDescriptor(
                description="Topic used to receive line_offset information.",
            ),
        )

        self.declare_parameter(
            name="aruco_topic",
            value="aruco",
            descriptor=ParameterDescriptor(
                description="Topic used to receive aruco tag information (ID + relative direction of the robot to the tag's top).",
            ),
        )

        self.declare_parameter(
            name="line_turning_topic",
            value="line_turning",
            descriptor=ParameterDescriptor(
                description="Topic used to receive information about line visibility in each third of the camera.",
            ),
        )

        self.declare_parameter(
            name="user_cmd_topic",
            value="user_cmd",
            descriptor=ParameterDescriptor(
                description="Topic used to receive route nodes from the user.",
            ),
        )

        self.declare_parameter(
            name="twist_topic",
            value="cmd_vel",
            descriptor=ParameterDescriptor(
                description="Topic used to send twist messages.",
            ),
        )

        self.declare_parameter(
            name="map_graph_filepath",
            value="map_graph.yaml",
            descriptor=ParameterDescriptor(
                description="Filepath of the map graph yaml file. It contains the adjacency map as well as the map's dimensions",
            ),
        )

        self.declare_parameter(
            name="default_pos",
            value="A1",
            descriptor=ParameterDescriptor(
                description="Default starting position of the robot."
            ),
        )

        self.declare_parameter(
            name="default_dir",
            value="north",
            descriptor=ParameterDescriptor(
                description="Default starting rotation of the robot."
            ),
        )

        self.declare_parameter(
            name="frequency",
            value=0.005,
            descriptor=ParameterDescriptor(
                description="Frequency the timer controlling the robot's state machine will be running at.",
                floating_point_range=[
                    FloatingPointRange(
                        from_value=0,
                        to_value=float("inf"),
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="pause_time",
            value=1.0,
            descriptor=ParameterDescriptor(
                description="Time (in seconds) to pause the robot for after reaching a node/turning correctly.",
                floating_point_range=[
                    FloatingPointRange(
                        from_value=0,
                        to_value=float("inf"),
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="lost_timeout",
            value=5.0,
            descriptor=ParameterDescriptor(
                description="Time (in seconds) after the robot will be considered lost (counting starts when line turning retrieves [0 0 0].",
                floating_point_range=[
                    FloatingPointRange(
                        from_value=0,
                        to_value=float("inf"),
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="k_p",
            value=1.0,
            descriptor=ParameterDescriptor(
                description="Factor for the proportional part of the PID controller.",
                floating_point_range=[
                    FloatingPointRange(
                        from_value=0,
                        to_value=float("inf"),
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="k_i",
            value=0.0,
            descriptor=ParameterDescriptor(
                description="Factor for the integral part of the PID controller.",
                floating_point_range=[
                    FloatingPointRange(
                        from_value=0,
                        to_value=float("inf"),
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="k_d",
            value=0.0,
            descriptor=ParameterDescriptor(
                description="Factor for the derivative part of the PID controller.",
                floating_point_range=[
                    FloatingPointRange(
                        from_value=0,
                        to_value=float("inf"),
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="linear_vel",
            value=0.5,
            descriptor=ParameterDescriptor(
                description="Linear velocity value of the robot, in m/s.",
            ),
        )

        self.declare_parameter(
            name="angular_vel",
            value=10.0,
            descriptor=ParameterDescriptor(
                description="Linear velocity value of the robot, in m/s.",
            ),
        )

        self.declare_parameter(
            name="linear_vel_multiplier",
            value=1.0,
            descriptor=ParameterDescriptor(
                description="Linear velocity multiplier. Used to fine-control the actual values being sent.",
            ),
        )

        self.declare_parameter(
            name="angular_vel_multiplier",
            value=1.0,
            descriptor=ParameterDescriptor(
                description="Angular velocity multiplier. Used to fine-control the actual values being sent.",
            ),
        )

        self.line_offset_topic = self.get_parameter("line_offset_topic").value
        self.line_turning_topic = self.get_parameter("line_turning_topic").value
        self.aruco_topic = self.get_parameter("aruco_topic").value
        self.user_cmd_topic = self.get_parameter("user_cmd_topic").value
        self.twist_topic = self.get_parameter("twist_topic").value

        pkg_path = Path(get_package_share_directory(self.get_namespace().strip("/")))

        yaml_path = pkg_path / self.get_parameter("map_graph_filepath").value

        with open(yaml_path) as stream:
            try:
                load = cast(dict[str, Any] | None, yaml.safe_load(stream))

                if load is None:
                    self.get_logger().error("YAML file is empty.")
                    return

                self.graph = load["graph"]
                self.graph_height = load["graph_height"]
                self.graph_width = load["graph_width"]

            except yaml.YAMLError as e:
                self.get_logger().error(str(e))
        self.curr_pos = self.get_parameter("default_pos").value

        dir = str(self.get_parameter("default_dir").value).upper()

        self.curr_dir = gridbot.ArucoDirection[dir]

        self.pause_time = self.get_parameter("pause_time").value
        self.lost_timeout = self.get_parameter("lost_timeout").value

        self.frequency = self.get_parameter("frequency").value
        self.k_p = self.get_parameter("k_p").value
        self.k_i = self.get_parameter("k_i").value
        self.k_d = self.get_parameter("k_d").value

        self.linear_vel = self.get_parameter("linear_vel").value
        self.angular_vel = self.get_parameter("angular_vel").value

        self.linear_vel_multiplier = self.get_parameter("linear_vel_multiplier").value
        self.angular_vel_multiplier = self.get_parameter("angular_vel_multiplier").value

        result: ListParametersResult = self.list_parameters([], depth=0)

        parameters = self.get_parameters(list(result.names))

        self.get_logger().info("=" * 40)
        for parameter in parameters:
            self.get_logger().info(f"{parameter.name}: {parameter.value}")

        self.get_logger().info(
            "graph:\n" + "\n".join(f"{k}: {v}" for k, v in self.graph.items())
        )
        self.get_logger().info("=" * 40)

    def _cleanup(self):
        self.twist_pub.publish(Twist())

    def _generate_route(self, start_node: str, node_list: list[str]) -> list[RouteNode]:
        route = []

        node_pairs = list(zip([start_node] + node_list[:-1], node_list))

        paths_2d = []

        for start, goal in node_pairs:

            self.get_logger().debug(f"Calculating path from {start} to {goal}.")
            path = self.a_star(start, goal)
            self.get_logger().debug(f"A* calculated path: {path}")
            paths_2d.append(path)

        paths_1d = [node for path in paths_2d for node in path]
        path = [node for node, _ in groupby(paths_1d)]

        node_pairs = list(zip(path[:-1], path[1:]))

        for start, goal in node_pairs:

            node = self.RouteNode(
                name=goal, directions=[], priority=1 if goal in node_list else 0
            )
            route.append(node)

        self._add_directions(start_node=path[0], route=route)
        self.get_logger().info("Route is:\n" + "\n".join(str(node) for node in route))
        return route

    def _add_directions(
        self,
        start_node: str,
        route: list[RouteNode],
    ):
        if not route:
            return

        curr_dir = self.curr_dir
        curr_node = start_node

        for next_node in route:

            target_dir: gridbot.ArucoDirection | None = self._get_direction_between(
                curr_node,
                next_node.name,
            )

            if target_dir is None:
                return

            dir_diff = (target_dir.value - curr_dir.value) % 4

            match dir_diff:
                case 0:
                    # Already facing the correct direction.
                    pass

                case 1:
                    next_node.directions.append(gridbot.RobotDirection.TURN_RIGHT)

                case 2:
                    # 180 degree turn.

                    left_dir = gridbot.ArucoDirection((curr_dir.value - 1) % 4)
                    right_dir = gridbot.ArucoDirection((curr_dir.value + 1) % 4)

                    neighbours = self.graph.get(curr_node, {})

                    for neighbour in neighbours:
                        direction = self._get_direction_between(curr_node, neighbour)

                        if direction == left_dir:
                            self.get_logger().info(
                                f"{neighbour} to the left of {curr_node}"
                            )
                        if direction == right_dir:
                            self.get_logger().info(
                                f"{neighbour} to the right of {curr_node}"
                            )

                    has_left = any(
                        self._get_direction_between(curr_node, neighbour) == left_dir
                        for neighbour in neighbours
                    )

                    has_right = any(
                        self._get_direction_between(curr_node, neighbour) == right_dir
                        for neighbour in neighbours
                    )

                    if has_right and not has_left:
                        next_node.directions.append(gridbot.RobotDirection.TURN_RIGHT)
                        next_node.directions.append(gridbot.RobotDirection.TURN_RIGHT)
                    elif not has_right and has_left:
                        next_node.directions.append(gridbot.RobotDirection.TURN_LEFT)
                        next_node.directions.append(gridbot.RobotDirection.TURN_LEFT)
                    elif has_right and has_left:
                        next_node.directions.append(gridbot.RobotDirection.TURN_LEFT)
                        next_node.directions.append(gridbot.RobotDirection.TURN_LEFT)
                    else:
                        next_node.directions.append(gridbot.RobotDirection.TURN_LEFT)

                case 3:
                    next_node.directions.append(gridbot.RobotDirection.TURN_LEFT)

            next_node.directions.append(gridbot.RobotDirection.MOVE)

            curr_dir = target_dir
            curr_node = next_node.name

    def _get_direction_between(
        self, node_a: str, node_b: str
    ) -> gridbot.ArucoDirection | None:
        xa, ya = gridbot.node_to_coords(node_a)
        xb, yb = gridbot.node_to_coords(node_b)

        dx, dy = xb - xa, yb - ya

        dx = max(-1, min(1, dx))
        dy = max(-1, min(1, dy))

        match (dx, dy):
            case (0, 1):
                return gridbot.ArucoDirection.NORTH
            case (1, 0):
                return gridbot.ArucoDirection.EAST
            case (0, -1):
                return gridbot.ArucoDirection.SOUTH
            case (-1, 0):
                return gridbot.ArucoDirection.WEST
            case _:
                self.get_logger().error(
                    f"Nodes {node_a} and {node_b} are not adjacent."
                )
                return None

    def available_directions(self, node: str) -> set[int]:
        return {
            direction.value
            for neighbour in self.graph.get(node, {})
            if (direction := self._get_direction_between(node, neighbour)) is not None
        }

    def a_star(self, start_node: str, goal_node: str):
        if start_node not in self.graph or goal_node not in self.graph:
            self.get_logger().error(
                f"Point {start_node} or {goal_node} is out of bounds."
            )
            return []

        open_set = []
        heapq.heappush(open_set, (0, start_node))

        came_from = {}
        g_score = defaultdict(lambda: float("inf"))
        g_score[start_node] = 0
        f_score = defaultdict(lambda: float("inf"))
        f_score[start_node] = self.manhattan(start_node, goal_node)

        while open_set:
            curr_f, curr_node = heapq.heappop(open_set)

            if curr_node == goal_node:
                path = self.reconstruct_path(came_from, curr_node)

                return path

            for neighbour, weight in self.graph[curr_node].items():

                tentative_g_score = g_score[curr_node] + weight

                if tentative_g_score < g_score[neighbour]:
                    came_from[neighbour] = curr_node
                    g_score[neighbour] = tentative_g_score

                    f_score[neighbour] = tentative_g_score + self.manhattan(
                        neighbour, goal_node
                    )

                    heapq.heappush(
                        open_set,
                        (f_score[neighbour], neighbour),
                    )

        self.get_logger().error(f"No path found between {start_node} and {goal_node}.")
        return []

    def manhattan(self, node_a, node_b):
        x1, y1 = gridbot.node_to_coords(node_a)
        x2, y2 = gridbot.node_to_coords(node_b)
        return abs(x1 - x2) + abs(y1 - y2)

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


def main(args=None):

    node_name = "route_navigator"
    print(f"Hi from {node_name}.")

    rclpy.init(args=args)

    route_navigator = RouteNavigator(node_name)
    rclpy.spin(route_navigator)

    route_navigator.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
