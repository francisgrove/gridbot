from functools import partial

import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64, Int8, String
from geometry_msgs.msg import Twist

import gridbot.gridbot_helpers as gridbot
from gridbot_interfaces.srv import GenerateRoute, GetDims
from gridbot_interfaces.msg import RouteNode, SimpleAruco


class PID:

    def __init__(self, k_p: float, k_i: float, k_d: float):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        self.integral = 0.0
        self.prev_error = None
        self.prev_t = None

    def update(self, error, t):
        out = self.k_p * error

        if self.prev_t is not None and t > self.prev_t:
            dt = t - self.prev_t
            self.integral += error * dt
            out += self.k_i * self.integ_error + self.k_d * (error - self.prev_error)

        self.prev_error = error
        self.prev_t = t
        return out


class RouteNavigator(Node):

    line_offset_topic: str
    aruco_topic: str
    twist_topic: str
    line_turning_topic: str
    user_cmd_topic: str

    curr_state: gridbot.RobotState
    curr_pos: str
    curr_dir: gridbot.ArucoDirection

    robot_route: list[RouteNode]

    linear_vel: float
    angular_vel: float

    pause_start = 0

    graph_width: int = None
    graph_height: int = None

    node_reached: bool
    last_line_observation: gridbot.LineObservation
    center_found: bool
    clear_route: bool

    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.setup_parameters()

        self.robot_route = []

        self.pid_output = 0.0

        self.curr_state = gridbot.RobotState.IDLE

        self.last_line_observation = None
        self.clear_route = False
        self.node_reached = False
        self.center_found = False

        self._dims_pending = False
        self._dims_timer = self.create_timer(1.0, self.request_dims)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
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
            msg_type=Twist, topic=self.twist_topic, qos_profile=qos
        )

        self.timer = self.create_timer(
            timer_period_sec=self.timer_period, callback=self.timer_callback
        )

        self.route_client = self.create_client(GenerateRoute, self.route_service)
        self.dims_client = self.create_client(GetDims, self.dims_service)

        self.context.on_shutdown(self.cleanup)


# =================== DIMS SERVICE ===================
    def request_dims(self):
        if self.graph_width is not None or self.graph_height is not None:
            return

        if not self.dims_client.service_is_ready():
            self.get_logger().info("GetDims service not available.")
            return

        self._dims_pending = True
        future = self.dims_client.call_async(GetDims.Request())
        future.add_done_callback(self.on_dims_response)
 
    def on_dims_response(self, future):
        self._dims_pending = False
        try:
            result = future.result()

        except Exception as e:
            self.get_logger().error(f"GetDims service call failed: {e}")
            return

        self.graph_width = result.graph_width
        self.graph_height = result.graph_height
        self.get_logger().info(
            f"Map dimensions are: {self.graph_width} x {self.graph_height}"
        )

# =================== ROUTE SERVICE ===================
    def request_route(self, nodes: list[str], looking_dir: gridbot.ArucoDirection):

        if not self.route_client.service_is_ready():
            self.get_logger().info("GenerateRoute service not available.")
            return

        self.route_request = GenerateRoute.Request()
        self.route_request.user_nodes = nodes
        self.route_request.looking_dir = looking_dir.value
        future = self.route_client.call_async(self.route_request)

        future.add_done_callback(self.on_route_response)

    def on_route_response(self, future):
        try:
            result = future.result()

            self.robot_route = result.route

        except Exception as e:
            self.get_logger().error(f"GenerateRoute service call failed: {e}")
            return

# =================== CALLBACKS ===================
    def user_cmd_callback(self, msg: String):

        self.get_logger().info(f"Received user command:\n{msg.data}")

        if msg.data == "STOP":
            self.get_logger().info(f"Stopping robot at first available pause.")
            self.clear_route = True
            return

        nodes_from_msg = [self.curr_pos] + [node for node in msg.data.split()]

        self.request_route(nodes=nodes_from_msg, looking_dir=self.curr_dir)

    def line_offset_callback(self, msg: Float64) -> None:
        err = float(msg.data())
        t = self.get_clock.now().nanoseconds

        self.pid_output = self.pid.update(error=err, t=t)

    def aruco_callback(self, msg: SimpleAruco):
        if self.curr_state != gridbot.RobotState.MOVING:
            return

        msg_id = int(msg.id)
        msg_dir = int(msg.direction)

        received_node = gridbot.aruco_to_node(msg_id, self.graph_height)
        self.get_logger().info(
            f"Received ArUco data:\n\tTag: {msg_id} -> {received_node}\n\tDirection: {msg_dir} -> {gridbot.ArucoDirection(msg_dir).name.capitalize()}"
        )

        expected_node = self.robot_route[0].name

        if received_node != expected_node:

            self.get_logger().info(f"Expected {expected_node}, but got {received_node}")

            # obtain all nodes that have priority (were specified by the user)
            priority_nodes = [received_node] + [
                node.name for node in self.robot_route if node.priority == 1
            ]

            self.request_route(nodes=priority_nodes, looking_dir=self.curr_dir)

        node_name = gridbot.aruco_to_node(msg_id, self.graph_height)
        dir_enum = gridbot.ArucoDirection(msg_dir)

        if self.curr_pos != node_name:
            self.curr_pos = node_name
        if self.curr_dir != dir_enum:
            self.curr_dir = dir_enum

        self.node_reached = True
        return

    def line_turning_callback(self, msg: Int8) -> None:

        if self.curr_state != gridbot.RobotState.TURNING:
            return

        flags = gridbot.LineObservation(msg.data)

        self.get_logger().info(
            f"Road flags:\n [{int(bool(flags & gridbot.LineObservation.LEFT))} "
            f"{int(bool(flags & gridbot.LineObservation.CENTER))} "
            f"{int(bool(flags & gridbot.LineObservation.RIGHT))}]"
        )

        if self.last_line_observation is None:
            self.last_line_observation = flags
            return
        elif self.last_line_observation == flags:
            return

        curr_center = flags & gridbot.LineObservation.CENTER

        if (
            not curr_center
            and self.last_line_observation & gridbot.LineObservation.CENTER
        ):
            self.center_found = False

        # Only count a line being found if previously the line wasn't in the center
        if (
            curr_center
            and not self.last_line_observation & gridbot.LineObservation.CENTER
        ):
            self.center_found = True

        self.last_line_observation = flags

# =================== PUBLISHER CALLBACK ===================

    def timer_callback(self):
        twist_msg = Twist()

        if not self.robot_route:
            return

        next_state = None
        next_node = None
        next_dir = None

        if self.curr_state != gridbot.RobotState.PAUSED:

            next_node = self.robot_route[0] if len(self.robot_route) > 0 else None

            next_dir = (
                gridbot.RobotDirection(next_node.directions[0])
                if next_node is not None and len(next_node.directions) > 0
                else None
            )

            if next_node is None or next_dir is None:
                return

            match next_dir:
                case gridbot.RobotDirection.MOVE:
                    next_state = gridbot.RobotState.MOVING
                case (
                    gridbot.RobotDirection.TURN_LEFT | gridbot.RobotDirection.TURN_RIGHT
                ):
                    next_state = gridbot.RobotState.TURNING
                case _:
                    self.get_logger().error(f"Direction {next_dir} not supported.")

        if next_state is not None and self.curr_state != next_state:
            self.curr_state = next_state
            self.get_logger().info(
                f"Robot state set to: {self.curr_state.name.capitalize()}"
            )

        match self.curr_state:
            case gridbot.RobotState.MOVING:

                twist_msg.linear.x = float(self.linear_vel)
                twist_msg.angular.z = float(self.angular_vel * self.pid_output)

                if self.node_reached:
                    self.get_logger().info(f"Node {self.curr_pos} reached.")

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

                twist_msg.angular.z = float(self.angular_vel * turn_dir)

                if self.center_found:
                    self.get_logger().info(f"Line centered")
                    self.center_found = False
                    self.curr_state = gridbot.RobotState.PAUSED
                    self.pause_start = self.get_clock().now().nanoseconds

            case gridbot.RobotState.PAUSED:
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0

                time_passed = self.get_clock().now().nanoseconds - self.pause_start
                if time_passed > self.pause_start * 1e9:
                    self.get_logger().info(f"Unpaused after {time_passed/1e9} seconds.")

                    if self.clear_route:
                        self.get_logger().info("Cleared route.")
                        self.robot_route.clear()
                        self.clear_route = False

                    if (
                        len(self.robot_route) > 0
                        and len(self.robot_route[0].directions) > 1
                    ):
                        self.robot_route[0].directions.pop(0)
                    elif len(self.robot_route) > 0:
                        self.robot_route.pop(0)

                    if len(self.robot_route) > 0:
                        self.get_logger().info(
                            f"Next route plan: {self.robot_route[0].directions[0].name} for node {self.robot_route[0].name}"
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

# =================== PARAM SETUP ===================
    def setup_parameters(self):
        p = partial(gridbot.declare_param, self)

        # Topics / services
        self.line_offset_topic = p("line_offset_topic", "line_offset", "Topic used to receive line_offset information.")
        self.aruco_topic = p("aruco_topic", "aruco", "Topic used to receive aruco tag information (ID + relative direction of the robot to the tag's top).")
        self.line_turning_topic = p("line_turning_topic", "line_turning", "Topic used to receive information about line visibility in each third of the camera.")
        self.user_cmd_topic = p("user_cmd_topic", "user_cmd", "Topic used to receive route nodes from the user.")
        self.twist_topic = p("twist_topic", "cmd_vel", "Topic used to send twist messages.")
        self.route_service = p("route_service", "generate_route", "Service used to generate robot's grid route.")
        self.dims_service = p("dims_service", "get_dims", "Service used to obtain robot's grid dimensions.")

        # Start state
        self.curr_pos = p("start_pos", "A1", "Default starting position of the robot.")
        self.curr_dir = gridbot.ArucoDirection[
            str(p("start_dir", "north", "Default starting rotation of the robot.")).upper()
        ]

        # Timing
        self.timer_period = p("timer_period", 0.005, "Timer period of the robot's state machine.", min_value=0.0)
        self.pause_duration = p("pause_duration", 1.0, "Duration (in seconds) to pause the robot for after reaching a node/turning correctly.", min_value=0.0)
        self.lost_timeout = p("lost_timeout", 5.0, "Time (in seconds) after the robot will be considered lost (counting starts when line turning retrieves [0 0 0].", min_value=0.0)

        # PID
        self.pid = PID(
            p("k_p", 1.0, "Factor for the proportional part of the PID controller.", min_value=0.0),
            p("k_i", 0.0, "Factor for the integral part of the PID controller.", min_value=0.0),
            p("k_d", 0.0, "Factor for the derivative part of the PID controller.", min_value=0.0),
        )

        # Velocities
        self.linear_vel = p("linear_vel", 0.15, "Normalized forward speed (0..1).", min_value=0.0, max_value=1.0)
        self.angular_vel = p("angular_vel", 0.3, "Normalized turning speed (0..1).", min_value=0.0, max_value=1.0)

        gridbot.log_params(self)

    def cleanup(self):
        self.twist_pub.publish(Twist())


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
