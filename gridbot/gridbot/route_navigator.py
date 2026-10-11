from builtins import list

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import (
    ParameterDescriptor,
    FloatingPointRange,
    ListParametersResult,
)
import rclpy.parameter
from rclpy.timer import Timer
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64, Int8, String
from geometry_msgs.msg import Twist

import gridbot.gridbot_helpers as gridbot
from gridbot_interfaces.srv import GenerateRoute, GetDims
from gridbot_interfaces.msg import RouteNode, SimpleAruco


class RouteNavigator(Node):

    line_offset_topic: str
    aruco_topic: str
    twist_topic: str
    line_turning_topic: str
    user_cmd_topic: str

    curr_state: gridbot.RobotState
    curr_pos: str
    curr_dir: gridbot.ArucoDirection

    robot_route: list[RouteNode] = []

    curr_error = 0.0
    integ_error = 0.0
    prev_error = 0.0

    k_p = 0.0
    k_i = 0.0
    k_d = 0.0
    pid_output = 0.0

    linear_vel: float
    angular_vel: float

    linear_vel_multiplier: float
    angular_vel_multiplier: float

    prev_time = 0
    pause_time = 0

    graph_width: int = None
    graph_height: int = None

    node_reached: bool
    last_line_observation: gridbot.LineObservation
    center_found: bool
    clear_route: bool

    # timer: Timer
    # handler: ParameterEventHandler

    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.prev_time = 0

        self.prev_error = 0.0
        self.curr_error = 0.0
        self.integ_error = 0.0

        self.pid_output = 0.0

        self.curr_state = gridbot.RobotState.IDLE

        self.last_line_observation = None

        self.clear_route = False
        self.node_reached = False
        self.center_found = False

        self._dims_pending = False
        self._dims_timer = self.create_timer(1.0, self.request_dims)

        self.setup_parameters()

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
            timer_period_sec=(self.frequency), callback=self.timer_callback
        )

        self.route_client = self.create_client(GenerateRoute, self.route_service)

        self.dims_client = self.create_client(GenerateRoute, self.dims_service)

        # self.handler = ParameterEventHandler(self)
        # for parameter_name in (
        #     "k_p",
        #     "k_i",
        #     "k_d",
        #     "linear_vel_multiplier",
        #     "angular_vel_multiplier",
        # ):
        #     self.handler.add_parameter_callback(
        #         parameter_name=parameter_name,
        #         node_name=node_name,
        #         callback=self.parameter_callback,
        #     )

        self.context.on_shutdown(self.cleanup)

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

    def parameter_callback(self, p: rclpy.parameter.Parameter) -> None:
        value = rclpy.parameter.parameter_value_to_python(p.value)

        self.get_logger().info(f"Received an update to parameter: {p.name}: {value}")

        setattr(self, p.name, value)

    def user_cmd_callback(self, msg: String):

        self.get_logger().info(f"Received user command:\n{msg.data}")

        if msg.data == "STOP":
            self.get_logger().info(f"Stopping robot at first available pause.")
            self.clear_route = True
            return

        nodes_from_msg = [self.curr_pos] + [node for node in msg.data.split()]

        self.request_route(nodes=nodes_from_msg, looking_dir=self.curr_dir)

    def line_offset_callback(self, msg: Float64) -> None:
        error = float(msg.data)

        now = self.get_clock().now().nanoseconds

        if self.prev_time == 0:
            self.prev_time = now
            self.prev_error = error
            return

        duration = now - self.prev_time
        dt = duration * 1e-9

        self.integ_error += error * dt
        deriv_error = (error - self.prev_error) / dt

        pid_output = (
            self.k_p * error + self.k_i * self.integ_error + self.k_d * deriv_error
        )

        if pid_output != self.pid_output:
            self.pid_output = pid_output

        self.prev_error = error
        self.prev_time = now

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
                gridbot.RobotDirection(next_node.directions[0])
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
                    self.get_logger().info(f"Node {self.curr_pos} reached.")

                    self.node_reached = False
                    self.curr_state = gridbot.RobotState.PAUSED
                    self.pause_time = self.get_clock().now().nanoseconds
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
                    self.get_logger().info(f"Line centered")
                    self.center_found = False
                    self.curr_state = gridbot.RobotState.PAUSED
                    self.pause_time = self.get_clock().now().nanoseconds

            case gridbot.RobotState.PAUSED:
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0

                time_passed = self.get_clock().now().nanoseconds - self.pause_time
                if time_passed > self.pause_time * 1e9:
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

    def setup_parameters(self):
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
            name="route_service",
            value="generate_route",
            descriptor=ParameterDescriptor(
                description="Service used to generate robot's grid route.",
            ),
        )

        self.declare_parameter(
            name="dims_service",
            value="get_dims",
            descriptor=ParameterDescriptor(
                description="Service used to obtain robot's grid dimensions.",
            ),
        )

        self.declare_parameter(
            name="start_pos",
            value="A1",
            descriptor=ParameterDescriptor(
                description="Default starting position of the robot."
            ),
        )

        self.declare_parameter(
            name="start_dir",
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

        self.route_service = self.get_parameter("route_service").value
        self.dims_service = self.get_parameter("dims_service").value

        self.curr_pos = self.get_parameter("start_pos").value

        self.curr_dir = gridbot.ArucoDirection[
            str(self.get_parameter("start_dir").value).upper()
        ]

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

        for parameter in parameters:
            self.get_logger().info(f"{parameter.name}: {parameter.value}")

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
