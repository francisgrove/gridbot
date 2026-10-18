from functools import partial

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rcl_interfaces.srv import GetParameters
from std_msgs.msg import Float64, String
from builtin_interfaces.msg import Duration

import gridbot.gridbot_helpers as gridbot
from gridbot_interfaces.msg import SimpleAruco, RouteRunMetadata, RouteStepResult
from gridbot_interfaces.srv import GenerateRoute, GetDims


class TestEvaluator(Node):

    def __init__(self, node_name: str):
        super().__init__(node_name)

        self._setup_parameters()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        observer_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.aruco_sub = self.create_subscription(
            SimpleAruco, self.aruco_topic, self.aruco_callback, observer_qos
        )

        self.line_offset_sub = self.create_subscription(
            Float64, self.line_offset_topic, self.line_offset_callback, observer_qos
        )

        self.user_cmd_pub = self.create_publisher(String, self.user_cmd_topic, qos)
        self.metadata_pub = self.create_publisher(RouteRunMetadata, self.metadata_topic, qos)
        self.step_result_pub = self.create_publisher(RouteStepResult, self.step_result_topic, qos)

        self.deviations: list[float] = []
        self.step_start_time = None
        self.current_step = 0
        self.current_cmd_idx = 0
        self.expected_route: list[str] = []
        self.total_steps = 0
        self.route_active = False
        self.run_complete = False

        self.param_clients = {
            "route_navigator": self.create_client(
                GetParameters, f"{self.get_namespace()}/route_navigator/get_parameters"
            ),
            "grid_processor": self.create_client(
                GetParameters, f"{self.get_namespace()}/grid_processor/get_parameters"
            ),
            "route_generator": self.create_client(
                GetParameters, f"{self.get_namespace()}/route_generator/get_parameters"
            ),
        }

        self.route_client = self.create_client(
            GenerateRoute, f"{self.get_namespace()}/route_generator/generate_route"
        )
        self.dims_client = self.create_client(
            GetDims, f"{self.get_namespace()}/route_generator/get_dims"
        )

        self.graph_height = None
        self.graph_width = None
        self.dims_ready = False
        self._dims_pending = False

        self.collected_params = {}
        self.params_ready = False
        self.metadata_published = False
        self.awaiting_route = False

        self.create_timer(2.0, self.startup_tick)

    def startup_tick(self):
        if self.run_complete:
            return

        if not self.dims_ready:
            self.fetch_dims()
            return

        if not self.params_ready:
            self.fetch_remote_params()
            return

        if not self.metadata_published:
            self.publish_metadata()
            self.metadata_published = True

        if not self.route_active and not self.awaiting_route and self.current_cmd_idx < len(self.user_commands):
            self.request_route()

    def fetch_dims(self):
        if self._dims_pending:
            return

        if not self.dims_client.service_is_ready():
            self.get_logger().info("Waiting for GetDims service...")
            return

        self._dims_pending = True
        future = self.dims_client.call_async(GetDims.Request())
        future.add_done_callback(self.on_dims_response)

    def on_dims_response(self, future):
        self._dims_pending = False
        try:
            result = future.result()
            self.graph_width = result.graph_width
            self.graph_height = result.graph_height
            self.dims_ready = True
            self.get_logger().info(
                f"Graph dimensions: {self.graph_width} x {self.graph_height}"
            )
        except Exception as e:
            self.get_logger().error(f"GetDims call failed: {e}")

    def fetch_remote_params(self):
        params_to_fetch = {
            "route_navigator": [
                "linear_vel", "angular_vel", "k_p", "k_i", "k_d", "start_dir",
            ],
            "grid_processor": [
                "centerline_node_offset", "centerline_step", "line_roi_height",
            ],
            "route_generator": [
                "turn_penalty_factor",
            ],
        }

        for node_name, param_names in params_to_fetch.items():
            client = self.param_clients[node_name]
            if not client.service_is_ready():
                self.get_logger().info(f"Waiting for {node_name} parameter service...")
                return

            if node_name in self.collected_params:
                continue

            req = GetParameters.Request()
            req.names = param_names
            future = client.call_async(req)
            future.add_done_callback(
                lambda f, nn=node_name, pn=param_names: self.on_params_response(f, nn, pn)
            )

        if all(n in self.collected_params for n in params_to_fetch):
            self.params_ready = True

    def on_params_response(self, future, node_name, param_names):
        try:
            result = future.result()
            params = {}
            for name, val in zip(param_names, result.values):
                if val.type == 2:  # INTEGER
                    params[name] = val.integer_value
                elif val.type == 3:  # DOUBLE
                    params[name] = val.double_value
                elif val.type == 4:  # STRING
                    params[name] = val.string_value
            self.collected_params[node_name] = params
            self.get_logger().info(f"Fetched params from {node_name}: {params}")
        except Exception as e:
            self.get_logger().error(f"Failed to fetch params from {node_name}: {e}")

    def publish_metadata(self):
        nav = self.collected_params.get("route_navigator", {})
        gp = self.collected_params.get("grid_processor", {})
        rg = self.collected_params.get("route_generator", {})

        msg = RouteRunMetadata()
        msg.run_id = self.run_id
        msg.route_name = self.route_name
        msg.route_nodes = self.user_commands
        msg.route_difficulty = self.route_difficulty

        msg.linear_vel = float(nav.get("linear_vel", 0.0))
        msg.angular_vel = float(nav.get("angular_vel", 0.0))
        msg.centerline_node_offset = int(gp.get("centerline_node_offset", 0))
        msg.centerline_step = int(gp.get("centerline_step", 0))
        msg.line_roi_height = float(gp.get("line_roi_height", 0.0))
        msg.turn_penalty_factor = float(rg.get("turn_penalty_factor", 0.0))
        msg.k_p = float(nav.get("k_p", 0.0))
        msg.k_i = float(nav.get("k_i", 0.0))
        msg.k_d = float(nav.get("k_d", 0.0))

        self.metadata_pub.publish(msg)
        self.get_logger().info(f"Published run metadata for run_id={self.run_id}")

    def request_route(self):
        cmd = self.user_commands[self.current_cmd_idx]
        nodes = [self.curr_pos] + [node for node in cmd.split()]

        if not self.route_client.service_is_ready():
            self.get_logger().info("GenerateRoute service not available.")
            return

        self.awaiting_route = True

        self.route_request = GenerateRoute.Request()
        self.route_request.user_nodes = nodes
        self.route_request.looking_dir = self.curr_dir.value

        self.get_logger().info(
            f"Requesting route {self.current_cmd_idx + 1}/{len(self.user_commands)}: {cmd}"
        )

        future = self.route_client.call_async(self.route_request)
        future.add_done_callback(self.on_route_response)

    def on_route_response(self, future):
        self.awaiting_route = False
        try:
            result = future.result()
            self.expected_route = [node.name for node in result.route]
            self.total_steps = len(self.expected_route)

            self.get_logger().info(
                f"Route received: {' -> '.join(self.expected_route)} "
                f"({len(self.expected_route)} steps)"
            )

            self.start_route()
        except Exception as e:
            self.get_logger().error(f"GenerateRoute call failed: {e}")

    def start_route(self):
        cmd = self.user_commands[self.current_cmd_idx]

        self.current_step = 0
        self.deviations.clear()
        self.route_active = True
        self.step_start_time = self.get_clock().now()

        msg = String()
        msg.data = cmd
        self.user_cmd_pub.publish(msg)

    def aruco_callback(self, msg: SimpleAruco):
        if not self.route_active or not self.expected_route:
            return

        received_node = gridbot.aruco_to_node(msg.id, self.graph_height)
        expected_node = self.expected_route[0]

        now = self.get_clock().now()
        step_dur = now - self.step_start_time
        dur_msg = Duration()
        dur_msg.sec = int(step_dur.nanoseconds // 1e9)
        dur_msg.nanosec = int(step_dur.nanoseconds % 1e9)

        result = RouteStepResult()
        result.run_id = self.run_id
        result.step_number = self.current_step
        result.total_steps = self.total_steps
        result.target_marker = SimpleAruco(
            id=gridbot.node_to_aruco(expected_node, self.graph_height),
            direction=msg.direction,
        )
        result.actual_marker = msg
        result.step_duration = dur_msg

        if self.deviations:
            result.mean_deviation_px = sum(self.deviations) / len(self.deviations)
            result.min_deviation_px = min(self.deviations)
            result.max_deviation_px = max(self.deviations)

        if received_node == expected_node:
            is_last = len(self.expected_route) == 1
            result.status = (
                RouteStepResult.STATUS_COMPLETED if is_last
                else RouteStepResult.STATUS_CORRECT
            )
            self.expected_route.pop(0)
            self.current_step += 1
        else:
            result.status = RouteStepResult.STATUS_WRONG_FIELD

        self.step_result_pub.publish(result)
        self.get_logger().info(
            f"Step {result.step_number}: status={result.status} "
            f"expected={expected_node} actual={received_node}"
        )

        self.deviations.clear()
        self.step_start_time = self.get_clock().now()

        if result.status == RouteStepResult.STATUS_COMPLETED:
            self.route_finished()

    def line_offset_callback(self, msg: Float64):
        if self.route_active:
            self.deviations.append(abs(msg.data))

    def check_lost_timeout(self):
        if not self.route_active or self.step_start_time is None:
            return

        now = self.get_clock().now()
        step_elapsed = (now - self.step_start_time).nanoseconds / 1e9
        if step_elapsed <= self.lost_timeout:
            return

        self.get_logger().warn(
            f"Lost timeout ({self.lost_timeout}s) exceeded at step {self.current_step}"
        )
        self.publish_lost_result()

    def publish_lost_result(self):
        now = self.get_clock().now()
        step_dur = now - self.step_start_time
        dur_msg = Duration()
        dur_msg.sec = int(step_dur.nanoseconds // 1e9)
        dur_msg.nanosec = int(step_dur.nanoseconds % 1e9)

        result = RouteStepResult()
        result.run_id = self.run_id
        result.step_number = self.current_step
        result.total_steps = self.total_steps
        result.status = RouteStepResult.STATUS_LOST
        result.step_duration = dur_msg

        if self.expected_route:
            expected_node = self.expected_route[0]
            result.target_marker = SimpleAruco(
                id=gridbot.node_to_aruco(expected_node, self.graph_height),
                direction=0,
            )

        if self.deviations:
            result.mean_deviation_px = sum(self.deviations) / len(self.deviations)
            result.min_deviation_px = min(self.deviations)
            result.max_deviation_px = max(self.deviations)

        self.step_result_pub.publish(result)
        self.route_finished()

    def route_finished(self):
        self.route_active = False
        self.current_cmd_idx += 1

        if self.current_cmd_idx >= len(self.user_commands):
            self.get_logger().info("All routes completed. Evaluator done. Shutting down.")
            self.run_complete = True
            self.destroy_node()
            rclpy.try_shutdown()
        else:
            self.get_logger().info(
                f"Route done. Next: {self.current_cmd_idx + 1}/{len(self.user_commands)}"
            )

    def _setup_parameters(self):
        p = partial(gridbot.declare_param, self)

        self.run_id = p("run_id", 0, "Unique ID for this test run.")
        self.route_name = p("route_name", "", "Name/label for this test run.")
        self.route_difficulty = p("route_difficulty", 0, "Route difficulty: 0=easy, 1=medium, 2=hard.")
        self.curr_pos = p("start_pos", "A1", "Starting position of the robot.")
        self.curr_dir = gridbot.ArucoDirection[
            str(p("start_dir", "north", "Starting direction of the robot.")).upper()
        ]

        self.user_commands = list(
            p("user_commands", [""], "String array of user commands to send sequentially.")
        )

        self.lost_timeout = p("lost_timeout", 60.0, "Seconds before declaring the robot lost on a step.", min_value=1.0)

        # Topics
        self.aruco_topic = p("aruco_topic", "aruco", "Topic to receive aruco detections.")
        self.line_offset_topic = p("line_offset_topic", "line_offset", "Topic to receive line offset.")
        self.user_cmd_topic = p("user_cmd_topic", "user_cmd", "Topic to send user commands to route_navigator.")
        self.metadata_topic = p("metadata_topic", "run_metadata", "Topic to publish run metadata.")
        self.step_result_topic = p("step_result_topic", "step_result", "Topic to publish step results.")

        self.create_timer(1.0, self.check_lost_timeout)

        gridbot.log_params(self)


def main(args=None):
    node_name = "test_evaluator"
    print(f"Hi from {node_name}.")
    rclpy.init(args=args)
    node = TestEvaluator(node_name)
    rclpy.spin(node)
    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
