from functools import partial

import rclpy
from rclpy.node import Node
from rclpy.lifecycle import LifecycleNode
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import rclpy.parameter
from rclpy.parameter_event_handler import ParameterEventHandler

from gpiozero import Motor

import gridbot.gridbot_helpers as gridbot


class MotorDriver(LifecycleNode):

    twist_topic: str

    left_enable_pin: int
    left_forward_pin: int
    left_backward_pin: int

    right_enable_pin: int
    right_forward_pin: int
    right_backward_pin: int

    wheel_radius: float
    wheel_separation: float

    frequency: float

    vel_multiplier: float

    left_wheel: Motor
    right_wheel: Motor

    ramp_up_duration: float
    invert_direction: bool

    _target_left: float = 0.0
    _target_right: float = 0.0
    _current_left: float = 0.0
    _current_right: float = 0.0

    handler: ParameterEventHandler

    def __init__(self, node_name: str):
        super().__init__(node_name)

        self._setup_parameters()

        self.left_wheel = Motor(
            enable=self.left_enable_pin,
            forward=self.left_forward_pin,
            backward=self.left_backward_pin,
            pwm=True,
        )

        if self.left_wheel.pin_factory is None:
            raise RuntimeError("Motor has no pin factory")

        self.left_wheel.pin_factory.pwm_default_freq = self.frequency

        self.right_wheel = Motor(
            enable=self.right_enable_pin,
            forward=self.right_forward_pin,
            backward=self.right_backward_pin,
            pwm=True,
        )

        if self.right_wheel.pin_factory is None:
            raise RuntimeError("Motor has no pin factory")

        self.right_wheel.pin_factory.pwm_default_freq = self.frequency

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.twist_sub = self.create_subscription(
            msg_type=Twist,
            topic=self.twist_topic,
            callback=self.twist_callback,
            qos_profile=qos,
        )

        self.handler = ParameterEventHandler(self)

        self.handler = self.handler.add_parameter_callback(
            parameter_name="vel_multiplier",
            node_name=node_name,
            callback=self.vel_multiplier_callback,
        )

        self._last_ramp_time = self.get_clock().now().nanoseconds / 1e9
        self._ramp_timer = self.create_timer(0.02, self._ramp_callback)

        self.context.on_shutdown(self.cleanup)

    def vel_multiplier_callback(self, p: rclpy.parameter.Parameter) -> None:
        self.get_logger().info(
            f"Received an update to parameter: {p.name}: {rclpy.parameter.parameter_value_to_python(p.value)}"
        )

        self.vel_multiplier = rclpy.parameter.parameter_value_to_python(p.value)  # type: ignore

    def cleanup(self):
        self._target_left = 0.0
        self._target_right = 0.0
        self._current_left = 0.0
        self._current_right = 0.0
        self.left_wheel.value = 0
        self.right_wheel.value = 0

    def twist_callback(self, msg: Twist):
        self.get_logger().info(
            f"Received - Linear: {msg.linear.x:.2f}, Angular: {msg.angular.z:.2f}"
        )

        direction = -1.0 if self.invert_direction else 1.0

        linear_vel = max(-1.0, min(direction * msg.linear.x, 1.0))
        angular_vel = max(-1.0, min(direction * msg.angular.z, 1.0))

        right_wheel_vel = (
            linear_vel + (angular_vel * self.wheel_separation / 2.0)
        ) # / self.wheel_radius
        left_wheel_vel = (
            linear_vel - (angular_vel * self.wheel_separation / 2.0)
        ) # / self.wheel_radius

        self.get_logger().info(
            f"Raw wheel values - Left: {left_wheel_vel:.4f}, Right: {right_wheel_vel:.4f}"
        )

        left_val = self.vel_multiplier * max(-1.0, min(left_wheel_vel, 1.0))
        right_val = self.vel_multiplier * max(-1.0, min(right_wheel_vel, 1.0))

        self._target_left = left_val
        self._target_right = right_val

    def _ramp_callback(self):
        now = self.get_clock().now().nanoseconds / 1e9
        dt = now - self._last_ramp_time
        self._last_ramp_time = now

        if self.ramp_up_duration <= 0.0:
            self._current_left = self._target_left
            self._current_right = self._target_right
        else:
            max_step = dt / self.ramp_up_duration

            for attr in ("left", "right"):
                current = getattr(self, f"_current_{attr}")
                target = getattr(self, f"_target_{attr}")
                diff = target - current
                if abs(diff) <= max_step:
                    current = target
                else:
                    current += max_step if diff > 0 else -max_step
                setattr(self, f"_current_{attr}", current)

        self.left_wheel.value = self._current_left
        self.right_wheel.value = self._current_right

    def _setup_parameters(self):
        p = partial(gridbot.declare_param, self)

        self.twist_topic = p("twist_topic", "cmd_vel", "Name of the subscriber topic from which the node will take the Twist messages.")

        # Pins
        self.left_enable_pin = p("left_enable_pin", 6, "Integer value representing the pin used for enabling the left wheel.")
        self.right_enable_pin = p("right_enable_pin", 26, "Integer value representing the pin used for enabling the right wheel.")
        self.left_forward_pin = p("left_forward_pin", 12, "Integer value representing the pin used for controlling the forward motion of the left wheel.")
        self.left_backward_pin = p("left_backward_pin", 13, "Integer value representing the pin used for controlling the backward motion of the left wheel.")
        self.right_forward_pin = p("right_forward_pin", 20, "Integer value representing the pin used for controlling the forward motion of the right wheel.")
        self.right_backward_pin = p("right_backward_pin", 21, "Integer value representing the pin used for controlling the backward motion of the right wheel.")

        # Geometry
        self.wheel_radius = p("wheel_radius", 0.021, "Radius of the wheels in meters.")
        self.wheel_separation = p("wheel_separation", 0.08382, "Distance between the centres of the two wheels in meters.")

        # Output
        self.vel_multiplier = p("vel_multiplier", 1.0, "Multiplier of the robot's velocity. 1.0 means full speed, 0.0 means nothing.", min_value=0.0, max_value=1.0)
        self.frequency = p("frequency", 50.0, "Frequency of the PWM pins.")
        self.ramp_up_duration = p("ramp_up_duration", 0.5, "Time in seconds to ramp from 0 to full speed. 0 disables ramping.", min_value=0.0)
        self.invert_direction = p("invert_direction", False, "Whether to invert the direction of robot's movements.")

        gridbot.log_params(self)


def main(args=None):
    node_name: str = "motor_driver"
    print(f"Hi from {node_name}.")
    rclpy.init(args=args)

    motor_driver = MotorDriver(node_name)
    rclpy.spin(motor_driver)
    motor_driver.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
