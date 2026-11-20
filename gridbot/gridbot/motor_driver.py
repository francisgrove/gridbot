from functools import partial

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import rclpy.parameter
from rclpy.parameter_event_handler import ParameterEventHandler

from gpiozero import Motor

import gridbot.gridbot_helpers as gridbot


class MotorDriver(Node):

    twist_topic: str

    left_enable_pin: int
    left_forward_pin: int
    left_backward_pin: int

    right_enable_pin: int
    right_forward_pin: int
    right_backward_pin: int

    frequency: float
    invert_direction: bool
    cmd_timeout: float

    left_wheel: Motor
    right_wheel: Motor

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

        self.right_wheel = Motor(
            enable=self.right_enable_pin,
            forward=self.right_forward_pin,
            backward=self.right_backward_pin,
            pwm=True,
        )

        for motor in (self.left_wheel, self.right_wheel):
            motor.forward_device.frequency = self.frequency
            motor.backward_device.frequency = self.frequency

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

        self.context.on_shutdown(self.cleanup)

    def vel_multiplier_callback(self, p: rclpy.parameter.Parameter) -> None:
        self.get_logger().info(
            f"Received an update to parameter: {p.name}: {rclpy.parameter.parameter_value_to_python(p.value)}"
        )

        self.vel_multiplier = rclpy.parameter.parameter_value_to_python(p.value)  # type: ignore

    def cleanup(self):
        self.left_wheel.value = 0
        self.right_wheel.value = 0

    def twist_callback(self, msg: Twist):

        self._last_cmd_time = self.get_clock().now()

        direction = -1.0 if self.invert_direction else 1.0

        linear_vel = direction * msg.linear.x
        angular_vel = direction * msg.angular.z

        # left_wheel_vel = (linear_vel - angular_vel * self.wheel_separation / 2.0) / self.wheel_radius
        # right_wheel_vel = (linear_vel + angular_vel * self.wheel_separation / 2.0) / self.wheel_radius
        left_wheel_vel = linear_vel - angular_vel
        right_wheel_vel = linear_vel + angular_vel

        left_val = max(
            -1.0, min(self.vel_multiplier * self.left_wheel_mult * left_wheel_vel, 1.0)
        )
        right_val = max(
            -1.0,
            min(self.vel_multiplier * self.right_wheel_mult * right_wheel_vel, 1.0),
        )

        left_val = self._apply_deadband(left_val)
        right_val = self._apply_deadband(right_val)

        self.get_logger().info(
            f"Wheel values - Left: {left_val:.4f}, Right: {right_val:.4f}"
        )

        self.left_wheel.value = left_val
        self.right_wheel.value = right_val

    def _apply_deadband(self, val):
        if val == 0.0:
            return 0.0
        sign = 1.0 if val > 0 else -1.0
        return sign * max(self.min_pwm, abs(val))

    def _setup_parameters(self):
        p = partial(gridbot.declare_param, self)

        self.twist_topic = p(
            "twist_topic",
            "cmd_vel",
            "Name of the subscriber topic from which the node will take the Twist messages.",
        )

        # Pins
        self.left_enable_pin = p(
            "left_enable_pin",
            6,
            "Integer value representing the pin used for enabling the left wheel.",
        )
        self.right_enable_pin = p(
            "right_enable_pin",
            26,
            "Integer value representing the pin used for enabling the right wheel.",
        )
        self.left_forward_pin = p(
            "left_forward_pin",
            12,
            "Integer value representing the pin used for controlling the forward motion of the left wheel.",
        )
        self.left_backward_pin = p(
            "left_backward_pin",
            13,
            "Integer value representing the pin used for controlling the backward motion of the left wheel.",
        )
        self.right_forward_pin = p(
            "right_forward_pin",
            20,
            "Integer value representing the pin used for controlling the forward motion of the right wheel.",
        )
        self.right_backward_pin = p(
            "right_backward_pin",
            21,
            "Integer value representing the pin used for controlling the backward motion of the right wheel.",
        )

        # Geometry
        self.wheel_radius = p("wheel_radius", 0.021, "Radius of the wheels in meters.")
        self.wheel_separation = p(
            "wheel_separation",
            0.08382,
            "Distance between the centres of the two wheels in meters.",
        )

        # Output
        self.vel_multiplier = p(
            "vel_multiplier",
            1.0,
            "Multiplier of the robot's velocity. 1.0 means full speed, 0.0 means nothing.",
            min_value=0.0,
            max_value=1.0,
        )
        self.left_wheel_mult = p(
            "left_wheel_mult",
            1.0,
            "Per-wheel multiplier for the left motor.",
            min_value=0.0,
        )
        self.right_wheel_mult = p(
            "right_wheel_mult",
            1.0,
            "Per-wheel multiplier for the right motor.",
            min_value=0.0,
        )
        self.min_pwm = p(
            "min_pwm",
            0.0,
            "Minimum PWM duty cycle for non-zero commands. Values below this are bumped up to overcome motor deadband.",
            min_value=0.0,
            max_value=1.0,
        )
        self.frequency = p("frequency", 100.0, "Frequency of the PWM pins.")
        self.invert_direction = p(
            "invert_direction",
            False,
            "Whether to invert the direction of robot's movements.",
        )

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
