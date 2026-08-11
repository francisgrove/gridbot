import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from rcl_interfaces.msg import (
    ParameterDescriptor,
    ListParametersResult,
)
import numpy as np
from math import pi
from gpiozero import Motor

"""
This node utilises lgpio to directly communicate with the motor drivers, as opposed to using a ROS2 Control with a hardware interface. The reasoning for that is because Alphabot2 does not have wheel encoders, making odometry feedback unobtainable reliably. In the future, it's planned to port this code onto Alphabot1, which does have such encoders. In such case, this node is going to likely be retired from use, as ROS2 control needs its own C++ code structure.
"""


class MotorDriver(Node):

    twist_topic: str

    left_enable_pin: int
    left_forward_pin: int
    left_backward_pin: int

    right_enable_pin: int
    right_forward_pin: int
    right_backward_pin: int

    wheel_radius: float
    wheel_separation: float

    max_rpm: int
    frequency: float

    max_linear_vel: float
    max_angular_vel: float

    clamp_motor_min: float
    clamp_motor_max: float

    left_wheel: Motor
    right_wheel: Motor

    max_rad: int

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

        self.max_rad_s = (self.max_rpm / 60) * 2 * pi

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

        self.context.on_shutdown(self.cleanup)

    def cleanup(self):
        self.left_wheel.value = 0
        self.right_wheel.value = 0

    def twist_callback(self, msg: Twist):
        self.get_logger().info(
            f"Received - Linear: {msg.linear.x:.2f}, Angular: {msg.angular.z:.2f}"
        )

        linear_vel = np.clip(-msg.linear.x, -self.max_linear_vel, self.max_linear_vel)
        angular_vel = np.clip(
            msg.angular.z, -self.max_angular_vel, self.max_angular_vel
        )

        left_wheel_vel = (
            linear_vel + (angular_vel * self.wheel_separation / 2.0)
        ) / self.wheel_radius
        right_wheel_vel = (
            linear_vel - (angular_vel * self.wheel_separation / 2.0)
        ) / self.wheel_radius

        # normalize
        left_wheel_vel /= self.max_rad_s
        right_wheel_vel /= self.max_rad_s

        left_val = max(-1.0, min((left_wheel_vel / self.max_rad_s), 1.0))
        right_val = max(-1.0, min((right_wheel_vel / self.max_rad_s), 1.0))

        self.get_logger().info(
            f"Outputting to Pins - Left: {left_val:.4f}, Right: {right_val:.4f}"
        )

        self.left_wheel.value = self._motor_scale(left_val)
        self.right_wheel.value = self._motor_scale(right_val)

    def _motor_scale(self, wheel_val: float):
        if abs(wheel_val) < 1e-3:
            return 0.0

        sign = np.sign(wheel_val)
        x = abs(wheel_val)

        return sign * (self.clamp_motor_min + (self.clamp_motor_max) * x)

    def _setup_parameters(self):
        self.declare_parameter(
            "twist_topic",
            "cmd_vel",
            descriptor=ParameterDescriptor(
                descritpion="Name of the subscriber topic frmo which the node will take the Twist messages."
            ),
        )

        self.declare_parameter(
            "left_enable_pin",
            6,
            descriptor=ParameterDescriptor(
                descritpion="Integer value representing the pin used for enabling the left wheel."
            ),
        )

        self.declare_parameter(
            "right_enable_pin",
            26,
            descriptor=ParameterDescriptor(
                descritpion="Integer value representing the pin used for enabling the right wheel."
            ),
        )

        self.declare_parameter(
            "left_forward_pin",
            12,
            descriptor=ParameterDescriptor(
                descritpion="Integer value representing the pin used for controlling the forward motion of the left wheel."
            ),
        )

        self.declare_parameter(
            "left_backward_pin",
            13,
            descriptor=ParameterDescriptor(
                descritpion="Integer value representing the pin used for controlling the backward motion of the left wheel."
            ),
        )

        self.declare_parameter(
            "right_forward_pin",
            20,
            descriptor=ParameterDescriptor(
                descritpion="Integer value representing the pin used for controlling the forward motion of the right wheel."
            ),
        )

        self.declare_parameter(
            "right_backward_pin",
            21,
            descriptor=ParameterDescriptor(
                descritpion="Integer value representing the pin used for controlling the backward motion of the right wheel."
            ),
        )

        self.declare_parameter(
            "wheel_radius",
            2.0,
            descriptor=ParameterDescriptor(
                description="Radius of the wheels in centimeters."
            ),
        )

        self.declare_parameter(
            "wheel_separation",
            7.0,
            descriptor=ParameterDescriptor(
                description="Distance between the centres of the two wheels in centimeters."
            ),
        )

        self.declare_parameter(
            "max_linear_vel",
            1.0,
            descriptor=ParameterDescriptor(
                description="Maximum linear velcoity for the robot.", read_only=True
            ),
        )

        self.declare_parameter(
            "max_angular_vel",
            20.0,
            descriptor=ParameterDescriptor(
                description="Maximum angular velcoity for the robot.", read_only=True
            ),
        )

        self.declare_parameter(
            "max_rpm",
            600,
            descriptor=ParameterDescriptor(
                description="Maximum RPM of the wheel motors.", read_only=True
            ),
        )

        self.declare_parameter(
            "frequency",
            50.0,
            descriptor=ParameterDescriptor(description="Frequency of the PWM pins."),
        )

        self.declare_parameter(
            "clamp_motor_min",
            0.1,
            descriptor=ParameterDescriptor(
                description="Minimum value to which the wheel values will be clamped"
            ),
        )

        self.declare_parameter(
            "clamp_motor_max",
            0.1,
            descriptor=ParameterDescriptor(
                description="Maximum value to which the wheel values will be clamped"
            ),
        )

        self.twist_topic = self.get_parameter("twist_topic").value

        self.left_enable_pin = self.get_parameter("left_enable_pin").value

        self.left_forward_pin = self.get_parameter("left_forward_pin").value

        self.left_backward_pin = self.get_parameter("left_backward_pin").value

        self.right_enable_pin = self.get_parameter("right_enable_pin").value

        self.right_forward_pin = self.get_parameter("right_forward_pin").value

        self.right_backward_pin = self.get_parameter("right_backward_pin").value

        self.wheel_radius = self.get_parameter("wheel_radius").value

        self.wheel_separation = self.get_parameter("wheel_separation").value

        self.max_linear_vel = self.get_parameter("max_linear_vel").value

        self.max_angular_vel = self.get_parameter("max_angular_vel").value

        self.max_rpm = self.get_parameter("max_rpm").value

        self.frequency = self.get_parameter("frequency").value

        self.clamp_motor_min = self.get_parameter("clamp_motor_min").value
        self.clamp_motor_max = self.get_parameter("clamp_motor_max").value

        result: ListParametersResult = self.list_parameters([], depth=0)
        parameters = self.get_parameters(list(result.names))

        self.get_logger().info("=" * 40)
        for param in parameters:
            self.get_logger().info(f"{param.name}: {param.value}")
        self.get_logger().info("=" * 40)


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
