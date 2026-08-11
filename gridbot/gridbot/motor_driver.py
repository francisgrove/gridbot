import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from rcl_interfaces.msg import ParameterDescriptor

import numpy as np
from math import pi
from gpiozero import Motor


"""
This node utilises lgpio to directly communicate with the motor drivers, as opposed to using a ROS2 Control with a hardware interface. The reasoning for that is because Alphabot2 does not have wheel encoders, making odometry feedback unobtainable reliably. In the future, it's planned to port this code onto Alphabot1, which does have such encoders. In such case, this node is going to likely be retired from use, as ROS2 control needs its own C++ code structure.
"""


class MotorDriver(Node):

    twist_topic = None

    left_enable_pin = 0
    left_forward_pin = 0
    left_backward_pin = 0

    right_enable_pin = 0
    right_forward_pin = 0
    right_backward_pin = 0

    wheel_radius = 0.0
    wheel_separation = 0.0

    max_rpm = 0
    frequency = 0.0

    max_linear_vel = 0.0
    max_angular_vel = 0.0

    left_wheel = None
    right_wheel = None

    max_rad = None

    def __init__(self):
        super().__init__("motor_driver")

        self._setup_parameters()
        
        self.left_wheel = Motor(
            enable=self.left_enable_pin,
            forward=self.left_forward_pin,
            backward=self.left_backward_pin,
            pwm=True,
        )
        self.left_wheel.pin_factory.pwm_default_freq = self.frequency

        self.right_wheel = Motor(
            enable=self.right_enable_pin,
            forward=self.right_forward_pin,
            backward=self.right_backward_pin,
            pwm=True,
        )
        self.right_wheel.pin_factory.pwm_default_freq = self.frequency

        self.max_rad_s = (self.max_rpm / 60) * 2 * pi

        # setup sub
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

    def twist_callback(self, msg: Twist):
        # twist messages are always
        # linear: X
        # angular: Z
        # the node that

        linear_vel = np.clip(msg.linear.x, 0, self.max_linear_vel)
        angular_vel = np.clip(msg.angular.z, 0, self.max_angular_vel)

        # rad/s
        left_wheel_vel = (
            linear_vel - (angular_vel * self.wheel_separation / 2.0)
        ) / self.wheel_radius
        right_wheel_vel = (
            linear_vel + (angular_vel * self.wheel_separation / 2.0)
        ) / self.wheel_radius

        # normalize
        left_wheel_vel /= self.max_rad_s
        right_wheel_vel /= self.max_rad_s

        self.left_wheel.value = left_wheel_vel
        self.right_wheel.value = right_wheel_vel

    def cleanup(self):
        self.left_wheel.value = 0
        self.right_wheel.value = 0

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
            50,
            descriptor=ParameterDescriptor(description="Frequency of the PWM pins."),
        )


        self.twist_topic = (
            self.get_parameter("twist_topic").get_parameter_value().string_value
        )

        self.left_enable_pin = (
            self.get_parameter("left_enable_pin").get_parameter_value().integer_value
        )

        self.left_forward_pin = (
            self.get_parameter("left_forward_pin").get_parameter_value().integer_value
        )

        self.left_backward_pin = (
            self.get_parameter("left_backward_pin").get_parameter_value().integer_value
        )

        self.right_enable_pin = (
            self.get_parameter("right_enable_pin").get_parameter_value().integer_value
        )

        self.right_forward_pin = (
            self.get_parameter("right_forward_pin").get_parameter_value().integer_value
        )

        self.right_backward_pin = (
            self.get_parameter("right_backward_pin").get_parameter_value().integer_value
        )

        self.wheel_radius = (
            self.get_parameter("wheel_radius").get_parameter_value().double_value
        )

        self.wheel_separation = (
            self.get_parameter("wheel_separation").get_parameter_value().double_value
        )

        self.max_linear_vel = (
            self.get_parameter("max_linear_vel").get_parameter_value().double_value
        )

        self.max_angular_vel = (
            self.get_parameter("max_angular_vel").get_parameter_value().double_value
        )

        self.max_rpm = self.get_parameter("max_rpm").get_parameter_value().integer_value

        self.frequency = (
            self.get_parameter("frequency").get_parameter_value().double_value
        )


        param_names = self.list_parameters([], depth=10).names
        params = self.get_parameters(param_names)

        self.get_logger().info("=" * 40)
        for param in params:
            self.get_logger().info(f"{param.name}: {param.value}")
        self.get_logger().info("=" * 40)



def main(args=None):
    print("Hi from motor_driver.")
    rclpy.init(args=args)

    motor_driver = MotorDriver()
    rclpy.spin(motor_driver)
    motor_driver.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
