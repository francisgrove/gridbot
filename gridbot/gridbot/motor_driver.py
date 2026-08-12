import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from rcl_interfaces.msg import (
    ParameterDescriptor,
    ListParametersResult,
    FloatingPointRange,
)
import rclpy.parameter
from rclpy.parameter_event_handler import ParameterEventHandler
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

    frequency: float

    vel_multiplier: float
    
    left_wheel: Motor
    right_wheel: Motor


    invert_direction: bool

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
            callback=self.vel_multiplier_callback, # type: ignore
        ) # pyright: ignore[reportAttributeAccessIssue]

        self.context.on_shutdown(self.cleanup)

    def vel_multiplier_callback(self, p: rclpy.parameter.Parameter) -> None:
        self.get_logger().info(f"Received an update to parameter: {p.name}: {rclpy.parameter.parameter_value_to_python(p.value)}")

        self.vel_multiplier = rclpy.parameter.parameter_value_to_python(p.value)         # type: ignore


    def cleanup(self):
        self.left_wheel.value = 0
        self.right_wheel.value = 0


    def twist_callback(self, msg: Twist):
        self.get_logger().info(
            f"Received - Linear: {msg.linear.x:.2f}, Angular: {msg.angular.z:.2f}"
        )

        direction = -1.0 if self.invert_direction else 1.0

        linear_vel = max(-1.0, min(direction*msg.linear.x, 1.0))
        angular_vel = max(-1.0, min(direction*msg.angular.z, 1.0))


        right_wheel_vel = (
            linear_vel + (angular_vel * self.wheel_separation / 2.0)
        ) / self.wheel_radius
        left_wheel_vel = (
            linear_vel - (angular_vel * self.wheel_separation / 2.0)
        ) / self.wheel_radius

        self.get_logger().info(
            f"Raw wheel values - Left: {left_wheel_vel:.4f}, Right: {right_wheel_vel:.4f}"
        )

        left_val = self.vel_multiplier*max(-1.0, min(left_wheel_vel, 1.0))
        right_val = self.vel_multiplier*max(-1.0, min(right_wheel_vel, 1.0))

        self.get_logger().info(
            f"Outputting to Pins - Left: {left_val:.4f}, Right: {right_val:.4f}"
        )

        self.left_wheel.value = left_val
        self.right_wheel.value = right_val



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
            0.021,
            descriptor=ParameterDescriptor(
                description="Radius of the wheels in meters."
            ),
        )

        self.declare_parameter(
            "wheel_separation",
            0.08382,
            descriptor=ParameterDescriptor(
                description="Distance between the centres of the two wheels in meters."
            ),
        )

        self.declare_parameter(
            "vel_multiplier",
            1.0,
            descriptor=ParameterDescriptor(
                description="Multiplier of the robot's velocity. 1.0 means full speed, 0.0 means nothing.",
                floating_point_range=[
                    FloatingPointRange(
                        from_value=0,
                        to_value=1.0,
                        )
                ],

            ),
        )


        self.declare_parameter(
            "frequency",
            50.0,
            descriptor=ParameterDescriptor(description="Frequency of the PWM pins."),
        )

        self.declare_parameter(
            "invert_direction",
            False,
            descriptor=ParameterDescriptor(
                description="Whether to invert the direction of robot's movements."
            )
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
        self.vel_multiplier = self.get_parameter("vel_multiplier").value
        self.frequency = self.get_parameter("frequency").value
        self.invert_direction = self.get_parameter("invert_direction").value

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
