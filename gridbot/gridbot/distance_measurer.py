from functools import partial

import rclpy
from rclpy.node import Node

from gpiozero import DistanceSensor
import gridbot.gridbot_helpers as gridbot

from std_msgs.msg import Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class DistanceMeasurer(Node):

    def __init__(self, node_name):
        super().__init__(node_name)

        self._setup_parameters()

        self.sensor = DistanceSensor(echo=self.echo_pin, trigger=self.trigger_pin)


        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.distance_pub = self.create_publisher(
            msg_type=Float32,
            topic=self.distance_topic,
            qos_profile = qos
        )

        self.timer = self.create_timer(0.01, self.distance_callback)

    def distance_callback(self):

        msg = Float32()
        dist = self.sensor.distance * 100
        msg.data=dist
        self.get_logger().info(f"Distance: {dist:.4f} cm.")
        self.distance_pub.publish(msg)

    def _setup_parameters(self):
        p = partial(gridbot.declare_param,self)

        self.distance_topic = p("distance_topic", "ranging", "Topic used to send SR04 distance messages to.")
        self.trigger_pin = p("trigger_pin", 22, "GPIO pin for the trigger pin of the SR04 distance sensor.")
        self.echo_pin = p("echo_pin", 27, "GPIO pin for the echo pin of the SR04 distance sensor.")

        


def main(args=None):
    node_name = "distance_measurer"
    print(f"Hi from {node_name}.")
    rclpy.init(args=args)

    distance_measurer = DistanceMeasurer(node_name)
    rclpy.spin(distance_measurer)
    distance_measurer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
