import rclpy
from rclpy.node import Node
import time
from rcl_interfaces.msg import ParameterDescriptor, IntegerRange, ListParametersResult

from gridbot_interfaces.srv import CenterCamera

from adafruit_servokit import ServoKit


class CameraCenterer(Node):

    pulse_min: int
    pulse_min: int
    frequency: int
    addr: int

    def __init__(self, node_name):
        super().__init__(node_name=node_name)


        self.service = self.create_service(
            CenterCamera, node_name + "/center_camera", self.center_camera_callback
        )

        self.kit = ServoKit(channels=16)
        
        self._setup_parameters()
        
        self.kit.servo[0].set_pulse_width_range(min_pulse=self.pulse_min, max_pulse=self.pulse_max)
        self.kit.servo[1].set_pulse_width_range(min_pulse=self.pulse_min, 
        max_pulse=self.pulse_max)




    def center_camera_callback(self, request, response):

        pan = request.pan_angle
        tilt = request.tilt_angle

        self.kit.servo[0].angle=pan
        self.kit.servo[1].angle=tilt

        time.sleep(1.0)

        self.kit.servo[0].angle = None
        self.kit.servo[1].angle = None
        

        self.get_logger().info("Servos centered");
        response.success = True
        return response

    def _setup_parameters(self):
       
        self.declare_parameter(
                    name="pulse_min",
                    value=500,
                    descriptor=ParameterDescriptor(
                        description="Minimum pulse PWM for the WS-S90 servos."
                    )
                )
        self.declare_parameter(
                    name="pulse_max",
                    value=2500,
                    descriptor=ParameterDescriptor(
                        description="Maximum pulse PWM for the WS-S90 servos."
                    )
                )

        self.declare_parameter(
            name="hz",
            value=50,
            descriptor=ParameterDescriptor(
                description="Frequency of the servos."
            )
        )

        self.declare_parameter(
            name="addr",
            value=64,
            descriptor=ParameterDescriptor(
                description="I2C address of the PCA 9685 motor driver, in decimal (default 0x40 = 64)."
            )
        )

        self.pulse_min = self.get_parameter("pulse_min").value
        self.pulse_max = self.get_parameter("pulse_max").value
        self.frequency = 1/self.get_parameter("hz").value

        self.addr = self.get_parameter("addr").value
        if (self.kit._pca.i2c_device.device_address != self.addr):
            # self.kit._pca.i2c_device.device_address = self.addr
            self.get_logger().info(f"PCA address wasn't set to {hex(self.addr)}")

        return
        


def main(args=None):
    node_name = "camera_centerer"
    print(f"Hi from {node_name}")

    rclpy.init(args=args)

    camera_centerer = CameraCenterer(node_name)
    rclpy.spin(camera_centerer)

    camera_centerer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
