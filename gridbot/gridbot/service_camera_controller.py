import rclpy
from rclpy.node import Node
import time
from rcl_interfaces.msg import ParameterDescriptor
import threading

from gridbot_interfaces.srv import AngleCamera
from gpiozero import Button

from adafruit_servokit import ServoKit


class PanTiltController:

    button: Button

    def __init__(self, pulse_min, pulse_max, button_pin, pan_angle=90, tilt_angle=90):

        # PCA-9685 on the board
        self.kit = ServoKit(channels=16)

        for chan in (0, 1):
            self.kit.servo[chan].set_pulse_width_range(
                min_pulse=pulse_min, max_pulse=pulse_max
            )

        self.pan_angle = pan_angle
        self.tilt_angle = tilt_angle

        self._lock = threading.Lock()

        self.button = Button(self.center_pin, bounce_time=0.05, pull_up=True)
        self.button.when_activated = self._center

    def _center(self):
        """
        Wrapper for predetermined centering angles, used via the joystick center button.
        """
        with self._lock:
            self.set_angle(self.pan_angle, self.tilt_angle)

    def set_angle(self, pan_angle, tilt_angle):
        """
        Sets servos to specified angles.
        """
        with self._lock:
            try:
                self.kit.servo[0].angle = pan_angle
                time.sleep(1.0)
                self.kit.servo[1].angle = tilt_angle
                time.sleep(1.0)
            finally:
                self.kit.servo[0].angle = None
                self.kit.servo[1].angle = None

    def close(self):
        if self.button is not None:
            self.button.close()


class CameraController(Node):

    pulse_min: int
    pulse_min: int
    addr: int
    button_pin: int

    def __init__(self, node_name):
        super().__init__(node_name=node_name)

        self._setup_parameters()

        self.service = self.create_service(
            AngleCamera, node_name + "/set_angles", self.center_camera_callback
        )


        self.controller = PanTiltController(
            self.pulse_min, self.pulse_max, self.button_pin
        )

        if self.controller.kit._pca.i2c_device.device_address != self.addr:
            self.get_logger().warn(f"PCA address != {hex(self.addr)}")

    def center_camera_callback(self, request, response):

        pan = request.pan_angle % 180
        tilt = request.tilt_angle % 180

        self.controller.set_angle(pan, tilt)
        self.get_logger().info(f"Servos set to pan: {pan}, tilt: {tilt}")
        response.success = True
        return response

    def _setup_parameters(self):

        self.declare_parameter(
            name="pulse_min",
            value=500,
            descriptor=ParameterDescriptor(
                description="Minimum pulse PWM for the WS-S90 servos."
            ),
        )
        self.declare_parameter(
            name="pulse_max",
            value=2500,
            descriptor=ParameterDescriptor(
                description="Maximum pulse PWM for the WS-S90 servos."
            ),
        )

        self.declare_parameter(
            name="addr",
            value=64,
            descriptor=ParameterDescriptor(
                description="I2C address of the PCA 9685 motor driver, in decimal (default 0x40 = 64)."
            ),
        )

        self.declare_parameter(
            "button_pin",
            value=7,
            descriptor=ParameterDescriptor(
                description="GPIO pin of the button used for centering."
            ),
        )

        self.pulse_min = self.get_parameter("pulse_min").value
        self.pulse_max = self.get_parameter("pulse_max").value

        self.addr = self.get_parameter("addr").value

        return

    def destroy_node(self):
        self.controller.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraController("camera_controller")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
