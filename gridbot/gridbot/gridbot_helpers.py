from enum import Enum, IntFlag, auto

from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, FloatingPointRange, IntegerRange


class RobotState(Enum):
    """
    States of the robot during navigation
    """

    PAUSED = 0
    MOVING = 1
    TURNING = 2
    IDLE = 3


class RobotDirection(Enum):
    """
    Directions A* will produce.
    """

    MOVE = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2


class LineObservation(IntFlag):
    """
    Possible appearances of the line during turning of the robot.
    """

    NONE = 0
    CENTER = auto()
    LEFT = auto()
    RIGHT = auto()


class ArucoDirection(Enum):
    """
    Directions of the ArUco tags relative to the robot's camera
    """

    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


def aruco_to_node(id: int, graph_height: int) -> str:
    """
    Convert ArUco tag id to a specific node label.
    """
    col_idx = id // graph_height
    row_idx = id % graph_height

    col = chr(65 + col_idx)
    row = row_idx + 1

    return f"{col}{row}"


def node_to_aruco(node: str, graph_height: int) -> int:
    """
    Convert ArUco tag id to specific a specific node label.
    """

    col = ord(node[0]) - 65
    row = int(node[1:]) - 1
    return col * graph_height + row


def node_to_coords(node: str) -> tuple:
    """
    Convert a node label to graph coordinates.
    """

    x = ord(node[0]) - 65
    y = int(node[1:]) - 1

    return (x, y)


def coords_to_node(coords: tuple) -> str:
    """
    Convert graph coordinates to a node label.
    """

    col = chr(coords[0] + 65)
    row = str(coords[1] + 1)

    return f"{col}{row}"


class RouteNode:
    """
    Describes a singular node in the robot's route.
    """

    def __init__(self, name, priority, directions):
        self.name = name
        self.priority = priority
        self.directions = directions

    def __str__(self):
        return f"{self.name}: ({[dir.name for dir in self.directions]})"

    def __repr__(self):
        return f"{self.name}: ({[dir.name for dir in self.directions]})"


def declare_param(
    node: Node,
    name: str,
    default,
    description: str = "",
    min_value=None,
    max_value=None,
):
    """
    Declare a parameter on the node and return its value.

    If min_value or max_value is given, a range is attached to the descriptor
    (FloatingPointRange for float defaults, IntegerRange for int defaults).
    The default may also be a Parameter.Type for parameters without a default value.
    """
    descriptor = ParameterDescriptor(description=description)

    if min_value is not None or max_value is not None:
        if isinstance(default, float):
            descriptor.floating_point_range = [
                FloatingPointRange(
                    from_value=float(min_value) if min_value is not None else -float("inf"),
                    to_value=float(max_value) if max_value is not None else float("inf"),
                )
            ]
        elif isinstance(default, int) and not isinstance(default, bool):
            descriptor.integer_range = [
                IntegerRange(
                    from_value=min_value if min_value is not None else -(2**63),
                    to_value=max_value if max_value is not None else 2**63 - 1,
                )
            ]
        else:
            raise TypeError(
                f"Parameter '{name}': ranges are only supported for int and float defaults."
            )

    return node.declare_parameter(name, default, descriptor).value


def log_params(node: Node):
    """
    Log the names and values of all parameters declared on the node.
    """
    names = node.list_parameters([], depth=0).names
    for parameter in node.get_parameters(list(names)):
        node.get_logger().info(f"{parameter.name}: {parameter.value}")
