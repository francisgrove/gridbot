from enum import Enum, IntFlag, auto


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
    Possible appearences of the line during turning of the robot.
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
