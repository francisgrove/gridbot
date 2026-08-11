from enum import Enum, IntFlag, auto

"""
States of the robot during navigation
"""


class RobotState(Enum):
    PAUSED = 0
    MOVING = 1
    TURNING = 2
    IDLE = 3


"""
Directions A* will produce.
"""


class RobotDirection(Enum):
    MOVE = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2


""""
Possible appearences of the line during turning of the robot. 
"""


class LineObservation(IntFlag):
    NONE = 0
    CENTER = auto()
    LEFT = auto()
    RIGHT = auto()


"""
Directions of the ArUco tags relative to the robot's camera
"""


class ArucoDirection(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


def aruco_to_node(id: int, graph_height: int) -> str:
    """
    Convert ArUco tag id to specific label
    """
    col_idx = id // graph_height
    row_idx = id % graph_height

    col = chr(65 + col_idx)
    row = row_idx + 1

    print(f"From id={id}->\t{col}{row}")

    return f"{col}{row}"


def node_to_aruco(node: str, graph_height: int) -> int:
    """
    Convert ArUco tag id to specific label
    """
    col = ord(node[0]) - 65
    row = int(node[1:]) - 1
    return col * graph_height + row


def node_to_coords(node: str) -> tuple:
    x = ord(node[0]) - 65
    y = int(node[1:]) - 1

    return (x, y)


def coords_to_node(coords: tuple) -> str:
    col = chr(coords[0] + 65)
    row = str(coords[1] + 1)

    return f"{col}{row}"
