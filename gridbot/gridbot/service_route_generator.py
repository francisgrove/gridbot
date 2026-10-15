import math

import rclpy
from rclpy.node import Node

import yaml
from itertools import groupby
import heapq
from collections import defaultdict

import gridbot.gridbot_helpers as gridbot
from gridbot_interfaces.srv import GenerateRoute, GetDims
from gridbot_interfaces.msg import RouteNode


class RouteGenerator(Node):

    graph: dict[str, dict[str, int]]
    graph_width: int
    graph_height: int

    looking_dir: gridbot.ArucoDirection

    def __init__(self, node_name):
        super().__init__(node_name)

        graph_path = gridbot.declare_param(
            self,
            "graph_filepath",
            "",
            "Filepath of the map graph yaml file. It contains the adjacency map as well as the map's dimensions.",
        )

        self.turn_penalty_factor = gridbot.declare_param(
            self,
            "turn_penalty_factor",
            1.0,
            "Factor [0.0, 1.0] for turn penalty. 0.0 disables turn penalty.",
        )

        with open(graph_path) as f:
            try:

                load = yaml.safe_load(f)

                if load is None:
                    self.get_logger().error(
                        f"Given graph file '{graph_path}' is empty."
                    )

                self.graph = load["graph"]

                self.graph_height = gridbot.declare_param(
                    self, "graph_height", load["graph_height"], "Height of the graph."
                )
                self.graph_width = gridbot.declare_param(
                    self, "graph_width", load["graph_width"], "Width of the graph."
                )

            except yaml.YAMLError as e:
                self.get_logger().error(str(e))

        self.route_service = self.create_service(
            GenerateRoute, f"{node_name}/generate_route", self.generate_route_callback
        )

        self.dims_service = self.create_service(
            GetDims, f"{node_name}/get_dims", self.get_dims_callback
        )

        self.get_logger().info("Graph Loaded successfully")

    def get_dims_callback(self, request, response):
        response.graph_width = self.graph_width
        response.graph_height = self.graph_height

        return response

    def generate_route_callback(self, request, response):

        nodes: list[str] = request.user_nodes

        for node in nodes:
            if node not in self.graph:
                self.get_logger().warn(
                    f"Node '{node}' given as in user command not in graph."
                )
                nodes.remove(node)

        if not nodes:
            self.get_logger().error("No valid nodes provided.")
            response.route = []
            return response

        self.looking_dir = gridbot.ArucoDirection(request.looking_dir)

        route = []

        start_node = nodes[0]

        node_pairs = list(zip([start_node] + nodes[:-1], nodes))

        paths_2d = []

        for start, goal in node_pairs:
            self.get_logger().debug(f"Calculating path from {start} to {goal}.")
            path = self.a_star(start, goal)
            self.get_logger().debug(f"A* calculated path: {path}")
            paths_2d.append(path)

        paths_1d = [node for path in paths_2d for node in path]
        path = [node for node, _ in groupby(paths_1d)]

        node_pairs = list(zip(path[:-1], path[1:]))

        for start, goal in node_pairs:

            node = RouteNode(
                name=goal, directions=[], priority=1 if goal in nodes else 0
            )
            route.append(node)

        self._add_directions(start_node=path[0], route=route)
        self.get_logger().info(
            "Route is:\n" + "\n".join(str(node.name) for node in route)
        )

        response.route = route

        return response

    def _add_directions(
        self,
        start_node: str,
        route: list[gridbot.RouteNode],
    ):
        if not route:
            return

        curr_dir = self.looking_dir
        curr_node = start_node

        for next_node in route:

            target_dir: gridbot.ArucoDirection | None = self._get_direction_between(
                curr_node,
                next_node.name,
            )

            if target_dir is None:
                return

            direction_diff = (target_dir.value - curr_dir.value) % 4

            match direction_diff:
                case 0:
                    # Already facing the correct direction.
                    pass

                case 1:
                    next_node.directions.append(gridbot.RobotDirection.TURN_RIGHT.value)

                case 2:
                    # 180 degree turn.

                    left_dir = gridbot.ArucoDirection((curr_dir.value - 1) % 4)
                    right_dir = gridbot.ArucoDirection((curr_dir.value + 1) % 4)

                    neighbors = self.graph.get(curr_node, {})

                    for neighbor in neighbors:
                        direction = self._get_direction_between(curr_node, neighbor)

                        if direction == left_dir:
                            self.get_logger().debug(
                                f"{neighbor} to the left of {curr_node}"
                            )
                        if direction == right_dir:
                            self.get_logger().debug(
                                f"{neighbor} to the right of {curr_node}"
                            )

                    has_left = any(
                        self._get_direction_between(curr_node, neighbor) == left_dir
                        for neighbor in neighbors
                    )

                    has_right = any(
                        self._get_direction_between(curr_node, neighbor) == right_dir
                        for neighbor in neighbors
                    )

                    turn = None

                    if has_right and not has_left:
                        turn = gridbot.RobotDirection.TURN_LEFT.value
                    else:
                        turn = gridbot.RobotDirection.TURN_RIGHT.value

                    # turn in the neighbor-less direction (only one penalty)
                    # if both directions have neighbors, turn (right) twice
                    # if none, turn (right) once
                    next_node.directions.append(turn)
                    if has_right and has_left:
                        next_node.directions.append(turn)

                case 3:
                    next_node.directions.append(gridbot.RobotDirection.TURN_LEFT.value)

            next_node.directions.append(gridbot.RobotDirection.MOVE.value)

            curr_dir = target_dir
            curr_node = next_node.name

    def _get_distance_between(self, node_a: str, node_b: str) -> tuple[int, int]:
        xa, ya = gridbot.node_to_coords(node_a)
        xb, yb = gridbot.node_to_coords(node_b)

        dx, dy = xb - xa, yb - ya

        return (dx, dy)

    def _get_direction_between(
        self, node_a: str, node_b: str
    ) -> gridbot.ArucoDirection | None:

        dx, dy = self._get_distance_between(node_a, node_b)

        dx = max(-1, min(1, dx))
        dy = max(-1, min(1, dy))

        match (dx, dy):
            case (0, 1):
                return gridbot.ArucoDirection.NORTH
            case (1, 0):
                return gridbot.ArucoDirection.EAST
            case (0, -1):
                return gridbot.ArucoDirection.SOUTH
            case (-1, 0):
                return gridbot.ArucoDirection.WEST
            case _:
                self.get_logger().error(
                    f"Nodes {node_a} and {node_b} are not adjacent."
                )
                return None

    def available_directions(self, node: str) -> set[int]:
        return {
            direction.value
            for neighbor in self.graph.get(node, {})
            if (direction := self._get_direction_between(node, neighbor)) is not None
        }

    class GridCell:
        cost: float
        name: str
        direction: gridbot.ArucoDirection

        def __init__(self, cost, name, direction):
            self.cost = cost
            self.name = name
            self.direction = direction

        def __lt__(self, other):
            return self.cost < other.cost

        def __le__(self, other):
            return self.cost <= other.cost

        def __eq__(self, other):
            print(type(other))
            return (self.name, self.direction) == (other.name, other.direction)

        def __hash__(self):
            return hash((self.name, self.direction))

    def a_star(self, start_node: str, goal_node: str):
        if start_node not in self.graph or goal_node not in self.graph:
            self.get_logger().error(
                f"Point {start_node} or {goal_node} is out of bounds."
            )
            return []

        open_set = []
        closed_set = set()
        came_from = {}

        start_cell = self.GridCell(
            cost=self.manhattan(start_node, goal_node),
            name=start_node,
            direction=self.looking_dir,
        )
        heapq.heappush(open_set, start_cell)

        g_score = defaultdict(lambda: float("inf"))
        g_score[start_cell] = 0.0

        while open_set:
            curr_cell = heapq.heappop(open_set)

            if curr_cell in closed_set:
                continue

            closed_set.add(curr_cell)

            if curr_cell.name == goal_node:
                path = self.reconstruct_path(came_from, curr_cell)

                return path

            for n_name, n_weight in self.graph[curr_cell.name].items():

                self.get_logger().info(
                    f"[{curr_cell.name}]'s neighbor: {n_name} with weight == {n_weight}"
                )

                target_dir = self._get_direction_between(curr_cell.name, n_name)

                dir_diff = (target_dir.value - curr_cell.direction.value) % 4

                n_cell = self.GridCell(cost=0.0, name=n_name, direction=target_dir)

                tentative_g_score = g_score[curr_cell] + n_weight + dir_diff * self.turn_penalty_factor

                if tentative_g_score < g_score[n_cell]:

                    came_from[n_cell] = curr_cell

                    g_score[n_cell] = tentative_g_score

                    n_cell.cost = tentative_g_score + self.manhattan(n_name, goal_node)

                    heapq.heappush(open_set, n_cell)

        self.get_logger().info(f"No path found between {start_node} and {goal_node}.")
        return []

    def manhattan(self, node_a, node_b):
        xa, ya = gridbot.node_to_coords(node_a)
        xb, yb = gridbot.node_to_coords(node_b)
        return abs(xa - xb) + abs(ya - yb)

    def reconstruct_path(self, came_from, current):
        path = [current.name]
        while current in came_from:
            current = came_from[current]
            path.append(current.name)
        path.reverse()
        return path


def main(args=None):
    node_name = "route_generator"
    print(f"Hi from {node_name}")

    rclpy.init(args=args)

    map_generator = RouteGenerator(node_name)
    rclpy.spin(map_generator)

    map_generator.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
