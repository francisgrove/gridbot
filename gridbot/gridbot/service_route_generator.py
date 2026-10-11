import math

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import (
    ParameterDescriptor,
)

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

    def __init__(self, node_name):
        super().__init__(node_name)

        self.declare_parameter(
            name="graph_filepath",
            value="",
            descriptor=ParameterDescriptor(
                description="Filepath of the map graph yaml file. It contains the adjacency map as well as the map's dimensions.",
            ),
        )

        graph_path = self.get_parameter("graph_filepath").value

        with open(graph_path) as f:
            try:

                load = yaml.safe_load(f)

                if load is None:
                    self.get_logger().error(
                        f"Given graph file '{graph_path}' is empty."
                    )

                self.graph = load["graph"]

                self.declare_parameter(
                    name="graph_height",
                    value=load["graph_height"],
                    descriptor=ParameterDescriptor(description="Height of the graph."),
                )

                self.declare_parameter(
                    name="graph_width",
                    value=load["graph_width"],
                    descriptor=ParameterDescriptor(description="Width of the graph."),
                )

                self.graph_width = self.get_parameter("graph_width").value
                self.graph_height = self.get_parameter("graph_height").value

            except yaml.YAMLError as e:
                self.get_logger().error(str(e))

        self.route_service = self.create_service(
            GenerateRoute, f"{node_name}/generate_route", self.generate_route_callback
        )

        self.dims_service = self.create_service(
            GetDims, f"{node_name}/get_dims",self.get_dims_callback
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

        looking_dir = gridbot.ArucoDirection(request.looking_dir)

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

        self._add_directions(start_dir=looking_dir, start_node=path[0], route=route)
        self.get_logger().info("Route is:\n" + "\n".join(str(node) for node in route))

        response.route = route

        return response

    def _add_directions(
        self,
        start_dir: gridbot.RobotDirection,
        start_node: str,
        route: list[gridbot.RouteNode],
    ):
        if not route:
            return

        curr_dir = start_dir
        curr_node = start_node

        for next_node in route:

            target_dir: gridbot.ArucoDirection | None = self._get_direction_between(
                curr_node,
                next_node.name,
            )

            if target_dir is None:
                return

            dir_diff = (target_dir.value - curr_dir.value) % 4

            match dir_diff:
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

                    if has_right and has_left:
                        next_node.directions.append(
                            gridbot.RobotDirection.TURN_LEFT.value
                        )
                        next_node.directions.append(
                            gridbot.RobotDirection.TURN_LEFT.value
                        )
                    elif has_right and not has_left:
                        next_node.directions.append(
                            gridbot.RobotDirection.TURN_RIGHT.value
                        )
                        next_node.directions.append(
                            gridbot.RobotDirection.TURN_RIGHT.value
                        )
                    elif not has_right and has_left:
                        next_node.directions.append(
                            gridbot.RobotDirection.TURN_LEFT.value
                        )
                        next_node.directions.append(
                            gridbot.RobotDirection.TURN_LEFT.value
                        )
                    else:
                        next_node.directions.append(
                            gridbot.RobotDirection.TURN_LEFT.value
                        )

                case 3:
                    next_node.directions.append(gridbot.RobotDirection.TURN_LEFT.value)

            next_node.directions.append(gridbot.RobotDirection.MOVE.value)

            curr_dir = target_dir
            curr_node = next_node.name

    def _get_direction_between(
        self, node_a: str, node_b: str
    ) -> gridbot.ArucoDirection | None:
        xa, ya = gridbot.node_to_coords(node_a)
        xb, yb = gridbot.node_to_coords(node_b)

        dx, dy = xb - xa, yb - ya

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

    def a_star(self, start_node: str, goal_node: str):
        if start_node not in self.graph or goal_node not in self.graph:
            self.get_logger().error(
                f"Point {start_node} or {goal_node} is out of bounds."
            )
            return []

        open_set = []
        heapq.heappush(open_set, (0, start_node))

        came_from = {}
        g_score = defaultdict(lambda: float("inf"))
        g_score[start_node] = 0
        f_score = defaultdict(lambda: float("inf"))
        f_score[start_node] = self.manhattan(start_node, goal_node)
        self.get_logger().info(
            f"[start] Manhattan for {start_node} and {goal_node} == {f_score[start_node]}"
        )

        while open_set:
            curr_f, curr_node = heapq.heappop(open_set)

            if curr_node == goal_node:
                path = self.reconstruct_path(came_from, curr_node)
                return path

            for neighbor, weight in self.graph[curr_node].items():

                self.get_logger().info(
                    f"[{curr_node}]'s neighbor: {neighbor} with weight == {weight}"
                )

                tentative_g_score = g_score[curr_node] + weight

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = curr_node
                    g_score[neighbor] = tentative_g_score

                    manhattan = self.manhattan(neighbor, goal_node)

                    f_score[neighbor] = tentative_g_score + manhattan

                    self.get_logger().info(
                        f"Manhattan for {neighbor} and {goal_node} == {manhattan}"
                    )

                    self.get_logger().info(
                        f"F score for {neighbor} and {goal_node} == {f_score[neighbor]}"
                    )

                    if (f_score[neighbor], neighbor) not in open_set:
                        heapq.heappush(
                            open_set,
                            (f_score[neighbor], neighbor),
                        )
                    else:
                        self.get_logger().warn(f"{neighbor} already in open set.   ")

        self.get_logger().info(f"No path found between {start_node} and {goal_node}.")
        return []

    def manhattan(self, node_a, node_b):
        xa, ya = gridbot.node_to_coords(node_a)
        xb, yb = gridbot.node_to_coords(node_b)
        return abs(xa - xb) + abs(ya - yb)

    def euclidean(self, node_a, node_b):
        xa, ya = gridbot.node_to_coords(node_a)
        xb, yb = gridbot.node_to_coords(node_b)
        return math.sqrt((xa - xb) ** 2, (ya - yb) ** 2)

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
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
