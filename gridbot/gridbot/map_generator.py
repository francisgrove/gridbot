import rclpy
from rcl_interfaces.msg import ParameterDescriptor, IntegerRange, ListParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from ament_index_python.packages import get_package_share_directory
import os
import yaml
import gridbot.gridbot_helpers as gridbot

import cv2
import numpy as np

class _MapEdge:
    fr: str
    to: str
    weight: int

    def __init__(self, fr, to, weight):
        self.fr = fr
        self.to = to
        self.weight = weight

    def __str__(self):
        return f"({self.to}, {self.weight})"

    def __repr__(self):
        return f"{self.to}, {self.weight}"


class _MapNode:

    # coordinates of the ArUco's center
    x: int
    y: int

    tag_id: int
    label: str

    def __init__(self, x, y, tag_id, h):
        self.x = x
        self.y = y
        self.tag_id = tag_id
        self.label = gridbot.aruco_to_node(id=tag_id, graph_height=h)

    def __str__(self):
        return f"{self.label}"


class MapGenerator(Node):
    yaml_save_path: str
    img_save_path: str

    tag_size: int
    text_size: int
    img_width: int
    img_height: int

    line_color: tuple[int, int, int]
    line_length: int
    line_thickness: int

    padding: int
    text_padding: int

    graph_width: int
    graph_height: int

    aruco_dict: dict

    removed_nodes: list[str]
    edges: list[str]

    use_weights: bool

    node_map: dict[_MapNode, list[_MapEdge]] = {}

    def __init__(self, node_name):
        super().__init__(node_name)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

        self._setup_parameters()
        self._add_nodes()
        self._remove_nodes()
        self._add_edges()

        graph = self._generate_graph()

        self.get_logger().info(f"graph:\n{graph}")

        image = self._generate_image()

        success = cv2.imwrite(self.img_save_path, image)

        if success:
            self.get_logger().info(f"Saved map image to {self.img_save_path}.")
        else:
            self.get_logger().error(f"Error saving map image to {self.img_save_path}.")

        graph = self._generate_graph()

        yaml_contents = dict(
            graph_width=self.graph_width, graph_height=self.graph_height, graph=graph
        )

        with open(self.yaml_save_path, "w") as f:
            try:
                yaml.dump(yaml_contents, f, default_flow_style=False)
                self.get_logger().info(f"Saved map graph to {self.yaml_save_path}.")
            except:
                self.get_logger().error(
                    f"Error saving map graph to {self.yaml_save_path}."
                )

    def _setup_parameters(self):
        self.declare_parameter(
            name="yaml_save_path",
            value="config/",
            descriptor=ParameterDescriptor(
                description="Save path of the map yaml file. Relative to the package."
            ),
        )

        self.declare_parameter(
            name="img_save_path",
            value="assets/",
            descriptor=ParameterDescriptor(
                description="Save path of the generated map image."
            ),
        )

        self.declare_parameter(
            name="img_width",
            value=2048,
            descriptor=ParameterDescriptor(
                description="Width of generated image in pixels.",
                integer_range=[
                    IntegerRange(
                        from_value=1,
                        to_value=8192,
                        step=1,
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="img_height",
            value=2048,
            descriptor=ParameterDescriptor(
                description="Height of generated image in pixels.",
                integer_range=[
                    IntegerRange(
                        from_value=1,
                        to_value=8192,
                        step=1,
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="tag_size",
            value=30,
            descriptor=ParameterDescriptor(
                description="Size of the ArUco tag in pixels.",
                integer_range=[
                    IntegerRange(
                        from_value=1,
                        to_value=2048,
                        step=1,
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="text_size",
            value=25,
            descriptor=ParameterDescriptor(
                description="Size of edge weight text in pixels.",
                integer_range=[
                    IntegerRange(
                        from_value=0,
                        to_value=2048,
                        step=1,
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="line_color",
            value=[255, 0, 0],
            descriptor=ParameterDescriptor(description="RGB color of graph lines."),
        )

        self.declare_parameter(
            name="line_length",
            value=100,
            descriptor=ParameterDescriptor(
                description="Length of lines between nodes in pixels.",
                integer_range=[
                    IntegerRange(
                        from_value=1,
                        to_value=4096,
                        step=1,
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="padding",
            value=8,
            descriptor=ParameterDescriptor(
                description="Padding of a tag from the image border in pixels.",
                integer_range=[
                    IntegerRange(
                        from_value=0,
                        to_value=2048,
                        step=1,
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="text_padding",
            value=8,
            descriptor=ParameterDescriptor(
                description="Padding of weight text from its line in pixels.",
                integer_range=[
                    IntegerRange(
                        from_value=0,
                        to_value=2048,
                        step=1,
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="line_thickness",
            value=8,
            descriptor=ParameterDescriptor(
                description="Thickness of graph lines in pixels.",
                integer_range=[
                    IntegerRange(
                        from_value=1,
                        to_value=100,
                        step=1,
                    )
                ],
            ),
        )

        # Graph parameters

        self.declare_parameter(
            name="graph_width",
            value=6,
            descriptor=ParameterDescriptor(
                description="Number of columns in the graph.",
                integer_range=[
                    IntegerRange(
                        from_value=1,
                        to_value=26,
                        step=1,
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="graph_height",
            value=5,
            descriptor=ParameterDescriptor(
                description="Number of rows in the graph.",
                integer_range=[
                    IntegerRange(
                        from_value=1,
                        to_value=26,
                        step=1,
                    )
                ],
            ),
        )

        self.declare_parameter(
            "removed_nodes",
            Parameter.Type.STRING_ARRAY,
            ParameterDescriptor(description="List of disabled nodes in the graph."),
        )

        self.declare_parameter(
            "edges",
            Parameter.Type.STRING_ARRAY,
            ParameterDescriptor(
                description=(
                    "Graph edges formatted as FROM,TO,WEIGHT. " "Example: A1,B1,7"
                ),
            ),
        )

        self.declare_parameter(
            "use_weights",
            False,
            ParameterDescriptor(
                description=(
                    "Whether to use weights provided in edges, or to calculate them from node positions."
                )
            )
        )

        self.yaml_save_path = self.get_parameter("yaml_save_path").value
        self.img_save_path = self.get_parameter("img_save_path").value
        self.tag_size = self.get_parameter("tag_size").value
        self.text_size = self.get_parameter("text_size").value
        self.line_color = self.get_parameter("line_color").value

        self.img_height = self.get_parameter("img_height").value
        self.img_width = self.get_parameter("img_width").value

        self.graph_width = self.get_parameter("graph_width").value
        self.graph_height = self.get_parameter("graph_height").value

        line_color = self.get_parameter("line_color").value
        if len(line_color) != 3:
            self.get_logger().fatal(
                f"Paramater 'line_color' = {line_color} must have 3 elements."
            )
            for c in line_color:
                c = min(0, max(255, c))

        self.line_color = line_color

        self.line_length = self.get_parameter("line_length").value
        self.line_thickness = self.get_parameter("line_thickness").value

        self.padding = self.get_parameter("padding").value
        self.text_padding = self.get_parameter("text_padding").value

        self.removed_nodes = self.get_parameter("removed_nodes").value
        self.edges = self.get_parameter("edges").value

        self.use_weights = self.get_parameter("use_weights").value

        result: ListParametersResult = self.list_parameters([], depth=0)

        parameters = self.get_parameters(list(result.names))

        for parameter in parameters:
            self.get_logger().info(f"{parameter.name}: {parameter.value}")

    def _add_nodes(self):
        for y in range(self.graph_height):
            for x in range(self.graph_width):

                tag_id = x * self.graph_height + y

                tag_x = self.tag_size // 2 + x * (self.tag_size + self.line_length)

                tag_y = self.tag_size // 2 + y * (self.tag_size + self.line_length)

                map_node = _MapNode(
                    x=tag_x,
                    y=tag_y,
                    tag_id=tag_id,
                    h=self.graph_height,
                )

                self.get_logger().info(f"{map_node.label}: x={tag_x}, y={tag_y}")

                self.node_map[map_node] = []

    def _remove_nodes(self):
        for rm in self.removed_nodes:

            rm_node = next(
                (node for node in self.node_map if node.label == rm),
                None,
            )

            if rm_node is not None:
                self.node_map.pop(rm_node, None)

                self.get_logger().info(f"Removed node {rm} from graph.")

    def _are_linear(self, node_a: str, node_b: str) -> bool:
        """
        Checks if two nodes exist on either the same row or column.
        """

        xa, ya = gridbot.node_to_coords(node_a)
        xb, yb = gridbot.node_to_coords(node_b)

        return xa == xb or ya == yb

    def _add_edges(self):
        for edge in self.edges:
            parts = edge.split(",")

            if len(parts) != 3:
                self.get_logger().warning(f"Invalid edge definition: {edge}")
                continue

            fr = parts[0]
            to = parts[1]
            weight_str = int(parts[2])

            from_node = next(
                (node for node in self.node_map if node.label == fr),
                None,
            )

            to_node = next(
                (node for node in self.node_map if node.label == to),
                None,
            )

            if from_node is None:
                self.get_logger().warn(
                    f"Node {fr} (from) doesn't exist in graph. Skipping..."
                )
                continue

            if to_node is None:
                self.get_logger().warn(
                    f"Node {to} (to) doesn't exist in graph. Skipping..."
                )
                continue

            if not self._are_linear(node_a=fr, node_b=to):
                self.get_logger().warn(f"Edge {edge} not linear. Skipping...")
                continue

            fr_x, fr_y = gridbot.node_to_coords(fr)
            to_x, to_y = gridbot.node_to_coords(to)

            weight = 0

            if self.use_weights:
                weight=weight_str
            else:
                weight = abs(to_y - fr_y) + abs(to_x - fr_x)

            self.node_map[from_node].append(_MapEdge(fr=fr, to=to, weight=weight))

    def _generate_image(self):
        """
        Generate the graph image.
        """

        node_xs = [node.x for node in self.node_map]
        node_ys = [node.y for node in self.node_map]

        min_x = min(node_xs)
        max_x = max(node_xs)
        min_y = min(node_ys)
        max_y = max(node_ys)

        width = (max_x - min_x) + self.tag_size
        height = (max_y - min_y) + self.tag_size

        image = np.full(
            (height, width, 3),
            255,
            dtype=np.uint8,
        )

        for node_from, edges in self.node_map.items():
            for edge in edges:

                node_to = next(
                    (node for node in self.node_map if node.label == edge.to),
                    None,
                )

                if node_to is None:
                    continue

                from_x = node_from.x
                from_y = height - node_from.y

                to_x = node_to.x
                to_y = height - node_to.y

                cv2.line(
                    image,
                    (from_x, from_y),
                    (to_x, to_y),
                    self.line_color,
                    self.line_thickness,
                    cv2.LINE_4,
                )

                y_turn = -1 if from_y == min_y else 1
                x_turn = -1 if from_x == min_x else 1

                text_x_offset = x_turn * self.text_padding if from_x == to_x else 0
                text_y_offset = y_turn * self.text_padding if from_y == to_y else 0

                text_x = (from_x + to_x) // 2 - text_x_offset
                text_y = (from_y + to_y) // 2 - text_y_offset

                text_scale = cv2.getFontScaleFromHeight(
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    pixelHeight=self.text_size,
                    thickness=self.text_size,
                )

                self.get_logger().info(f"calculated text size: {text_scale}")

                cv2.putText(
                    image,
                    str(edge.weight),
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    text_scale,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                    False,
                )

        for node in self.node_map:

            marker = cv2.aruco.generateImageMarker(
                self.aruco_dict,
                node.tag_id,
                self.tag_size,
            )

            marker = cv2.cvtColor(
                marker,
                cv2.COLOR_GRAY2BGR,
            )

            
            center_x = node.x
            center_y = height - node.y

            x0 = center_x - self.tag_size // 2
            y0 = center_y - self.tag_size // 2

            x1 = x0 + self.tag_size
            y1 = y0 + self.tag_size


            image[y0:y1, x0:x1] = marker

        scale = min(
            (self.img_width - 2 * self.padding) / width,
            (self.img_height - 2 * self.padding) / height,
        )

        scaled_width = round(width * scale)
        scaled_height = round(height * scale)

        image = cv2.resize(
            image,
            (scaled_width, scaled_height),
            interpolation=cv2.INTER_NEAREST,
        )

        final_image = np.full(
            (self.img_height, self.img_width, 3),
            255,
            dtype=np.uint8,
        )

        x_offset = (
            self.padding + (self.img_width - 2 * self.padding - scaled_width) // 2
        )

        y_offset = (
            self.padding + (self.img_height - 2 * self.padding - scaled_height) // 2
        )

        final_image[
            y_offset : y_offset + scaled_height,
            x_offset : x_offset + scaled_width,
        ] = image

        self.get_logger().info(
            f"Generated map: {self.graph_width}*{self.graph_height}, {self.removed_nodes}"
        )

        return final_image

    def _generate_graph(self) -> dict[str, dict[str, int]]:
        graph: dict[str, dict[str, int]] = {}

        for node in self.node_map:
            graph[node.label] = {}

        for node_from, edges in self.node_map.items():
            for edge in edges:
                graph[node_from.label][edge.to] = edge.weight
                graph[edge.to][node_from.label] = edge.weight

        return graph


def main(args=None):
    node_name = "map_generator"
    print(f"Hi from {node_name}")

    rclpy.init(args=args)

    map_generator = MapGenerator(node_name)
    rclpy.spin_once(map_generator)

    map_generator.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
