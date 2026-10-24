from functools import partial

import rclpy
from rclpy.node import Node

from rclpy.parameter import Parameter


from gridbot_interfaces.srv import GenerateMap
import gridbot.gridbot_helpers as gridbot


import os
import yaml
import cv2
import numpy as np

from pathlib import Path


class MapGenerator(Node):

    # ============== INNER CLASSES ==============
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
        x: int
        y: int

        _tag_id: int
        _label: str

        def __init__(self, x, y, tag_id, h):
            self.x = x
            self.y = y
            self.tag_id = tag_id
            self.label = gridbot.aruco_to_node(id=tag_id, graph_height=h)

        def __str__(self):
            return f"{self.label}"

    # ============ PARAMS AND FIELDS ============

    MARKER_RAW_SIZE = 6  # assuming 4x4 markers
    DOTS_PER_CM = 118.11 # equivalent of 300 dpi


    yaml_save_path: str
    map_img_save_path: str
    markers_print_path: str

    map_name: str

    marker_scale: int
    text_size: int
    image_length: int

    line_color: tuple[int, int, int]
    marker_distance: int
    line_thickness: int

    border_padding: int
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

        self.service = self.create_service(
            GenerateMap, self.map_service_name, self.generate_map_callback
        )

    # ============== CALLBACK ==============

    def generate_map_callback(self, request, response):

        self.map_name = request.map_name

        overwrite: bool = request.overwrite

        markers_dir = Path(self.markers_print_path, self.map_name)

        if not overwrite:
            if Path(self.yaml_save_path, self.map_name + ".yaml").exists():
                response.success = False
                response.msg = f"map graph of name '{self.map_name}' already exists in {self.yaml_save_path}/{self.map_name}.yaml. To overwrite it, set the request's 'overwrite' param to true."
                return response
            elif Path(self.map_img_save_path, self.map_name + ".png").exists():
                response.success = False
                response.msg = f"map image of name '{self.map_name}' already exists in {self.map_img_save_path}/{self.map_name}.png. To overwrite it, set the request's 'overwrite' param to true."
                return response
            elif markers_dir.exists() and any(markers_dir.glob("*.png")):
                response.success = False
                response.msg = f"Marker images already exist in {markers_dir}. To overwrite them, set the request's 'overwrite' param to true."
                return response

        self._add_nodes()
        self._remove_nodes()
        self._add_edges()

        graph = self._generate_graph()

        marker_px = self.MARKER_RAW_SIZE * self.marker_scale
        internal_dim = max(
            (self.graph_width - 1) * (marker_px + self.marker_distance) + marker_px,
            (self.graph_height - 1) * (marker_px + self.marker_distance) + marker_px,
        )
        upscale = (self.image_length - 2 * self.border_padding) / internal_dim
        marker_final_px = marker_px * upscale
        model_base_size = 2.0
        marker_m_at_scale1 = marker_final_px * model_base_size / self.image_length
        sim_scale = (self.marker_desired_cm / 100.0) / marker_m_at_scale1


        yaml_contents = dict(
            graph_width=self.graph_width,
            graph_height=self.graph_height,
            sim_scale=round(sim_scale, 4),
            marker_scale=self.marker_scale,
            graph=graph,
        )

        yaml_path = str(Path(self.yaml_save_path, self.map_name + ".yaml"))

        with open(yaml_path, "w") as f:
            try:
                yaml.dump(yaml_contents, f, default_flow_style=False)
            except:
                response.success = False
                response.msg = f"Error saving map graph to {yaml_path}."
                return response

        map_img, markers = self._generate_image()

        map_path = str(Path(self.map_img_save_path, self.map_name + ".png"))

        success = cv2.imwrite(map_path, map_img)

        if not success:
            response.success = False
            response.msg = f"Error saving map image to {map_path}."
            return response

        if markers_dir.exists():
            for old_file in markers_dir.iterdir():
                old_file.unlink()
        else:
            markers_dir.mkdir(parents=True)

        for name, image in markers.items():
            marker_path = str(markers_dir / name)
            success = cv2.imwrite(marker_path, image)

            if not success:
                response.success = False
                response.msg = f"Error saving marker '{name}' image to {marker_path}."
                return response

        response.success = True
        response.msg = (
            f"Successfully created {self.map_name}'s map graph, map image and marker images. "
            f"---------Set the scale of the grid in your world to:---------: "
            f"<scale>{sim_scale:.4f} {sim_scale:.4f} 1</scale>"
        )
        return response

    # =========== HELPER METHODS ===========

    def _add_nodes(self):
        marker_px = self.MARKER_RAW_SIZE * self.marker_scale

        for y in range(self.graph_height):
            for x in range(self.graph_width):

                tag_id = x * self.graph_height + y

                tag_x = marker_px // 2 + x * (marker_px + self.marker_distance)

                tag_y = marker_px // 2 + y * (marker_px + self.marker_distance)

                map_node = self._MapNode(
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
                weight = weight_str
            else:
                weight = abs(to_y - fr_y) + abs(to_x - fr_x)

            self.node_map[from_node].append(self._MapEdge(fr=fr, to=to, weight=weight))

    def _generate_image(self):
        """
        Generate the graph image.
        """

        marker_px = self.MARKER_RAW_SIZE * self.marker_scale

        width = (self.graph_width - 1) * (marker_px + self.marker_distance) + marker_px
        height = (self.graph_height - 1) * (
            marker_px + self.marker_distance
        ) + marker_px

        image = np.full(
            (height, width, 3),
            255,
            dtype=np.uint8,
        )
        markers = {}

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

        scale = (self.image_length - 2 * self.border_padding) / max(width, height)

        scaled_width = round(width * scale)
        scaled_height = round(height * scale)


        print_marker_px = round(self.marker_desired_cm * self.DOTS_PER_CM)

        for node in self.node_map:

            marker = cv2.aruco.generateImageMarker(
                self.aruco_dict,
                node.tag_id,
                marker_px,
            )

            print_marker = cv2.aruco.generateImageMarker(
                self.aruco_dict,
                node.tag_id,
                print_marker_px,
            )

            marker_name = node.label + "_" + self.map_name + ".png"
            markers[marker_name] = print_marker

            marker = cv2.cvtColor(
                marker,
                cv2.COLOR_GRAY2BGR,
            )

            center_x = node.x
            center_y = height - node.y

            x0 = center_x - marker_px // 2
            y0 = center_y - marker_px // 2

            image[y0 : y0 + marker_px, x0 : x0 + marker_px] = marker


        image = cv2.resize(
            image,
            (scaled_width, scaled_height),
            interpolation=cv2.INTER_NEAREST,
        )

        final_image = np.full(
            (self.image_length, self.image_length, 3),
            255,
            dtype=np.uint8,
        )

        x_offset = (
            self.border_padding
            + (self.image_length - 2 * self.border_padding - scaled_width) // 2
        )

        y_offset = (
            self.border_padding
            + (self.image_length - 2 * self.border_padding - scaled_height) // 2
        )

        final_image[
            y_offset : y_offset + scaled_height,
            x_offset : x_offset + scaled_width,
        ] = image

        min_x = marker_px // 2
        min_y = marker_px // 2

        text_scale = cv2.getFontScaleFromHeight(
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            pixelHeight=round(self.text_size * scale),
            thickness=round(self.text_size * scale),
        )

        for node_from, edges in self.node_map.items():
            for edge in edges:

                node_to = next(
                    (node for node in self.node_map if node.label == edge.to),
                    None,
                )

                if node_to is None:
                    continue

                from_x = round(node_from.x * scale) + x_offset
                from_y = round((height - node_from.y) * scale) + y_offset

                to_x = round(node_to.x * scale) + x_offset
                to_y = round((height - node_to.y) * scale) + y_offset

                y_turn = -1 if node_from.y == (height - min_y) else 1
                x_turn = -1 if node_from.x == min_x else 1

                text_x_offset = (
                    round(x_turn * self.text_padding) if from_x == to_x else 0
                )
                text_y_offset = (
                    round(y_turn * self.text_padding) if from_y == to_y else 0
                )

                text_x = (from_x + to_x) // 2 - text_x_offset
                text_y = (from_y + to_y) // 2 - text_y_offset

                cv2.putText(
                    final_image,
                    str(edge.weight),
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    text_scale,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                    False,
                )

        self.get_logger().info(
            f"Generated map: {self.graph_width}*{self.graph_height}, {self.removed_nodes}"
        )

        return final_image, markers

    def _generate_graph(self) -> dict[str, dict[str, int]]:
        graph: dict[str, dict[str, int]] = {}

        for node in self.node_map:
            graph[node.label] = {}

        for node_from, edges in self.node_map.items():
            for edge in edges:
                graph[node_from.label][edge.to] = edge.weight
                graph[edge.to][node_from.label] = edge.weight

        return graph

    def _setup_parameters(self):
        p = partial(gridbot.declare_param, self)

        # Save paths
        self.yaml_save_path = p(
            "yaml_save_path",
            "config/",
            "Save path of the map yaml file. Relative to the package.",
        )
        self.map_img_save_path = p(
            "map_img_save_path", "assets/", "Save path of the generated map image."
        )
        self.markers_print_path = p(
            "markers_print_path",
            "assets/",
            "Save path of the generated marker images, useful for printing.",
        )

        self.map_service_name = p("map_service_name", "generate_map", "Name of the service used to generate the map.")


        # Image parameters
        self.image_length = p(
            "image_length",
            4096,
            "Side length of the square output image in pixels.",
            min_value=1,
            max_value=8192,
        )
        self.marker_scale = p(
            "marker_scale",
            10,
            "Scale multiplier for the ArUco marker.",
            min_value=1,
            max_value=200,
        )
        self.text_size = p(
            "text_size",
            25,
            "Size of edge weight text in final image pixels.",
            min_value=0,
            max_value=2048,
        )

        self.marker_desired_cm = p(
            "marker_desired_cm",
            8.5,
            "Desired size of markers in cm, used to rescale the sim model and prints.",
            min_value=0.0,
            max_value=100.0,
        )


        line_color = p("line_color", [255, 0, 0], "RGB color of graph lines.")
        if len(line_color) != 3:
            self.get_logger().fatal(
                f"Parameter 'line_color' = {line_color} must have 3 elements."
            )
            for c in line_color:
                c = min(0, max(255, c))

        self.line_color = line_color

        self.marker_distance = p(
            "marker_distance",
            20,
            "Distance between markers in internal pixels.",
            min_value=1,
            max_value=4096,
        )
        self.border_padding = p(
            "border_padding",
            200,
            "Padding from the image border in final image pixels.",
            min_value=0,
            max_value=2048,
        )
        self.text_padding = p(
            "text_padding",
            25,
            "Padding of weight text from its edge in final image pixels.",
            min_value=0,
            max_value=2048,
        )
        self.line_thickness = p(
            "line_thickness",
            4,
            "Thickness of graph lines in internal pixels.",
            min_value=1,
            max_value=100,
        )

        # Graph parameters
        self.graph_width = p(
            "graph_width",
            6,
            "Number of columns in the graph.",
            min_value=1,
            max_value=26,
        )
        self.graph_height = p(
            "graph_height", 5, "Number of rows in the graph.", min_value=1, max_value=26
        )
        self.removed_nodes = p(
            "removed_nodes",
            Parameter.Type.STRING_ARRAY,
            "List of disabled nodes in the graph.",
        )
        self.edges = p(
            "edges",
            Parameter.Type.STRING_ARRAY,
            "Graph edges formatted as FROM,TO,WEIGHT. Example: A1,B1,7",
        )
        self.use_weights = p(
            "use_weights",
            False,
            "Whether to use weights provided in edges, or to calculate them from node positions.",
        )

        gridbot.log_params(self)


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
