import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import (
    ParameterDescriptor,
    FloatingPointRange,
    IntegerRange,
    ListParametersResult,
)
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64, Int8
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import gridbot.gridbot_helpers as gridbot
import rclpy.parameter
from rclpy.parameter_event_handler import ParameterEventHandler

from gridbot_interfaces.msg import SimpleAruco

import numpy as np
import colorsys

import cv2
from cv_bridge import CvBridge

bridge = CvBridge()


class GridProcessor(Node):

    camera_topic: str
    line_offset_topic: str
    tracked_image_topic: str
    prepare_tracked_image: bool
    aruco_topic: str
    line_turning_topic: str
    line_color: list
    line_roi_height: float
    center_line_roi_width: int
    sides_line_roi_width: int

    aruco_roi_width: int
    aruco_roi_height: float

    line_turning_threshold: float
    aruco_padding: int
    midline_step: int
    followed_node_offset: int

    centerline_color: tuple = (0, 0, 200)
    node_color: tuple = (0, 150, 0)
    followed_node_color: tuple = (0, 255, 0)

    aruco_upper_color: tuple = (255, 0, 255)
    aruco_top_color: tuple = (0, 255, 255)

    img_w: int = 0
    img_h: int = 0
    center_roi_mask = None
    left_roi_mask  = None
    right_roi_mask = None
    aruco_roi_mask = None

    _is_tracking_marker: bool = False
    _old_points  = None
    _old_gray = None

    last_found_aruco: int
    last_found_direction: gridbot.ArucoDirection

    clahe: cv2.CLAHE

    h_thresh_low: int = 0
    s_low: int = 0
    v_low: int = 0

    h_thresh_high: int = 0
    s_high: int = 0
    v_high: int = 0

    handler: ParameterEventHandler

    def __init__(self, node_name: str):
        super().__init__(node_name)

        self._setup_parameters()

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

        aruco_params = cv2.aruco.DetectorParameters()

        aruco_params.minMarkerPerimeterRate = 0.5
        aruco_params.polygonalApproxAccuracyRate = 0.01
        aruco_params.minCornerDistanceRate = 0.1
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        aruco_params.cornerRefinementWinSize = 5
        aruco_params.cornerRefinementMaxIterations = 40

        self.detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        self.last_found_aruco = -1
        self.last_found_direction = None

        self.lk_params = dict(
            winSize=(50, 50),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self._old_gray = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.camera_sub = self.create_subscription(
            CompressedImage, self.camera_topic, self.image_callback, qos_profile=qos
        )
        self.line_offset_pub = self.create_publisher(
            Float64, self.line_offset_topic, qos_profile=qos
        )
        self.aruco_pub = self.create_publisher(
            SimpleAruco, self.aruco_topic, qos_profile=qos
        )
        self.tracked_image_pub = self.create_publisher(
            CompressedImage, self.tracked_image_topic, qos_profile=qos
        )
        self.line_turning_pub = self.create_publisher(
            Int8, self.line_turning_topic, qos_profile=qos
        )

        self.line_mask_pub = self.create_publisher(
            CompressedImage, "camera/line_mask", qos_profile=qos
        )

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.handler = ParameterEventHandler(self)

        for parameter_name in (
            "h_thresh_low",
            "s_low",
            "v_low",
            "h_thresh_high",
            "s_high",
            "v_high",
            "line_color",
        ):
            self.handler.add_parameter_callback(
                parameter_name=parameter_name,
                node_name=node_name,
                callback=self.hsv_parameter_callback,
            )

    def hsv_parameter_callback(self, p: rclpy.parameter.Parameter) -> None:
        value = rclpy.parameter.parameter_value_to_python(p.value)

        self.get_logger().info(f"Received an update to parameter: {p.name}: {value}")

        setattr(self, p.name, value)

    def image_callback(self, msg: CompressedImage):
        frame_raw = bridge.compressed_imgmsg_to_cv2(
            cmprs_img_msg=msg, desired_encoding="bgr8"
        )

        img_hsv = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2HSV)
        
        h, s, v = cv2.split(img_hsv)
        v = self.clahe.apply(v)
        img_hsv = cv2.merge([h, s, v])

        equalized = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

        frame = equalized

        if frame is None:
            self.get_logger().error("Error while trying to read frame.")
            return

        # in case image resolution changes (shouldn't, but...), update ROIs
        h, w = frame.shape[:2]
        if self.img_w != w or self.img_h != h:
            self.img_w = w
            self.img_h = h
            self._prepare_rois()

        upper_edge, top_edge, direction = None, None, None
        upper_edge, top_edge, direction = self.aruco_tracking(frame)

        centerline, target_node = self.line_tracking(frame)
        self.turning_tracking(frame)

        if self.prepare_tracked_image:
            tracked_image = self.prepare_image_with_trackers(
                frame, upper_edge, top_edge, direction, centerline, target_node, 1.5
            )

            tracked_image_msg = bridge.cv2_to_compressed_imgmsg(cvim=tracked_image)
            self.tracked_image_pub.publish(tracked_image_msg)

    def _quantize_image(self, img: np.ndarray, divider: int):

        frame = img // divider * divider + divider // 2

        return frame

    def _prepare_rois(self):
        """
        Prepares ROIs for left, center and right "line tracking areas".
        Prepares a ROI for ArUco tag searching.
        """

        self.get_logger().info("Setting up ROI masks.")
        roi_base = np.zeros((self.img_h, self.img_w), dtype=np.uint8)

        height_px = int(self.img_h * self.line_roi_height)
        sides_width_px = int(self.img_w * self.sides_line_roi_width)
        center_width_px = int(self.img_w * self.center_line_roi_width)

        (
            self.left_roi_mask,
            self.center_roi_mask,
            self.right_roi_mask,
            self.aruco_roi_mask,
        ) = [roi_base.copy() for _ in range(4)]

        self.left_roi_mask[
            self.img_h - height_px : self.img_h,
            :sides_width_px,
        ] = 255

        self.right_roi_mask[
            self.img_h - height_px : self.img_h,
            self.img_w - sides_width_px : self.img_w,
        ] = 255

        self.center_roi_mask[
            self.img_h - height_px : self.img_h,
            self.img_w // 2
            - center_width_px // 2 : self.img_w // 2
            + center_width_px // 2,
        ] = 255

        aruco_height_px = int(self.img_h * self.aruco_roi_height)
        aruco_width_px = int(self.img_w * self.aruco_roi_width)

        self.aruco_roi_mask[
            self.img_h - aruco_height_px : self.img_h,
            self.img_w // 2
            - aruco_width_px // 2 : self.img_w // 2
            + aruco_width_px // 2,
        ] = 255

        return

    def _calculate_aruco_direction(self, points):
        """
        Calculates relative direction of the ArUco marker based on two top points (world-relative)s.
        """

        tl = points[0]
        tr = points[1]

        dx = tr[0] - tl[0]
        dy = tr[1] - tl[1]

        angle_d = (np.degrees(np.arctan2(dy, dx))) % 360

        threshold = 45

        if 360 - threshold <= angle_d or 0 <= angle_d < threshold:
            return gridbot.ArucoDirection.NORTH
        elif threshold <= angle_d < 90 + threshold:
            return gridbot.ArucoDirection.WEST
        elif 90 + threshold <= angle_d < 180 + threshold:
            return gridbot.ArucoDirection.SOUTH
        else:  # 180 + threshold <= angle_d < 360-threshold
            return gridbot.ArucoDirection.EAST

    def aruco_tracking(self, frame: np.array):
        """
        Detects the largest ArUco tag and track its two sets of corners:
            [0] - camera-relative upper left corner of an ArUco marker
            [1] - camera-relative upper right corner of an ArUco marker
            [2] - world-relative upper left corner of an ArUco marker
            [3] - world-relative upper right corner of an ArUco marker

            e.g.

            < - world-relative direction of the ArUco marker (north)
                upper - EAST
                [0]------------[1]
                |              |
                |              |
            <   |              |
                |              |
                |              |
                [2]------------[.]

            In that case, [0]--[1] make up the upper edge (highest edge of the marker on the camera (lowest Y))
            and [2]--[0] make up the top edge (detector-decided corner 0 and 1 of the marker)

            As per explanation above, [0] and [3] end up being the same point.

        Once the upper corners cross the boundary line at the bottom of the screen, the robot is considered to have "passed" the marker by standing on it.
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray_crop = cv2.bitwise_and(
            src1=self.aruco_roi_mask, src2=gray, mask=self.aruco_roi_mask
        )

        if self._is_tracking_marker:

            new_points, st, _ = cv2.calcOpticalFlowPyrLK(
                self._old_gray, gray_crop, self._old_points, None, **self.lk_params
            )

            # Immediatelly ignore a tracked point if it is below the bottom line
            bottom_line = self.img_h - self.aruco_padding

            if new_points is None or all(st == 0):
                self.get_logger().info(
                    f"Lost all points for tag={self.last_found_aruco}. Returning to detection..."
                )
                self._is_tracking_marker = False
                return None, None, None

            # Mark points past the line as 2 to differentiate from failed (==0)
            for i, p in enumerate(new_points):
                if p[0, 1] > bottom_line:
                    self.get_logger().debug(f"Point {p} below Y={bottom_line}.")
                    st[i] = 2

            upper_edge_new = new_points[:2]
            top_edge_new = new_points[2:]

            upper_st = st[:2]
            top_st = st[2:]

            upper_edge_good = upper_edge_new[upper_st == 1]
            top_edge_good = top_edge_new[top_st == 1]

            direction = self.last_found_direction

            if top_edge_good is None or len(top_edge_good) != 2 or any(top_st != 1):

                self.get_logger().info(
                    f"Lost top edge for tag={self.last_found_aruco}."
                )

                if direction is not None:
                    self.get_logger().info(
                        f"Direction obtained from last known:\n{self.last_found_direction.name.capitalize()}."
                    )
            else:
                direction = self._calculate_aruco_direction(
                    top_edge_good.reshape(-1, 2)
                )
                if direction != self.last_found_direction:
                    self.get_logger().info(
                        f"Direction obtained from new points:\n{direction.name.capitalize()}."
                    )
                    self.last_found_direction = direction

            if upper_edge_good is None or all(upper_st == 0):
                self.get_logger().info(
                    f"Lost upper edge for tag={self.last_found_aruco}. Returning to detection..."
                )
                self._is_tracking_marker = False
                return None, None, None

            # At least one point passed the line - publish, since we're likely on the point
            elif any(upper_st == 2):
                self.get_logger().info(
                    f"Marker {self.last_found_aruco} passed bottom of the screen at Y={bottom_line}. Publishing:..."
                )
                aruco_msg = SimpleAruco(
                    id=int(self.last_found_aruco),
                    direction=direction.value,
                )
                self.aruco_pub.publish(aruco_msg)

                # reset
                self._is_tracking_marker = False
                self.last_found_direction = None
                self.last_found_aruco = -1
                self._old_gray = None
                self._old_points = None
                return upper_edge_new, top_edge_new, direction

            self._old_gray = gray_crop
            self._old_points = new_points[st == 1].reshape(-1, 1, 2).astype(np.float32)
            return upper_edge_new, top_edge_new, direction

        else:
            corners, ids, _ = self.detector.detectMarkers(gray_crop)

            if ids is None:
                return None, None, None

            markers = {
                int(id): corners.reshape(4, 2)
                for id, corners in zip(ids.flatten(), corners)
            }

            largest_id, largest_corners = sorted(
                markers.items(), key=lambda item: cv2.contourArea(item[1]), reverse=True
            )[0]

            self.get_logger().info(
                f"Found largest marker:\nID={largest_id}\ncorners:\n{np.array(largest_corners,dtype=np.int32)}"
            )

            upper_points = sorted(
                sorted(largest_corners, key=lambda c: c[1])[:2], key=lambda c: c[0]
            )

            top_edge = largest_corners[:2]

            self.last_found_aruco = largest_id
            self._old_gray = gray_crop.copy()
            self._old_points = np.array(
                np.append(upper_points, top_edge), dtype=np.float32
            ).reshape(-1, 1, 2)
            self._is_tracking_marker = True
            return None, None, None

    def _make_window(self, name: str, src: np.array):
        cv2.namedWindow(winname=name, flags=cv2.WINDOW_FREERATIO)
        cv2.imshow(winname=name, mat=src)

    def line_tracking(self, frame_raw: np.array):
        """
        1. Masks out the frame by the line_color. Converts each bit island to a contour.
        2. Picks the contour by these criteria:
            a) Biggest area
            b) Has centroid in
        3. For this contour, finds a centerline node chain
        4. For picked node, finds its offset and sends it.
        """

        frame = cv2.medianBlur(frame_raw, 5)

        camera_center = (self.img_w // 2, self.img_h // 2)

        line_bitmask = self.get_mask_from_color(img=frame, color=self.line_color)

        line_bitmask_crop = cv2.bitwise_and(
            src1=line_bitmask, src2=line_bitmask, mask=self.center_roi_mask
        )

        line_contours = self.get_contours(line_bitmask_crop)

        if len(line_contours) == 0:
            return None, None

        max_line_contour = max(line_contours, key=cv2.contourArea)

        max_line_mask = np.zeros_like(frame)
        cv2.drawContours(max_line_mask, [max_line_contour], -1, (255, 255, 255), -1)


        mask_msg = bridge.cv2_to_compressed_imgmsg(max_line_mask)

        self.line_mask_pub.publish(mask_msg)

        centerline = self._get_centerline(max_line_mask)

        if centerline is None or len(centerline) == 0:
            return None, None

        # NOTE: by default, pick the node selected via self.followed_node_offset.
        # However, in case we don't have enough nodes, pick the furthest one.
        # While normally picking the furthest one could lead to turning too quickly,
        # in case of very few nodes (as node offset shouldn't be too high) it's fine.
        target_idx = min(len(centerline), self.followed_node_offset)

        target_node = centerline[-target_idx]

        pixel_error = camera_center[0] - target_node[0]

        pixel_error_normalized = pixel_error / self.img_w

        pixel_error_msg = Float64(data=pixel_error_normalized)
        self.line_offset_pub.publish(pixel_error_msg)

        return centerline, target_node

    def prepare_image_with_trackers(
        self,
        frame,
        upper_edge,
        top_edge,
        direction: gridbot.ArucoDirection,
        centerline,
        target_node,
        font_scale: float = 1.0,
    ):

        if centerline is not None:
            upper_pts = np.array(centerline, dtype=np.int32).reshape(-1, 1, 2)

            cv2.polylines(
                img=frame,
                pts=[upper_pts],
                isClosed=False,
                color=self.centerline_color,
                thickness=3,
            )

            for pt in upper_pts.reshape(-1, 2):
                if pt[0] == target_node[0] and pt[1] == target_node[1]:
                    cv2.circle(
                        img=frame,
                        center=pt,
                        radius=25,
                        color=self.followed_node_color,
                        thickness=4,
                    )
                else:
                    cv2.circle(
                        img=frame,
                        center=pt,
                        radius=15,
                        color=self.node_color,
                        thickness=2,
                    )

        if direction is not None and upper_edge is not None and top_edge is not None:

            upper_pts = np.array(upper_edge, dtype=np.int32).reshape(-1, 2)
            top_pts = np.array(top_edge, dtype=np.int32).reshape(-1, 2)

            center = upper_pts.mean(axis=0)

            cv2.putText(
                img=frame,
                text=f"{direction.name.capitalize()}",
                org=tuple(center.astype(int)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=self.aruco_upper_color,
                thickness=4,
                lineType=cv2.LINE_AA,
            )

            for idx, pt in enumerate(top_pts):

                cv2.putText(
                    img=frame,
                    text=f"{"TL" if idx % 2 == 0 else "TR"}",
                    org=pt,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    color=self.aruco_top_color,
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )

                cv2.circle(
                    img=frame,
                    center=pt,
                    radius=25,
                    color=self.aruco_top_color,
                    thickness=4,
                )

            for idx, pt in enumerate(upper_pts):

                cv2.putText(
                    img=frame,
                    text=f"{"UL" if idx % 2 == 0 else "UR"}",
                    org=pt,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    color=self.aruco_upper_color,
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )

                cv2.circle(
                    img=frame,
                    center=pt,
                    radius=15,
                    color=self.aruco_upper_color,
                    thickness=4,
                )

        roi_specs = (
            (self.left_roi_mask, (0, 0, 255), "Left ROI"),
            (self.right_roi_mask, (255, 0, 0), "Right ROI"),
            (self.center_roi_mask, (0, 255, 0), "Center ROI"),
            (self.aruco_roi_mask, (0, 255, 255), "Aruco ROI"),
        )

        for mask, color, name in roi_specs:
            ys, xs = np.where(mask > 0)
            if xs.size == 0 or ys.size == 0:
                continue

            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            cv2.rectangle(
                img=frame,
                pt1=(x_min, y_min),
                pt2=(x_max, y_max),
                color=color,
                thickness=2,
            )

            cv2.putText(
                img=frame,
                text=name,
                org=(x_min + 20, y_min + 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        return frame


    def turning_tracking(self, frame):
        """
        Sends signals regarding where lines appear/disappear (left, centre, right) side of the image, used to track the turning of the robot
        """


        line_bitmask = self.get_mask_from_color(img=frame, color=self.line_color)

        rois = {
            gridbot.LineObservation.LEFT: self.left_roi_mask,
            gridbot.LineObservation.CENTER: self.center_roi_mask,
            gridbot.LineObservation.RIGHT: self.right_roi_mask,
        }

        shares: dict[gridbot.LineObservation, int] = {}

        for region, roi_mask in rois.items():
            crop = cv2.bitwise_and(line_bitmask, line_bitmask, mask=roi_mask)

            col_has_line = np.count_nonzero(crop, axis=0) > 0
            row_has_line = np.count_nonzero(crop, axis=1) > 0

            width = np.count_nonzero(col_has_line)
            height = np.count_nonzero(row_has_line)

            # Ignore shapes that are wider than they are higher (lines going sideways)
            if width > height:
                shares[region] = 0
                continue

            roi_cols = np.where(np.count_nonzero(roi_mask, axis=0) > 0)[0]
            if roi_cols.size == 0:
                shares[region] = 0
                continue
            roi_left, roi_right = roi_cols[0], roi_cols[-1]

            line_cols = np.where(col_has_line)[0]
            if line_cols.size == 0:
                shares[region] = 0
                continue
            line_left, line_right = line_cols[0], line_cols[-1]

            # Only count shares which mask is WITHIN the roi (and not outside, or touching the borders) - this prevents premature readings
            if line_left <= roi_left or line_right >= roi_right:
                shares[region] = 0
                continue

            shares[region] = cv2.countNonZero(crop)

        all_line_pixels = sum(shares.values())

        flags = gridbot.LineObservation.NONE

        if all_line_pixels > 0:
            for region, share in shares.items():
                if share / all_line_pixels >= self.line_turning_threshold:
                    flags |= region

        int_msg = Int8()
        int_msg.data = flags

        self.line_turning_pub.publish(int_msg)

    def _get_centerline(self, src):
        centerline = []

        h, w = src.shape[:2]

        for i in range(0, h - self.midline_step, self.midline_step):
            row = src[i, :, 0] > 0

            if len(row) == 0:
                self.get_logger().info(f"Row {row} has no white pixels.")
                continue

            # based on https://stackoverflow.com/a/77593930

            lane_ranges = []
            inside_lane = False

            bound = []
            for j, b in enumerate(row):
                # lane start
                if b and not inside_lane:
                    bound.append(j)
                    inside_lane = True
                # lane end
                elif not b and inside_lane:
                    bound.append(j - 1)
                    inside_lane = False
                    lane_ranges.append(tuple(bound))
                    bound.clear()
                # edge case - lane bound not closed
                elif j + 1 == len(row) and inside_lane:
                    bound.append(j)
                    lane_ranges.append(tuple(bound))
                    inside_lane = False
                    bound.clear()

            # in case of multiple lane ranges, pick the one closest to image center
            center = w // 2
            leftmost = None
            rightmost = None
            best_range = None
            min_dist = float("inf")

            for r in lane_ranges:
                lane_center = (r[0] + r[1]) // 2
                dist = abs(lane_center - center)

                if dist < min_dist:
                    min_dist = dist
                    best_range = r

            if best_range is None:
                continue

            leftmost = best_range[0]
            rightmost = best_range[1]

            middle_x = (leftmost + rightmost) // 2
            centerline.append((middle_x, i))

        return centerline

    def _get_centroid(self, contour):
        M = cv2.moments(contour)

        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return (cx, cy)

    def get_contours(self, src):
        ret, thresh = cv2.threshold(src, 127, 255, 0)

        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        return contours

    def get_mask_from_color(self, img, color):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # color is RGB
        color_h, color_s, color_v = colorsys.rgb_to_hsv(
            r=color[0] / 255.0,
            g=color[1] / 255.0,
            b=color[2] / 255.0,
        )

        color_h = round(color_h * 179)
        color_s = round(color_s * 255)
        color_v = round(color_v * 255)

        h_low = color_h - self.h_thresh_low
        h_high = color_h + self.h_thresh_high

        s_low = max(0, self.s_low)
        s_high = min(255, self.s_high)

        v_low = max(0, self.v_low)
        v_high = min(255,self.v_high)


        if h_low <= 0 and h_high >= 179:
            # Entire hue spectrum
            lower = np.array([0, s_low, v_low])
            upper = np.array([179, s_high, v_high])

            self.get_logger().debug(
                f"HSV mask | ALL H | S {s_low}:{s_high} | V {v_low}:{v_high}"
            )

            return cv2.inRange(img_hsv, lower, upper)

        elif h_low < 0:
            # Wraps through 0
            lower_1 = np.array([0, s_low, v_low])
            upper_1 = np.array([h_high, s_high, v_high])

            lower_2 = np.array([180 + h_low, s_low, v_low])
            upper_2 = np.array([179, s_high, v_high])

            self.get_logger().debug(
                f"HSV mask | H {180 + h_low}:179 + 0:{h_high} | "
                f"S {s_low}:{s_high} | V {v_low}:{v_high}"
            )

            mask1 = cv2.inRange(img_hsv, lower_1, upper_1)
            mask2 = cv2.inRange(img_hsv, lower_2, upper_2)

            return cv2.bitwise_or(mask1, mask2)

        elif h_high > 179:
            # Wraps through 179 -> 0
            lower_1 = np.array([h_low, s_low, v_low])
            upper_1 = np.array([179, s_high, v_high])

            lower_2 = np.array([0, s_low, v_low])
            upper_2 = np.array([h_high - 180, s_high, v_high])

            self.get_logger().debug(
                f"HSV mask | H {h_low}:179 + 0:{h_high - 180} | "
                f"S {s_low}:{s_high} | V {v_low}:{v_high}"
            )

            mask1 = cv2.inRange(img_hsv, lower_1, upper_1)
            mask2 = cv2.inRange(img_hsv, lower_2, upper_2)

            return cv2.bitwise_or(mask1, mask2)

        else:
            # Normal, non-wrapping range
            lower = np.array([h_low, s_low, v_low])
            upper = np.array([h_high, s_high, v_high])

            self.get_logger().debug(
                f"HSV mask | H {h_low}:{h_high} | "
                f"S {s_low}:{s_high} | V {v_low}:{v_high}"
            )

            return cv2.inRange(img_hsv, lower, upper)

        
    def _setup_parameters(self):
        self.declare_parameter(
            name="camera_topic",
            value="camera/image_raw/compressed",
            descriptor=ParameterDescriptor(
                description="Topic used to receive camera images (compressed).",
            ),
        )

        self.declare_parameter(
            name="line_offset_topic",
            value="line_offset",
            descriptor=ParameterDescriptor(
                description="Topic used to send line_offset.",
            ),
        )

        self.declare_parameter(
            name="tracked_image_topic",
            value="tracked_image",
            descriptor=ParameterDescriptor(
                description="Topic used to send images with additional visuals.",
            ),
        )

        self.declare_parameter(
            name="prepare_tracked_image",
            value=False,
            descriptor=ParameterDescriptor(
                description="Whether to build and publish the tracked image visualization.",
            ),
        )

        self.declare_parameter(
            name="aruco_topic",
            value="aruco",
            descriptor=ParameterDescriptor(
                description="Topic used to send aruco ID + direction.",
            ),
        )

        self.declare_parameter(
            name="line_turning_topic",
            value="line_turning",
            descriptor=ParameterDescriptor(
                description="Topic used to send information about turning and line visibility.",
            ),
        )

        self.declare_parameter(
            name="line_color",
            value=[0, 0, 0],
            descriptor=ParameterDescriptor(
                description="RGB Color of the line to target.",
            ),
        )

        self.declare_parameter(
            name="line_roi_height",
            value=0.5,
            descriptor=ParameterDescriptor(
                description="Height of the line search ROI - [0..1] factor.",
                floating_point_range=[
                    FloatingPointRange(
                        from_value=0.0,
                        to_value=1.0,
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="center_line_roi_width",
            value=0.4,
            descriptor=ParameterDescriptor(
                description="Width of the central line search ROI - [0..1] factor.",
                floating_point_range=[
                    FloatingPointRange(
                        from_value=0.0,
                        to_value=1.0,
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="sides_line_roi_width",
            value=0.3,
            descriptor=ParameterDescriptor(
                description="Width of the side line search ROIs - [0..1] factor.",
                floating_point_range=[
                    FloatingPointRange(
                        from_value=0.0,
                        to_value=1.0,
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="aruco_roi_width",
            value=0.5,
            descriptor=ParameterDescriptor(
                description="Width of the ArUco search ROI - [0..1] factor.",
                floating_point_range=[
                    FloatingPointRange(
                        from_value=0.0,
                        to_value=1.0,
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="aruco_roi_height",
            value=0.5,
            descriptor=ParameterDescriptor(
                description="Height of the ArUco search ROI - [0..1] factor.",
                floating_point_range=[
                    FloatingPointRange(
                        from_value=0.0,
                        to_value=1.0,
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="aruco_padding",
            value=5,
            descriptor=ParameterDescriptor(
                description='Offset from the bottom of the frame where the node will count a found ArUco code as "Passed by".',
            ),
        )

        self.declare_parameter(
            name="line_turning_threshold",
            value=0.2,
            descriptor=ParameterDescriptor(
                description="0..1 factor of all of grid line's pixels in a specific ROI to be counted as having a line in it.",
                floating_point_range=[
                    FloatingPointRange(
                        from_value=0.0,
                        to_value=1.0,
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="midline_step",
            value=10,
            descriptor=ParameterDescriptor(
                description="Pixel rows skipped when looking for centerline nodes.",
                integer_range=[
                    IntegerRange(
                        from_value=1,
                        to_value=2**31 - 1,
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="followed_node_offset",
            value=4,
            descriptor=ParameterDescriptor(
                description="Which node from the bottom will be targeted.",
                integer_range=[
                    IntegerRange(
                        from_value=0,
                        to_value=2**31 - 1,
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="h_thresh_low",
            value=0,
            descriptor=ParameterDescriptor(
                description="Lower bound of the hue threshold for line detection.",
                integer_range=[
                    IntegerRange(
                        from_value=0,
                        to_value=179,
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="h_thresh_high",
            value=0,
            descriptor=ParameterDescriptor(
                description="Upper bound of the hue threshold for line detection.",
                integer_range=[
                    IntegerRange(
                        from_value=0,
                        to_value=179,
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="s_low",
            value=0,
            descriptor=ParameterDescriptor(
                description="Lower bound of the saturation threshold for line detection.",
                integer_range=[
                    IntegerRange(
                        from_value=0,
                        to_value=255,
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="s_high",
            value=0,
            descriptor=ParameterDescriptor(
                description="Upper bound of the saturation threshold for line detection.",
                integer_range=[
                    IntegerRange(
                        from_value=0,
                        to_value=255,
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="v_low",
            value=0,
            descriptor=ParameterDescriptor(
                description="Lower bound of the saturation threshold for line detection.",
                integer_range=[
                    IntegerRange(
                        from_value=0,
                        to_value=255,
                    )
                ],
            ),
        )

        self.declare_parameter(
            name="v_high",
            value=0,
            descriptor=ParameterDescriptor(
                description="Upper bound of the saturation threshold for line detection.",
                integer_range=[
                    IntegerRange(
                        from_value=0,
                        to_value=255,
                    )
                ],
            ),
        )

        self.camera_topic = self.get_parameter("camera_topic").value
        self.line_offset_topic = self.get_parameter("line_offset_topic").value
        self.tracked_image_topic = self.get_parameter("tracked_image_topic").value
        self.prepare_tracked_image = self.get_parameter("prepare_tracked_image").value
        self.aruco_topic = self.get_parameter("aruco_topic").value
        self.line_turning_topic = self.get_parameter("line_turning_topic").value

        line_color = self.get_parameter("line_color").value
        if isinstance(line_color, (list, tuple)):
            self.line_color = [
                int(np.clip(c, 0, 255)) for c in (list(line_color) + [0, 0, 0])[:3]
            ]
        else:
            self.line_color = [255, 0, 0]

        self.line_roi_height = self.get_parameter("line_roi_height").value
        self.center_line_roi_width = self.get_parameter("center_line_roi_width").value
        self.sides_line_roi_width = self.get_parameter("sides_line_roi_width").value

        self.aruco_roi_width = self.get_parameter("aruco_roi_width").value
        self.aruco_roi_height = self.get_parameter("aruco_roi_height").value
        self.aruco_padding = self.get_parameter("aruco_padding").value

        self.line_turning_threshold = self.get_parameter("line_turning_threshold").value

        self.midline_step = self.get_parameter("midline_step").value
        self.followed_node_offset = self.get_parameter("followed_node_offset").value

        self.h_thresh_low = self.get_parameter("h_thresh_low").value
        self.s_low = self.get_parameter("s_low").value
        self.v_low = self.get_parameter("v_low").value

        self.h_thresh_high = self.get_parameter("h_thresh_high").value
        self.s_high = self.get_parameter("s_high").value
        self.v_high = self.get_parameter("v_high").value

        result: ListParametersResult = self.list_parameters([], depth=0)

        parameters = self.get_parameters(list(result.names))


        for parameter in parameters:
            self.get_logger().info(f"{parameter.name}: {parameter.value}")




def main(args=None):
    node_name = "grid_processor"
    print(f"Hi from {node_name}.")
    rclpy.init(args=args)

    grid_processor = GridProcessor(node_name)
    rclpy.spin(grid_processor)
    grid_processor.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
