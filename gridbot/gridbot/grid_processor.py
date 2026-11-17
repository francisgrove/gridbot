from functools import partial

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float64, Int8
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import gridbot.gridbot_helpers as gridbot
import rclpy.parameter
from rclpy.parameter_event_handler import ParameterEventHandler

from gridbot_interfaces.msg import SimpleAruco

import numpy as np
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
    line_color_hsv: list
    line_roi_height: float
    centerline_roi_width: int

    aruco_roi_width: int
    aruco_roi_height: float

    line_turning_min_angle: float
    aruco_padding: int
    centerline_step: int
    centerline_node_offset: int

    centerline_color: tuple = (200, 0, 0)
    node_color: tuple = (0, 150, 0)
    followed_node_color: tuple = (0, 255, 0)

    aruco_upper_color: tuple = (255, 0, 255)
    aruco_top_color: tuple = (0, 255, 255)

    img_w: int = 0
    img_h: int = 0
    _center_roi: tuple = None
    _aruco_roi: tuple = None

    _is_tracking_marker: bool = False
    _lost_marker_top_points: bool = False
    _turning_line_exited: bool = False
    _old_points = None
    _old_gray = None

    last_found_aruco: int
    last_found_direction: gridbot.ArucoDirection

    h_low: int = 0
    s_low: int = 0
    v_low: int = 0

    h_high: int = 0
    s_high: int = 0
    v_high: int = 0

    _handler: ParameterEventHandler

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
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.camera_sub = self.create_subscription(
            Image, self.camera_topic, self.image_callback, qos_profile=qos
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

        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self._handler = ParameterEventHandler(self)

        for parameter_name in (
            "h_thresh_low",
            "s_low",
            "v_low",
            "h_thresh_high",
            "s_high",
            "v_high",
            "line_color_hsv",
        ):
            self._handler.add_parameter_callback(
                parameter_name=parameter_name,
                node_name=node_name,
                callback=self.hsv_parameter_callback,
            )

    def image_callback(self, msg: Image):
        frame_raw = bridge.imgmsg_to_cv2(img_msg=msg, desired_encoding="bgr8")

        if frame_raw is None:
            self.get_logger().error("Error while trying to read frame.")
            return

        # in case image resolution changes (shouldn't, but...), update ROIs
        h, w = frame_raw.shape[:2]
        if self.img_w != w or self.img_h != h:
            self.img_w = w
            self.img_h = h
            self._prepare_roi_limits()

        camera_top_points, marker_top_points, direction = None, None, None
        camera_top_points, marker_top_points, direction = self._aruco_tracking(frame_raw)

        self._color_mask_buf[:] = 0
        color_crop = frame_raw[self._color_roi_y_start :]

        crop_mask = self._get_mask_from_color(img=color_crop, color_hsv=self.line_color_hsv)
        self._color_mask_buf[self._color_roi_y_start :] = crop_mask

        centerline, target_node = self._line_tracking(self._color_mask_buf)
        self._turning_tracking(self._color_mask_buf)

        if self.prepare_tracked_image:
            tracked_image = self._prepare_image_with_trackers(
                frame_raw, camera_top_points, marker_top_points, direction, centerline, target_node, 1.5
            )

            tracked_image_msg = bridge.cv2_to_compressed_imgmsg(cvim=tracked_image)
            self.tracked_image_pub.publish(tracked_image_msg)

    def _aruco_tracking(self, frame: np.array):
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
                |               |
                |               |
            <   |               |
                |               |
                |               |
                [2]------------[.]

            In that case, [0]--[1] make up the upper edge (highest edge of the marker on the camera (lowest Y))
            and [2]--[0] make up the top edge (detector-decided corner 0 and 1 of the marker)

            As per explanation above, [0] and [3] end up being the same point.

        Once the upper corners cross the boundary line at the bottom of the screen, the robot is considered to have "passed" the marker by standing on it.
        """

        y0, _, x0, x1 = self._aruco_roi
        self._gray_buf[:] = 0
        self._gray_buf[y0:, x0:x1] = cv2.cvtColor(frame[y0:, x0:x1], cv2.COLOR_BGR2GRAY)
        gray_crop = self._gray_buf

        # ==================== TRACKING MODE ====================
        if self._is_tracking_marker:

            new_points, st, _ = cv2.calcOpticalFlowPyrLK(
                self._old_gray, gray_crop, self._old_points, None, **self.lk_params
            )

            # Immediately ignore a tracked point if it is below the bottom line
            bottom_line = self.img_h - self.aruco_padding

            if all(st == 0):
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

            camera_top_points_new = new_points[:2]
            marker_top_points_new = new_points[2:]

            camera_st = st[:2]
            marker_st = st[2:]

            marker_top_points_good = marker_top_points_new[marker_st == 1]

            direction = self.last_found_direction

            if any(marker_st != 1):
                if not self._lost_marker_top_points:
                    self._lost_marker_top_points = True
                    self.get_logger().info(
                        f"Lost top edge for tag={self.last_found_aruco}. "
                        f"Direction from last known: {self.last_found_direction.name.capitalize()}."
                    )
            else:
                self._lost_marker_top_points = False
                direction = self._calculate_aruco_direction(
                    marker_top_points_good.reshape(-1, 2)
                )
                if direction != self.last_found_direction:
                    self.get_logger().info(
                        f"Direction obtained from new points:\n{direction.name.capitalize()}."
                    )
                    self.last_found_direction = direction

            if any(camera_st == 0):
                self.get_logger().info(
                    f"Lost upper edge for tag={self.last_found_aruco}. Returning to detection..."
                )
                self._is_tracking_marker = False
                return None, None, None

            # At least one point passed the line - publish, since we're likely on the point
            elif any(camera_st == 2):
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
                return camera_top_points_new, marker_top_points_new, direction

            self._old_gray = gray_crop.copy()
            self._old_points = new_points.reshape(-1, 1, 2).astype(np.float32)
            return camera_top_points_new, marker_top_points_new, direction

        # ==================== DETECTING MODE ====================
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

            marker_top_points = largest_corners[:2]

            self.last_found_aruco = largest_id
            self.last_found_direction = self._calculate_aruco_direction(marker_top_points)
            self._old_gray = gray_crop.copy()
            self._old_points = np.array(
                np.append(upper_points, marker_top_points), dtype=np.float32
            ).reshape(-1, 1, 2)
            self._is_tracking_marker = True
            self._lost_marker_top_points = False
            return None, None, None

    def _line_tracking(self, line_bitmask: np.array):
        """
        1. Masks out the frame by the line_color. Converts each bit island to a contour.
        2. Picks the contour by these criteria:
            a) Biggest area
            b) Has centroid in
        3. For this contour, finds a centerline node chain
        4. For picked node, finds its offset and sends it.
        """

        camera_center = (self.img_w // 2, self.img_h // 2)

        y0, y1, x0, x1 = self._center_roi
        roi_slice = line_bitmask[y0:y1, x0:x1]

        line_contours = self._get_contours(roi_slice)

        if len(line_contours) == 0:
            return None, None

        max_line_contour = max(line_contours, key=cv2.contourArea)
        max_line_contour = max_line_contour + np.array([x0, y0])

        self._line_mask_buf[:] = 0
        cv2.drawContours(
            self._line_mask_buf, [max_line_contour], -1, (255, 255, 255), -1
        )

        centerline = self._get_centerline(self._line_mask_buf)

        if centerline is None or len(centerline) == 0:
            return None, None

        # by default, pick the node selected via self.centerline_node_offset.
        # However, in case we don't have enough nodes, pick the furthest one.
        # While normally picking the furthest one could lead to turning too quickly,
        # in case of very few nodes (as node offset shouldn't be too high) it's fine.
        target_idx = min(len(centerline), self.centerline_node_offset)

        target_node = centerline[target_idx - 1]

        # self.get_logger().info(f"For offset: {self.centerline_node_offset} we pick node: {target_idx} for centerline of length {len(centerline)})")

        pixel_error = camera_center[0] - target_node[0]

        pixel_error_normalized = pixel_error / self.img_w

        pixel_error_msg = Float64(data=pixel_error_normalized)
        self.line_offset_pub.publish(pixel_error_msg)

        return centerline, target_node

    def _turning_tracking(self, line_bitmask):
        """
        Checks if a vertical-enough line is visible in the center ROI.
        Publishes CENTER when the largest contour's bounding box angle
        exceeds line_turning_min_angle, NONE otherwise.
        """

        y0, y1, x0, x1 = self._center_roi
        roi_slice = line_bitmask[y0:y1, x0:x1]

        line_obs = gridbot.LineObservation.NONE

        contours = self._get_contours(roi_slice)
        if len(contours) > 0:

            max_contour = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(max_contour)
            roi_w = x1 - x0
            roi_center_x = roi_w // 2
            fifth_start = 2 * roi_w // 5
            fifth_end = 3 * roi_w // 5

            in_fifth = bw > 0 and by == 0 and bx < fifth_end and bx + bw > fifth_start

            if not in_fifth:
                self._turning_line_exited = True

            if in_fifth and self._turning_line_exited:
                if bx <= roi_center_x <= bx + bw:
                    angle = np.degrees(np.arctan2(bh, bw))
                    if angle >= self.line_turning_min_angle:
                        line_obs = gridbot.LineObservation.CENTER
                        self._turning_line_exited = False

        int_msg = Int8()
        int_msg.data = line_obs.value

        self.line_turning_pub.publish(int_msg)

    def _prepare_image_with_trackers(
        self,
        frame,
        camera_top_points,
        marker_top_points,
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

        if direction is not None and camera_top_points is not None and marker_top_points is not None:

            upper_pts = np.array(camera_top_points, dtype=np.int32).reshape(-1, 2)
            top_pts = np.array(marker_top_points, dtype=np.int32).reshape(-1, 2)

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
            (self._center_roi, (0, 255, 0), "Center ROI"),
            (self._aruco_roi, (0, 255, 255), "Aruco ROI"),
        )

        for (y0, y1, x0, x1), color, name in roi_specs:
            cv2.rectangle(
                img=frame,
                pt1=(x0, y0),
                pt2=(x1, y1),
                color=color,
                thickness=2,
            )

            cv2.putText(
                img=frame,
                text=name,
                org=(x0 + 20, y0 + 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        bottom_line_y = self.img_h - self.aruco_padding
        cv2.line(frame, (0, bottom_line_y), (self.img_w, bottom_line_y), (0, 0, 255), 2)

        return frame

    def _quantize_image(self, img: np.ndarray, divider: int):

        frame = img // divider * divider + divider // 2

        return frame

    def _prepare_roi_limits(self):
        """
        Computes ROI bounds (y_start, y_end, x_start, x_end) for line tracking
        areas and ArUco tag searching.
        """

        self.get_logger().info("Setting up ROI limits.")

        height_px = int(self.img_h * self.line_roi_height)
        center_width_px = int(self.img_w * self.centerline_roi_width)

        aruco_height_px = int(self.img_h * self.aruco_roi_height)
        aruco_width_px = int(self.img_w * self.aruco_roi_width)

        y_top = self.img_h - height_px

        self._center_roi = (
            y_top,
            self.img_h,
            self.img_w // 2 - center_width_px // 2,
            self.img_w // 2 + center_width_px // 2,
        )
        self._aruco_roi = (
            self.img_h - aruco_height_px,
            self.img_h,
            self.img_w // 2 - aruco_width_px // 2,
            self.img_w // 2 + aruco_width_px // 2,
        )

        self._color_mask_buf = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        self._line_mask_buf = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        self._gray_buf = np.zeros((self.img_h, self.img_w), dtype=np.uint8)

        self._color_roi_y_start = y_top

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

    def _get_centerline(self, src):
        centerline = []
        center = self.img_w // 2

        for i in range(0, self.img_h - self.centerline_step, self.centerline_step):
            nonzero_idx = np.nonzero(src[i, :, 0])[0]
            if nonzero_idx.size == 0:
                continue

            gaps = np.where(np.diff(nonzero_idx) > 1)[0]

            if gaps.size == 0:
                mid = int((nonzero_idx[0] + nonzero_idx[-1]) // 2)
            else:
                starts = np.empty(gaps.size + 1, dtype=nonzero_idx.dtype)
                ends = np.empty(gaps.size + 1, dtype=nonzero_idx.dtype)
                starts[0] = nonzero_idx[0]
                starts[1:] = nonzero_idx[gaps + 1]
                ends[:-1] = nonzero_idx[gaps]
                ends[-1] = nonzero_idx[-1]

                mids = (starts + ends) // 2
                mid = int(mids[np.argmin(np.abs(mids - center))])

            centerline.append((mid, i))

        centerline.reverse()
        return centerline

    def _get_centroid(self, contour):
        M = cv2.moments(contour)

        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return (cx, cy)

    def _get_contours(self, src):
        ret, thresh = cv2.threshold(src, 127, 255, 0)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    def _get_mask_from_color(self, img, color_hsv):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hue = color_hsv[0]
        sat = color_hsv[1]
        val = color_hsv[2]

        h_lower = hue - self.h_low
        h_upper = hue + self.h_high

        s_lower = max(0, sat - self.s_low)
        s_upper = min(255, sat + self.s_high)

        v_lower = max(0, val - self.v_low)
        v_upper = min(255, val + self.v_high)

        if h_lower <= 0 and h_upper >= 179:
            # Entire hue spectrum
            lower = np.array([0, s_lower, v_lower])
            upper = np.array([179, s_upper, v_upper])

            self.get_logger().debug(
                f"HSV mask | ALL H | S {s_lower}:{s_upper} | V {v_lower}:{v_upper}"
            )

            mask = cv2.inRange(img_hsv, lower, upper)

        elif h_lower < 0:
            # Wraps through 0
            lower_1 = np.array([0, s_lower, v_lower])
            upper_1 = np.array([h_upper, s_upper, v_upper])

            lower_2 = np.array([180 + h_lower, s_lower, v_lower])
            upper_2 = np.array([179, s_upper, v_upper])

            self.get_logger().debug(
                f"HSV mask | H {180 + h_lower}:179 + 0:{h_upper} | "
                f"S {s_lower}:{s_upper} | V {v_lower}:{v_upper}"
            )

            mask1 = cv2.inRange(img_hsv, lower_1, upper_1)
            mask2 = cv2.inRange(img_hsv, lower_2, upper_2)

            mask = cv2.bitwise_or(mask1, mask2)

        elif h_upper > 179:
            # Wraps through 179 -> 0
            lower_1 = np.array([h_lower, s_lower, v_lower])
            upper_1 = np.array([179, s_upper, v_upper])

            lower_2 = np.array([0, s_lower, v_lower])
            upper_2 = np.array([h_upper - 180, s_upper, v_upper])

            self.get_logger().debug(
                f"HSV mask | H {h_lower}:179 + 0:{h_upper - 180} | "
                f"S {s_lower}:{s_upper} | V {v_lower}:{v_upper}"
            )

            mask1 = cv2.inRange(img_hsv, lower_1, upper_1)
            mask2 = cv2.inRange(img_hsv, lower_2, upper_2)

            mask = cv2.bitwise_or(mask1, mask2)

        else:
            # Normal, non-wrapping range
            lower = np.array([h_lower, s_lower, v_lower])
            upper = np.array([h_upper, s_upper, v_upper])

            self.get_logger().debug(
                f"HSV mask | H {h_lower}:{h_upper} | "
                f"S {s_lower}:{s_upper} | V {v_lower}:{v_upper}"
            )

            mask = cv2.inRange(img_hsv, lower, upper)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel)
        return mask

    def hsv_parameter_callback(self, p: rclpy.parameter.Parameter) -> None:
        value = rclpy.parameter.parameter_value_to_python(p.value)

        self.get_logger().info(f"Received an update to parameter: {p.name}: {value}")

        setattr(self, p.name, value)

    def _setup_parameters(self):
        p = partial(gridbot.declare_param, self)

        # Topics
        self.camera_topic = p(
            "camera_topic",
            "camera/image_raw",
            "Topic used to receive camera images.",
        )
        self.line_offset_topic = p(
            "line_offset_topic", "line_offset", "Topic used to send line_offset."
        )
        self.tracked_image_topic = p(
            "tracked_image_topic",
            "tracked_image",
            "Topic used to send images with additional visuals.",
        )
        self.prepare_tracked_image = p(
            "prepare_tracked_image",
            False,
            "Whether to build and publish the tracked image visualization.",
        )
        self.aruco_topic = p(
            "aruco_topic", "aruco", "Topic used to send aruco ID + direction."
        )
        self.line_turning_topic = p(
            "line_turning_topic",
            "line_turning",
            "Topic used to send information about turning and line visibility.",
        )

        line_color_hsv = p(
            "line_color_hsv",
            [0, 0, 0],
            "HSV color of the line to target (OpenCV scale: H 0-179, S 0-255, V 0-255).",
        )
        if isinstance(line_color_hsv, (list, tuple)):
            hsv = (list(line_color_hsv) + [0, 0, 0])[:3]
            self.line_color_hsv = [
                int(np.clip(hsv[0], 0, 179)),
                int(np.clip(hsv[1], 0, 255)),
                int(np.clip(hsv[2], 0, 255)),
            ]
        else:
            self.line_color_hsv = [0, 255, 255]

        # ROIs
        self.line_roi_height = p(
            "line_roi_height",
            0.5,
            "Height of the line search ROI - [0..1] factor.",
            min_value=0.0,
            max_value=1.0,
        )
        self.centerline_roi_width = p(
            "centerline_roi_width",
            0.4,
            "Width of the central line search ROI - [0..1] factor.",
            min_value=0.0,
            max_value=1.0,
        )
        self.aruco_roi_width = p(
            "aruco_roi_width",
            0.5,
            "Width of the ArUco search ROI - [0..1] factor.",
            min_value=0.0,
            max_value=1.0,
        )
        self.aruco_roi_height = p(
            "aruco_roi_height",
            0.5,
            "Height of the ArUco search ROI - [0..1] factor.",
            min_value=0.0,
            max_value=1.0,
        )
        self.aruco_padding = p(
            "aruco_padding",
            5,
            'Offset from the bottom of the frame where the node will count a found ArUco code as "Passed by".',
        )

        # Line tracking
        self.line_turning_min_angle = p(
            "line_turning_min_angle",
            60.0,
            "Minimum bounding-box angle (degrees) for a line to count as vertical enough during turning.",
            min_value=0.0,
            max_value=90.0,
        )
        self.centerline_step = p(
            "centerline_step",
            10,
            "Pixel rows skipped when looking for centerline nodes.",
            min_value=1,
            max_value=2**31 - 1,
        )
        self.centerline_node_offset = p(
            "centerline_node_offset",
            4,
            "Which node from the bottom will be targeted.",
            min_value=0,
            max_value=2**31 - 1,
        )

        # HSV thresholds
        self.h_low = p(
            "h_thresh_low",
            0,
            "Lower bound of the hue threshold for line detection.",
            min_value=0,
            max_value=179,
        )
        self.h_high = p(
            "h_thresh_high",
            0,
            "Upper bound of the hue threshold for line detection.",
            min_value=0,
            max_value=179,
        )
        self.s_low = p(
            "s_low",
            0,
            "Lower bound of the saturation threshold for line detection.",
            min_value=0,
            max_value=255,
        )
        self.s_high = p(
            "s_high",
            0,
            "Upper bound of the saturation threshold for line detection.",
            min_value=0,
            max_value=255,
        )
        self.v_low = p(
            "v_low",
            0,
            "Lower bound of the value threshold for line detection.",
            min_value=0,
            max_value=255,
        )
        self.v_high = p(
            "v_high",
            0,
            "Upper bound of the value threshold for line detection.",
            min_value=0,
            max_value=255,
        )

        gridbot.log_params(self)


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
