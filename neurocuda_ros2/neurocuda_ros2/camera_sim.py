#!/usr/bin/env python3
"""
Camera Simulator Node — publishes test images for SNN inference.

Generates synthetic images (CIFAR-10-like patterns) using pure numpy
(zero external dependencies beyond numpy). No cv2, no PIL, no hardware.

Patterns mimic real CIFAR-10 objects:
  - airplane: horizontal fuselage + wings
  - car: body + roof + wheels
  - bird: body ellipse + head + beak
  - cat/dog: animal blob + ears
  - ship: hull + mast + sail
  - checkerboard, gradient, circle: geometric patterns

Usage:
  ros2 run neurocuda_ros2 camera_sim --ros-args -p pattern:=airplane
  ros2 run neurocuda_ros2 camera_sim --ros-args -p pattern:=random rate:=10.0
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import numpy as np
import random


# ===================================================================
# Pure-numpy drawing primitives (no cv2/PIL dependency)
# ===================================================================

def _draw_rect(img, x1, y1, x2, y2, color):
    """Fill a rectangle [x1:x2, y1:y2] with color."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x1 < x2 and y1 < y2:
        img[y1:y2, x1:x2] = color


def _draw_filled_circle(img, cx, cy, r, color):
    """Fill a circle at (cx,cy) with radius r using vectorized numpy."""
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    img[mask] = color


def _draw_filled_ellipse(img, cx, cy, rx, ry, color):
    """Fill an ellipse using vectorized numpy."""
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    mask = ((x - cx) ** 2) / (max(rx, 1) ** 2) + ((y - cy) ** 2) / (max(ry, 1) ** 2) <= 1.0
    img[mask] = color


def _draw_filled_triangle(img, pts, color):
    """Fill a triangle given 3 (x,y) points using barycentric coordinates."""
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    p0, p1, p2 = pts
    # Compute barycentric using area method
    denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
    if abs(denom) < 1:
        return
    a = ((p1[1] - p2[1]) * (x - p2[0]) + (p2[0] - p1[0]) * (y - p2[1])) / denom
    b = ((p2[1] - p0[1]) * (x - p2[0]) + (p0[0] - p2[0]) * (y - p2[1])) / denom
    c = 1 - a - b
    mask = (a >= 0) & (b >= 0) & (c >= 0)
    img[mask] = color


# ===================================================================
# Pattern generators
# ===================================================================

def _draw_airplane(h, w):
    img = np.ones((h, w, 3), dtype=np.uint8) * 128
    _draw_rect(img, w//4, h//3, 3*w//4, 2*h//3, (200, 200, 255))
    _draw_rect(img, w//6, h//5, 5*w//6, h//3, (180, 180, 240))
    _draw_rect(img, w//3, 2*h//3, 2*w//3, 3*h//4, (160, 160, 220))
    return img


def _draw_car(h, w):
    img = np.ones((h, w, 3), dtype=np.uint8) * 128
    _draw_rect(img, w//8, h//3, 7*w//8, 2*h//3, (255, 100, 100))
    _draw_rect(img, w//4, h//5, 3*w//4, h//3, (200, 80, 80))
    _draw_filled_circle(img, w//5, 3*h//4, w//10, (50, 50, 50))
    _draw_filled_circle(img, 4*w//5, 3*h//4, w//10, (50, 50, 50))
    return img


def _draw_bird(h, w):
    img = np.ones((h, w, 3), dtype=np.uint8) * 128
    _draw_filled_ellipse(img, w//2, h//2, w//6, h//4, (100, 255, 100))
    _draw_filled_circle(img, 2*w//3, h//3, w//10, (80, 220, 80))
    _draw_filled_ellipse(img, 3*w//4, h//3, w//12, h//16, (255, 200, 50))
    return img


def _draw_animal(h, w, kind):
    img = np.ones((h, w, 3), dtype=np.uint8) * 128
    color = (100, 100, 255) if kind == "cat" else (255, 150, 100)
    _draw_filled_ellipse(img, w//2, h//2, w//4, h//3, color)
    _draw_filled_circle(img, 3*w//4, h//3, w//6, color)
    _draw_filled_circle(img, 5*w//6, h//4 - h//12, w//20, (255, 255, 255))
    _draw_filled_circle(img, 5*w//6, h//4 - h//12, w//30, (0, 0, 0))
    if kind == "cat":
        pts = [(3*w//4 - w//12, h//6), (3*w//4, h//12), (3*w//4 + w//12, h//6)]
        _draw_filled_triangle(img, pts, color)
    return img


def _draw_ship(h, w):
    img = np.ones((h, w, 3), dtype=np.uint8) * 128
    pts = [(w//6, h//3), (5*w//6, h//3), (4*w//6, 2*h//3), (w//6, 2*h//3)]
    from itertools import combinations
    # Fill hull as a union of two triangles
    _draw_filled_triangle(img, (pts[0], pts[1], pts[2]), (255, 200, 100))
    _draw_filled_triangle(img, (pts[0], pts[2], pts[3]), (255, 200, 100))
    _draw_rect(img, w//3, h//6, w//3 + max(w//30, 1), h//3, (100, 80, 60))
    sail_pts = [(w//3 + max(w//30, 1), h//6), (w//3 + max(w//30, 1), h//3), (2*w//3, h//4)]
    _draw_filled_triangle(img, sail_pts, (255, 255, 255))
    return img


def _draw_checkerboard(h, w):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    sq = max(h, w) // 4
    for i in range(0, h, sq):
        for j in range(0, w, sq):
            if (i // sq + j // sq) % 2 == 0:
                _draw_rect(img, j, i, j + sq, i + sq, (200, 200, 200))
            else:
                _draw_rect(img, j, i, j + sq, i + sq, (50, 50, 50))
    return img


def _draw_gradient(h, w):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        val = int(255 * i / h)
        img[i, :] = [val, val, val]
    _draw_filled_circle(img, w//2, h//2, w//6, (255, 0, 0))
    return img


def _draw_circle(h, w):
    img = np.ones((h, w, 3), dtype=np.uint8) * 200
    _draw_filled_circle(img, w//2, h//2, min(h, w)//3, (50, 50, 255))
    return img


PATTERNS = {
    "airplane": _draw_airplane,
    "car": _draw_car,
    "bird": _draw_bird,
    "cat": lambda h, w: _draw_animal(h, w, "cat"),
    "dog": lambda h, w: _draw_animal(h, w, "dog"),
    "ship": _draw_ship,
    "checkerboard": _draw_checkerboard,
    "gradient": _draw_gradient,
    "circle": _draw_circle,
}


def np_to_ros_image(np_img, frame_id="camera_sim"):
    """Convert numpy (H, W, 3) uint8 RGB to sensor_msgs/Image."""
    msg = Image()
    msg.header = Header()
    msg.header.stamp.sec = 0
    msg.header.stamp.nanosec = 0
    msg.header.frame_id = frame_id
    msg.height = np_img.shape[0]
    msg.width = np_img.shape[1]
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = 3 * np_img.shape[1]
    msg.data = np_img.tobytes()
    return msg


class CameraSimNode(Node):
    """Publishes synthetic test images to /camera/image."""

    def __init__(self):
        super().__init__("camera_sim")
        self.declare_parameter("pattern", "random")
        self.declare_parameter("rate", 2.0)
        self.declare_parameter("resolution", [64, 64])
        self.declare_parameter("seed", 42)

        self._pattern = self.get_parameter("pattern").value
        self._rate = self.get_parameter("rate").value
        self._res = self.get_parameter("resolution").value
        seed = self.get_parameter("seed").value

        np.random.seed(seed)
        random.seed(seed)

        self._pub = self.create_publisher(Image, "/camera/image", 10)
        self._timer = self.create_timer(1.0 / self._rate, self._publish_image)
        self._pattern_names = list(PATTERNS.keys())
        self._counter = 0

        self.get_logger().info(
            f"Camera Sim ready | Pattern: {self._pattern} | "
            f"Rate: {self._rate} Hz | Resolution: {self._res}"
        )

    def _publish_image(self):
        h, w = self._res
        pattern = self._pattern

        if pattern == "random":
            pattern = random.choice(self._pattern_names)
            if self._counter % 5 == 0:
                self.get_logger().info(
                    f"  Pattern: {pattern} (image #{self._counter})"
                )

        draw_fn = PATTERNS.get(pattern, _draw_checkerboard)
        np_img = draw_fn(h, w)

        msg = np_to_ros_image(np_img)
        msg.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(msg)
        self._counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = CameraSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
