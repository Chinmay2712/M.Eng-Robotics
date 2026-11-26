"""lane_center_node_flipped.py - simulated car sees two white lanes on a black street.

Subscribes : camera_image   (sensor_msgs/Image)   any camera that publishes an Image
Publishes  : lane_image     (sensor_msgs/Image)   picture with car marker and steering arrow
             lane_offset    (std_msgs/Float32)    (lane_center_x - car_center_x) / (width / 2), -1 ... 1
                                                  positive = lane middle is to the right = steer right;
                                                  NaN only when no lane line is visible at all
             lane_found     (std_msgs/Bool)       True if the car is inside the lane (no WARNING)

WARNING (as in the demo) when the car body is outside the lane or touches a line, or no line is visible.
A line that is out of the picture is estimated from the other line and the lane width (orange ring).

Parameters (change while running with  ros2 param set /lane_center_node <name> <value>):
  rotate180     turn the picture by 180 degrees first (camera mounted upside down; default true)
  publish_image publish lane_image (default true; switch off for more speed)
  show_window   show the OpenCV window (default true; needs a desktop, not ssh)
  image_topic   topic to listen to (default camera_image; set at start only)
  method        'tophat' (default): tape = narrow and brighter than the floor next to it; adapts by itself to
                exposure and lighting. 'global': the old single grey level (uses thresh and peak_frac)
  min_contrast  tophat: tape must be at least this much brighter than the floor next to it (default 30)
  tophat_width_frac  tophat: bright things wider than this (fraction of W) count as background (default 0.18)
  thresh        global only: grey level for "white"; 0 = automatic
  peak_frac     global only: automatic threshold = this x brightness of the brightest 1 % (default 0.93)
  car_y_frac    height of the car marker, 0 = top ... 1 = bottom (default 0.82)
  car_half_width_frac  half width of the car marker (default 0.065 of the width)
  lane_width_frac      lane width used before both lines have been seen once (default 0.70)
  show_mask     second window with the black-and-white picture the detection works on (default false)
  max_width_frac, min_height_frac, min_sep_frac, max_border_frac   see vision_core_flipped.LANE_PARAMS

The node only works on the newest picture (queue depth 1): if it is slower than the camera it skips
pictures instead of falling behind, so the display does not lag. Once per second it logs the frame
rate and the processing time per frame.
"""
import time

import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge

from cvaity_vision import vision_core_flipped as vision_core

WINDOW = 'Midlane Street Detection'


class LaneCenterNode(Node):
    def __init__(self):
        super().__init__('lane_center_node')
        self.declare_parameter('image_topic', 'camera_image')
        self.declare_parameter('rotate180', True)
        self.declare_parameter('publish_image', True)
        self.declare_parameter('show_window', True)
        self.declare_parameter('method', 'tophat')
        self.declare_parameter('min_contrast', 30)
        self.declare_parameter('contrast_frac', 0.45)
        self.declare_parameter('tophat_width_frac', 0.18)
        self.declare_parameter('thresh', 0)
        self.declare_parameter('peak_frac', 0.93)
        self.declare_parameter('car_y_frac', 0.82)
        self.declare_parameter('max_width_frac', 0.14)
        self.declare_parameter('min_height_frac', 0.30)
        self.declare_parameter('min_sep_frac', 0.12)
        self.declare_parameter('max_border_frac', 0.6)
        self.declare_parameter('car_half_width_frac', 0.065)
        self.declare_parameter('lane_width_frac', 0.70)
        self.declare_parameter('show_mask', False)

        self.br = CvBridge()
        self.meter = vision_core.RateMeter()
        self.lane_state = vision_core.new_lane_state()   # remembers the lane between pictures
        topic = self.get_parameter('image_topic').value
        self.sub = self.create_subscription(Image, topic, self.image_callback, 1)     # depth 1: newest picture only
        self.pub_image = self.create_publisher(Image, 'lane_image', 1)
        self.pub_offset = self.create_publisher(Float32, 'lane_offset', 10)
        self.pub_found = self.create_publisher(Bool, 'lane_found', 10)
        self.show = bool(self.get_parameter('show_window').value)
        self.get_logger().info('Listening to "%s"; publishing lane_image, lane_offset, lane_found' % topic)

    def current_params(self):
        g = lambda n: self.get_parameter(n).value
        return {'method': str(g('method')), 'min_contrast': float(g('min_contrast')),
                'contrast_frac': float(g('contrast_frac')), 'tophat_width_frac': float(g('tophat_width_frac')),
                'thresh': int(g('thresh')), 'peak_frac': float(g('peak_frac')),
                'car_y_frac': float(g('car_y_frac')), 'max_width_frac': float(g('max_width_frac')),
                'min_height_frac': float(g('min_height_frac')), 'min_sep_frac': float(g('min_sep_frac')),
                'max_border_frac': float(g('max_border_frac')), 'car_half_width_frac': float(g('car_half_width_frac')),
                'lane_width_frac': float(g('lane_width_frac'))}

    def image_callback(self, msg):
        t_start = time.perf_counter()
        frame = self.br.imgmsg_to_cv2(msg, 'bgr8')
        if self.get_parameter('rotate180').value:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        res = vision_core.detect_lane(frame, self.current_params(), self.lane_state)
        out = vision_core.draw_lane(frame, res)

        offset = Float32()
        offset.data = float(res['offset_norm']) if res['offset_norm'] is not None else float('nan')
        found = Bool()
        found.data = bool(res['ok'])
        self.pub_offset.publish(offset)
        self.pub_found.publish(found)
        if self.get_parameter('publish_image').value:
            out_msg = self.br.cv2_to_imgmsg(out, encoding='bgr8')
            out_msg.header = msg.header
            self.pub_image.publish(out_msg)

        seen = 'both lines' if (res['seen_left'] and res['seen_right']) else 'one line (other estimated)'
        if res['status'] == 'inside':
            self.get_logger().info('lane_center_x=%.0f car_center_x=%.0f steering offset=%+.2f, %s' % (
                res['lane_center_x'], res['car'][0], res['offset_norm'], seen), throttle_duration_sec=1.0)
        elif res['status'] == 'outside':
            self.get_logger().warning('WARNING: car outside the lane (lane %.0f ... %.0f, car %.0f), %s' % (
                res['left_x'], res['right_x'], res['car'][0], seen), throttle_duration_sec=1.0)
        else:
            self.get_logger().warning('WARNING: no lane line visible', throttle_duration_sec=1.0)

        if self.show:
            try:
                cv2.imshow(WINDOW, out)
                if self.get_parameter('show_mask').value:
                    cv2.imshow(WINDOW + ' - mask', res['mask'])
                cv2.waitKey(1)
            except cv2.error:
                self.get_logger().warning('Cannot open a window here (no desktop?). Use rqt_image_view on '
                                          'lane_image instead, or start with -p show_window:=false')
                self.show = False

        report = self.meter.tick(time.perf_counter() - t_start)
        if report:
            self.get_logger().info(report)


def main(args=None):
    rclpy.init(args=args)
    node = LaneCenterNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
