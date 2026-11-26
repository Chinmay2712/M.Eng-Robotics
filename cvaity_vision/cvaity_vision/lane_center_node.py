"""lane_center_node.py - simulated car sees two white lanes on a black street.

Subscribes : camera_image   (sensor_msgs/Image)   any camera that publishes an Image
Publishes  : lane_image     (sensor_msgs/Image)   picture with car marker and steering arrow
             lane_offset    (std_msgs/Float32)    (lane_center_x - car_center_x) / (width / 2), -1 ... 1
                                                  positive = lane middle is to the right = steer right;
                                                  NaN when fewer than two lanes are visible
             lane_found     (std_msgs/Bool)       True if two lanes were found

Parameters (change while running with  ros2 param set /lane_center_node <name> <value>):
  image_topic   topic to listen to (default camera_image; set at start only)
  show_window   show the OpenCV window (default true; needs a desktop, not ssh)
  thresh        grey level for "white"; 0 = automatic (default)
  peak_frac     automatic threshold = this x brightness of the brightest 1 % (default 0.93)
  car_y_frac    height of the car marker, 0 = top ... 1 = bottom (default 0.82)
  max_width_frac, min_height_frac, min_sep_frac   see vision_core.LANE_PARAMS
"""
import math

import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge

from cvaity_vision import vision_core

WINDOW = 'Midlane Street Detection'


class LaneCenterNode(Node):
    def __init__(self):
        super().__init__('lane_center_node')
        self.declare_parameter('image_topic', 'camera_image')
        self.declare_parameter('show_window', True)
        self.declare_parameter('thresh', 0)
        self.declare_parameter('peak_frac', 0.93)
        self.declare_parameter('car_y_frac', 0.82)
        self.declare_parameter('max_width_frac', 0.14)
        self.declare_parameter('min_height_frac', 0.30)
        self.declare_parameter('min_sep_frac', 0.12)

        self.br = CvBridge()
        topic = self.get_parameter('image_topic').value
        self.sub = self.create_subscription(Image, topic, self.image_callback, 10)
        self.pub_image = self.create_publisher(Image, 'lane_image', 10)
        self.pub_offset = self.create_publisher(Float32, 'lane_offset', 10)
        self.pub_found = self.create_publisher(Bool, 'lane_found', 10)
        self.show = bool(self.get_parameter('show_window').value)
        self.get_logger().info('Listening to "%s"; publishing lane_image, lane_offset, lane_found' % topic)

    def current_params(self):
        g = lambda n: self.get_parameter(n).value
        return {'thresh': int(g('thresh')), 'peak_frac': float(g('peak_frac')),
                'car_y_frac': float(g('car_y_frac')), 'max_width_frac': float(g('max_width_frac')),
                'min_height_frac': float(g('min_height_frac')), 'min_sep_frac': float(g('min_sep_frac'))}

    def image_callback(self, msg):
        frame = self.br.imgmsg_to_cv2(msg, 'bgr8')
        res = vision_core.detect_lane(frame, self.current_params())
        out = vision_core.draw_lane(frame, res)

        offset = Float32()
        offset.data = float(res['offset_norm']) if res['ok'] else float('nan')
        found = Bool()
        found.data = bool(res['ok'])
        self.pub_offset.publish(offset)
        self.pub_found.publish(found)
        out_msg = self.br.cv2_to_imgmsg(out, encoding='bgr8')
        out_msg.header = msg.header
        self.pub_image.publish(out_msg)

        if res['ok']:
            self.get_logger().info('lane_center_x=%.0f car_center_x=%.0f steering offset=%+.2f' % (
                res['lane_center_x'], res['car'][0], res['offset_norm']), throttle_duration_sec=1.0)
        else:
            self.get_logger().warning('WARNING: fewer than two lanes visible (%d found)' % len(res['stripes']),
                                      throttle_duration_sec=1.0)

        if self.show:
            try:
                cv2.imshow(WINDOW, out)
                cv2.waitKey(1)
            except cv2.error:
                self.get_logger().warning('Cannot open a window here (no desktop?). Use rqt_image_view on '
                                          'lane_image instead, or start with -p show_window:=false')
                self.show = False


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
