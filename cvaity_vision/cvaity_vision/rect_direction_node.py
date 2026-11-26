"""rect_direction_node.py - white paper rectangle on black background: draw a line along it.

Subscribes : camera_image    (sensor_msgs/Image)
Publishes  : rect_image      (sensor_msgs/Image)  picture with the red line along the rectangle
             rect_angle      (std_msgs/Float32)   direction of the long axis in degrees,
                                                  0 = right, 90 = up, 0 <= angle < 180; NaN if not found

Parameters (change while running with  ros2 param set /rect_direction_node <name> <value>):
  image_topic   topic to listen to (default camera_image; set at start only)
  show_window   show the OpenCV window (default true; needs a desktop, not ssh)
  thresh        grey level for "white"; 0 = automatic (default)
  peak_frac     automatic threshold = this x brightness of the brightest 1 % (default 0.93)
  min_area_frac smallest accepted shape, fraction of the picture (default 0.01)
  arrow         draw an arrow head too (default false); head at the right end (upper end if vertical)
"""
import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge

from cvaity_vision import vision_core

WINDOW = 'Arrow Direction'


class RectDirectionNode(Node):
    def __init__(self):
        super().__init__('rect_direction_node')
        self.declare_parameter('image_topic', 'camera_image')
        self.declare_parameter('show_window', True)
        self.declare_parameter('thresh', 0)
        self.declare_parameter('peak_frac', 0.93)
        self.declare_parameter('min_area_frac', 0.01)
        self.declare_parameter('arrow', False)

        self.br = CvBridge()
        topic = self.get_parameter('image_topic').value
        self.sub = self.create_subscription(Image, topic, self.image_callback, 10)
        self.pub_image = self.create_publisher(Image, 'rect_image', 10)
        self.pub_angle = self.create_publisher(Float32, 'rect_angle', 10)
        self.show = bool(self.get_parameter('show_window').value)
        self.get_logger().info('Listening to "%s"; publishing rect_image, rect_angle' % topic)

    def current_params(self):
        g = lambda n: self.get_parameter(n).value
        return {'thresh': int(g('thresh')), 'peak_frac': float(g('peak_frac')),
                'min_area_frac': float(g('min_area_frac'))}

    def image_callback(self, msg):
        frame = self.br.imgmsg_to_cv2(msg, 'bgr8')
        res = vision_core.detect_rect(frame, self.current_params())
        out = vision_core.draw_rect(frame, res, arrow=bool(self.get_parameter('arrow').value))

        angle = Float32()
        angle.data = float(res['angle_deg']) if res['ok'] else float('nan')
        self.pub_angle.publish(angle)
        out_msg = self.br.cv2_to_imgmsg(out, encoding='bgr8')
        out_msg.header = msg.header
        self.pub_image.publish(out_msg)

        if res['ok']:
            self.get_logger().info('rectangle angle %.0f deg, centre (%.0f, %.0f)' % (
                res['angle_deg'], res['center'][0], res['center'][1]), throttle_duration_sec=1.0)
        else:
            self.get_logger().warning('WARNING: no rectangle with a clear direction', throttle_duration_sec=1.0)

        if self.show:
            try:
                cv2.imshow(WINDOW, out)
                cv2.waitKey(1)
            except cv2.error:
                self.get_logger().warning('Cannot open a window here (no desktop?). Use rqt_image_view on '
                                          'rect_image instead, or start with -p show_window:=false')
                self.show = False


def main(args=None):
    rclpy.init(args=args)
    node = RectDirectionNode()
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
