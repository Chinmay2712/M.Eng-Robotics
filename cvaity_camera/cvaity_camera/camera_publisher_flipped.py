import time

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# Defaults; change at start with  --ros-args -p out_width:=320 -p out_height:=240
#   width, height          size the CAMERA captures. Keep 640 x 480: smaller camera modes CROP the picture
#                          (narrower view), they do not shrink it.
#   out_width, out_height  size that is PUBLISHED; the full 640 x 480 view is shrunk to this (0 = no shrinking).
#                          Smaller = faster for the vision nodes, same field of view.
WIDTH = 640            # requested image size
HEIGHT = 480
FPS = 30               # requested camera frame rate
TIMER_PERIOD = 0.005   # [s] the read below waits for the next camera frame, so the real publish rate is the camera's (30 Hz)
MAX_FAILS = 30         # consecutive failed reads before the node shuts down (about 3 s)


class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, 'camera_image', 1)     # depth 1: a slow subscriber gets the newest picture, no queue
        self.declare_parameter('width', WIDTH)
        self.declare_parameter('height', HEIGHT)
        self.declare_parameter('fps', FPS)
        self.declare_parameter('out_width', 0)
        self.declare_parameter('out_height', 0)
        self.declare_parameter('device', 0)
        self.br = CvBridge()
        self.fail_count = 0

        # V4L2 backend: the normal one for a USB camera (avoids the GStreamer warning)
        self.cap = cv2.VideoCapture(int(self.get_parameter('device').value), cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error('Could not open videostream')
            exit(1)

        # image size; buffer 1 = always the newest frame
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.get_parameter('width').value))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.get_parameter('height').value))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, int(self.get_parameter('fps').value))
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.get_logger().info('Camera opened, image size %d x %d, %.0f fps (camera may use the nearest size it supports)' % (w, h, fps))

        ow, oh = int(self.get_parameter('out_width').value), int(self.get_parameter('out_height').value)
        self.out_size = (ow, oh) if ow > 0 and oh > 0 and (ow, oh) != (w, h) else None
        if self.out_size:
            self.get_logger().info('Publishing shrunk to %d x %d (same field of view)' % self.out_size)

        self.timer = self.create_timer(TIMER_PERIOD, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        stamp = self.get_clock().now().to_msg()      # time right after the capture
        if ret:
            self.fail_count = 0
            if self.out_size:
                frame = cv2.resize(frame, self.out_size, interpolation=cv2.INTER_AREA)
            msg = self.br.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = stamp
            msg.header.frame_id = 'camera'
            self.publisher_.publish(msg)
        else:
            self.fail_count += 1
            time.sleep(0.1)                          # a failed read returns at once: wait, so 30 failures take about 3 s
            if self.fail_count == 1 or self.fail_count % 10 == 0:
                self.get_logger().warning(
                    'Error in capturing image (%d in a row)' % self.fail_count)
            if self.fail_count >= MAX_FAILS:
                self.get_logger().error('Camera lost - shutting down')
                rclpy.shutdown()

    def destroy_node(self):
        super().destroy_node()
        self.cap.release()


def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CameraPublisher()
    try:
        rclpy.spin(camera_publisher)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    camera_publisher.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
