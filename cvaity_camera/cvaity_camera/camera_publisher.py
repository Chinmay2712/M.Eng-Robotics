import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

WIDTH = 640            # requested image size
HEIGHT = 480
TIMER_PERIOD = 0.1     # [s] publish rate 10 Hz
MAX_FAILS = 30         # consecutive failed reads before the node shuts down (about 3 s)


class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, 'camera_image', 10)
        self.br = CvBridge()
        self.fail_count = 0

        # V4L2 backend: the normal one for a USB camera (avoids the GStreamer warning)
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error('Could not open videostream')
            exit(1)

        # image size; buffer 1 = always the newest frame
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info('Camera opened, image size %d x %d' % (w, h))

        self.timer = self.create_timer(TIMER_PERIOD, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        stamp = self.get_clock().now().to_msg()      # time right after the capture
        if ret:
            self.fail_count = 0
            msg = self.br.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = stamp
            msg.header.frame_id = 'camera'
            self.publisher_.publish(msg)
        else:
            self.fail_count += 1
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