"""image_file_publisher.py - stand-in for the camera: publishes pictures from a file, a folder or a video.

Use it to test the vision nodes without the camera. It publishes on the same topic as the camera node,
so the other nodes cannot tell the difference. Do not run it together with camera_publisher.

  ros2 run cvaity_vision image_file_publisher --ros-args -p path:=/home/pi/test_images
  ros2 run cvaity_vision image_file_publisher --ros-args -p path:=/home/pi/test_images/pdf_lane_2.png
  ros2 run cvaity_vision image_file_publisher --ros-args -p path:=/home/pi/clip.mp4

Parameters:
  path               picture file, folder with pictures (png, jpg), or video file (mp4, avi, mkv)   [required]
  topic              default camera_image
  rate               pictures per second (default 10.0)
  seconds_per_image  how long each picture of a folder stays (default 4.0)
  width, height      pictures are scaled to this size (default 640 x 480)
"""
import os

import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.bmp')
VIDEO_EXT = ('.mp4', '.avi', '.mkv', '.mov')


class ImageFilePublisher(Node):
    def __init__(self):
        super().__init__('image_file_publisher')
        self.declare_parameter('path', '')
        self.declare_parameter('topic', 'camera_image')
        self.declare_parameter('rate', 10.0)
        self.declare_parameter('seconds_per_image', 4.0)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)

        path = os.path.expanduser(str(self.get_parameter('path').value))
        self.size = (int(self.get_parameter('width').value), int(self.get_parameter('height').value))
        self.br = CvBridge()
        self.video = None
        self.images = []
        self.index = 0
        self.ticks = 0
        rate = float(self.get_parameter('rate').value)
        self.ticks_per_image = max(1, int(float(self.get_parameter('seconds_per_image').value) * rate))

        if os.path.isdir(path):
            names = sorted(f for f in os.listdir(path) if f.lower().endswith(IMAGE_EXT))
            self.names = [os.path.join(path, f) for f in names]
        elif path.lower().endswith(VIDEO_EXT) and os.path.isfile(path):
            self.video = cv2.VideoCapture(path)
            self.names = [path]
        elif os.path.isfile(path):
            self.names = [path]
        else:
            self.names = []
        if not self.names or (self.video is not None and not self.video.isOpened()):
            self.get_logger().error('No pictures found. Start with  -p path:=<file, folder or video>  (got "%s")' % path)
            exit(1)

        if self.video is None:
            for f in self.names:
                im = cv2.imread(f)
                if im is None:
                    self.get_logger().warning('Cannot read %s, skipped' % f)
                    continue
                self.images.append((os.path.basename(f), cv2.resize(im, self.size)))
            if not self.images:
                self.get_logger().error('None of the files could be read')
                exit(1)
            self.get_logger().info('%d picture(s) loaded, %.1f s each' % (
                len(self.images), self.ticks_per_image / rate))
        else:
            self.get_logger().info('Playing video %s' % path)

        self.pub = self.create_publisher(Image, str(self.get_parameter('topic').value), 10)
        self.timer = self.create_timer(1.0 / rate, self.timer_callback)

    def next_frame(self):
        if self.video is not None:
            ok, frame = self.video.read()
            if not ok:                                  # end of the video: start again
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.video.read()
                if not ok:
                    return None, ''
            return cv2.resize(frame, self.size), os.path.basename(self.names[0])
        name, frame = self.images[self.index]
        self.ticks += 1
        if self.ticks >= self.ticks_per_image:
            self.ticks = 0
            self.index = (self.index + 1) % len(self.images)
            self.get_logger().info('now showing: %s' % self.images[self.index][0])
        return frame, name

    def timer_callback(self):
        frame, _ = self.next_frame()
        if frame is None:
            return
        msg = self.br.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        self.pub.publish(msg)

    def destroy_node(self):
        if self.video is not None:
            self.video.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImageFilePublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
