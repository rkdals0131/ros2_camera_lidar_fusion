import rclpy

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.node import Node 
from yolo_msgs.msg import *
from std_msgs.msg import *
from sensor_msgs.msg import *



class DebuggingPalette(Node):
    def __init__(self):
        super().__init__('debugging_palette')
        
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.create_subscription(DetectionArray, 'detections', self.detection_callback, qos_profile)
        self.get_logger().info("Debugging palette node initialized")

    def detection_callback(self, msg: DetectionArray):
        self.get_logger().info("--------------------------------")
        self.get_logger().info(f"Received {len(msg.detections)} detections")
        for detection in msg.detections:
            self.get_logger().info("***###***")
            self.get_logger().info(f"Detection name: {detection.class_name}")
            self.get_logger().info(f"Detection score: {detection.score}")
            self.get_logger().info(f"Detection center: {detection.bbox.center.position.x}, {detection.bbox.center.position.y}")
            self.get_logger().info(f"Detection size: {detection.bbox.size.x}, {detection.bbox.size.y}")
            self.get_logger().info("***###***")
        self.get_logger().info("-------------------------------")

def main(args=None):
    rclpy.init(args=args)
    debugging_palette = DebuggingPalette()
    rclpy.spin(debugging_palette)
    debugging_palette.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()  