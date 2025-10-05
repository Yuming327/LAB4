import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class ReactiveFollowGap(Node):
    def __init__(self):
        super().__init__('reactive_node')
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.BUBBLE_RADIUS = 0.5
        self.MAX_SPEED = 1.5
        self.MIN_SPEED = 0.5
        self.SAFE_DISTANCE = 1.0

    def preprocess_lidar(self, ranges):
        proc = np.array(ranges)
        proc[np.isinf(proc)] = 3.5
        return np.convolve(proc, np.ones(5)/5, mode='same')

    def find_max_gap(self, ranges):
        mask = ranges > 0
        gaps, start = [], None
        for i, val in enumerate(mask):
            if val and start is None: start = i
            elif not val and start is not None: gaps.append((start, i-1)); start=None
        if start is not None: gaps.append((start, len(mask)-1))
        return max(gaps, key=lambda x: x[1]-x[0])

    def find_best_point(self, start, end, ranges):
        return np.argmax(ranges[start:end+1]) + start

    def lidar_callback(self, data):
        proc = self.preprocess_lidar(data.ranges)
        closest = np.argmin(proc)
        start = max(0, closest - int(self.BUBBLE_RADIUS/data.angle_increment))
        end = min(len(proc)-1, closest + int(self.BUBBLE_RADIUS/data.angle_increment))
        proc[start:end+1] = 0
        s, e = self.find_max_gap(proc)
        best = self.find_best_point(s, e, proc)
        angle = data.angle_min + best * data.angle_increment
        speed = self.MIN_SPEED if np.min(proc) < self.SAFE_DISTANCE else self.MAX_SPEED
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = angle
        msg.drive.speed = speed
        self.drive_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ReactiveFollowGap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
