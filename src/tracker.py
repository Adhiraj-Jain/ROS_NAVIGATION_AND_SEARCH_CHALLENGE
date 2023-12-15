#! /usr/bin/env python

from typing import NoReturn

import rospy
import tf2_geometry_msgs
import tf2_ros

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Quaternion, Point
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class TrackRobotPath:
    def __init__(self) -> NoReturn:
        rospy.loginfo("Initializing TrackRobotPath...")
        self.node = rospy.init_node("search_nav_challenge:robot_position")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.path_pub = rospy.Publisher("/path", Path, queue_size=50)
        self.current_path_model = Path()

    def perform_pose_transform(self, msg):
        transform = self.tf_buffer.lookup_transform(
            "map", msg.header.frame_id, msg.header.stamp, rospy.Duration(1.0)
        )
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = msg.header.stamp
        pose.pose.position.x = msg.pose.pose.position.x
        pose.pose.position.y = msg.pose.pose.position.y
        pose.pose.position.z = msg.pose.pose.position.z
        pose.pose.orientation.w = msg.pose.pose.orientation.w
        pose_transformed = tf2_geometry_msgs.do_transform_pose(pose, transform)
        return pose_transformed

    def extract_position_and_orientation(self, pose_transformed):
        x = pose_transformed.pose.position.x
        y = pose_transformed.pose.position.y
        z = pose_transformed.pose.position.z
        _, _, yaw = euler_from_quaternion(
            [
                pose_transformed.pose.orientation.x,
                pose_transformed.pose.orientation.y,
                pose_transformed.pose.orientation.z,
                pose_transformed.pose.orientation.w,
            ]
        )
        return x, y, z, yaw

    def create_stamped_pose(self, msg, x, y, z, yaw):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = msg.header.stamp
        pose_stamped.header.frame_id = "map"
        pose_stamped.pose.position = Point(x, y, z)
        pose_stamped.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, yaw))
        return pose_stamped

    def did_pose_change(self, last_pose, pose_stamped) -> bool:
        x_resolution = y_resolution = 0.1
        last_x = last_y = None
        if last_pose:
            last_position = last_pose.position
            last_x = last_position.x if last_position else None
            last_y = last_position.y if last_position else None

        is_new_pose = (
            not last_pose
            and not last_x
            and not last_y
            or abs(last_x - pose_stamped.pose.position.x) > x_resolution
            or abs(last_y - pose_stamped.pose.position.y) > y_resolution
        )
        return is_new_pose

    def confirm_orientation(self, s, v):
        v.pose.orientation.w = s.pose.pose.orientation.w
        v.pose.orientation.z = s.pose.pose.orientation.z

    def get_latest_pose_in_model(self, path):
        return path.poses[-1].pose if path.poses else None

    def add_to_path_model(self, msg, path, pose_stamped):
        path.poses.append(pose_stamped)
        path.header.stamp = msg.header.stamp
        path.header.frame_id = "map"

    def odom_callback(self, msg) -> NoReturn:
        try:
            pose_transformed = self.perform_pose_transform(msg)
            x, y, z, yaw = self.extract_position_and_orientation(pose_transformed)
            pose_stamped = self.create_stamped_pose(msg, x, y, z, yaw)
            self.confirm_orientation(msg, pose_stamped)
            last_pose = self.get_latest_pose_in_model(self.current_path_model)
            is_new_pose = self.did_pose_change(last_pose, pose_stamped)
            if is_new_pose:
                self.add_to_path_model(msg, self.current_path_model, pose_stamped)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            rospy.logwarn("Unable to perform transformation to derive a path pose.")

    def execute(self) -> NoReturn:
        rospy.loginfo("Tracking robot path.")
        rate = rospy.Rate(3)
        while not rospy.is_shutdown():
            self.path_pub.publish(self.current_path_model)
            rate.sleep()


def main():
    path_tracker = TrackRobotPath()
    path_tracker.execute()


if __name__ == "__main__":
    main()
