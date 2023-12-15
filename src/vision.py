import numpy as np
import math

from typing import Optional

import tf
import rospy

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool


class HazardMarker:
    __HAZARD_TO_PUB_ID = {
        "Unknown": 0,
        "Explosive": 1,
        "Flammable Gas": 2,
        "Non-Flammable Gas": 3,
        "Dangerous When Wet": 4,
        "Flammable Solid": 5,
        "Spontaneously Combustible": 6,
        "Oxidizer": 7,
        "Organic Peroxide": 8,
        "Inhalation Hazard": 9,
        "Poison": 10,
        "Radioactive": 11,
        "Corrosive": 12,
    }

    __HAZARD_TO_OBJECT_ID = {
        "Uncertain": [],
        "Explosive": [6, 16],
        "Flammable Gas": [8, 17],
        "Non-Flammable Gas": [2, 20],
        "Dangerous When Wet": [9, 15],
        "Flammable Solid": [3, 18],
        "Spontaneously Combustible": [1, 25],
        "Oxidizer": [10, 22],
        "Organic Peroxide": [11, 21],
        "Inhalation Hazard": [4, 19],
        "Poison": [5, 23],
        "Radioactive": [7, 24],
        "Corrosive": [13, 14],
    }

    __START_MARKER = 12.0
    __MAX_HAZARD_TO_FIND = 5

    def __init__(self) -> None:
        self.node = rospy.init_node("search_nav_challenge:vision")
        self.marked_hazards = list()
        self.recognition_sub = rospy.Subscriber(
            "/objects", Float32MultiArray, self.recognition_callback
        )
        self.rviz_marker_pub = rospy.Publisher("/hazards", Marker, queue_size=3)
        self.start_pub = rospy.Publisher("/start_challenge", Bool, queue_size=1)
        self.all_hazards_detected_pub = rospy.Publisher(
            "/all_hazards_detected", Bool, queue_size=1
        )

    def __len__(self) -> int:
        return len(self.marked_hazards)

    def recognition_callback(self, msg):
        empty_message = len(msg.data) == 0
        if empty_message:
            return

        if self.is_start_marker(msg.data[0]):
            rospy.loginfo("Start marker detected.")
            return self.start_pub.publish(True)

        labels_in_image = list()
        for label_id in msg.data[::12]:
            labels_in_image.append(self.retrieve_hazard(label_id))

        no_objects_detected = len(labels_in_image) == 0
        if no_objects_detected:
            return

        rospy.loginfo(f"Detected hazards: {labels_in_image}")

        if len(labels_in_image) > 1:
            rospy.logwarn(f"Rolling {labels_in_image[1:]} to forward queue.")
        self.mark_hazard(labels_in_image[0])

        all_hazards_detected = len(self) > self.__MAX_HAZARD_TO_FIND
        if all_hazards_detected:
            self.all_hazards_detected_pub.publish(True)

    def is_start_marker(self, id) -> bool:
        # return id == 5.0
        return id == self.__START_MARKER

    def mark_hazard(self, label):
        if label not in self.marked_hazards:
            rospy.logwarn(f"Marking hazard: {label}")
            hazard_coordinates = self.get_image_location()
            if hazard_coordinates is not None:
                self.place_rviz_marker(hazard_coordinates, label)

    def retrieve_hazard(self, hazard_id) -> str:
        _id = int(hazard_id)
        for hazard, id_map in self.__HAZARD_TO_OBJECT_ID.items():
            if _id in id_map:
                return hazard
        return "Unknown"

    def get_image_location(self) -> Optional[Point]:
        # Get laser and pose data.
        scan_data = rospy.wait_for_message(
            "/scan", LaserScan, timeout=rospy.Duration(5)
        )
        if scan_data is None:
            return rospy.logerr("/scan has timed-out for a possible hazard mapping.")

        # Trim the laser scan data to only include the front 65 degrees.
        # scan_data.ranges = self.trim_scan_data_to_front(scan_data)
        # Find the closest laser.
        closest_range = min(scan_data.ranges)
        closest_index = scan_data.ranges.index(closest_range)
        # Find the angle corresponding to the closest laser.
        closest_angle = scan_data.angle_min + closest_index * scan_data.angle_increment
        # Find the x, y coordinates of the closest point in the laser scanner's coordinate frame.
        x = closest_range * math.cos(closest_angle)
        y = closest_range * math.sin(closest_angle)
        # Transform the x, y coordinates to the robot's coordinate frame.
        listener = tf.TransformListener()
        listener.waitForTransform(
            "base_link", scan_data.header.frame_id, rospy.Time(), rospy.Duration(1.0)
        )
        (trans, rot) = listener.lookupTransform(
            "base_link", scan_data.header.frame_id, rospy.Time(0)
        )
        closest_point_robot = tf.transformations.concatenate_matrices(
            tf.transformations.translation_matrix(trans),
            tf.transformations.quaternion_matrix(rot),
        ).dot([x, y, 0, 1])[0:3]
        # Transform the closest point in the robot's coordinate frame to the map coordinate frame.
        listener.waitForTransform("map", "base_link", rospy.Time(), rospy.Duration(1.0))
        (trans, rot) = listener.lookupTransform("map", "base_link", rospy.Time(0))
        closest_point_map = tf.transformations.concatenate_matrices(
            tf.transformations.translation_matrix(trans),
            tf.transformations.quaternion_matrix(rot),
        ).dot(
            [closest_point_robot[0], closest_point_robot[1], closest_point_robot[2], 1]
        )[
            0:3
        ]

        rospy.loginfo(
            f"Hazard detected at: x={closest_point_map[0]:.2f}, "
            + f"y={closest_point_map[1]:.2f}, z={closest_point_map[2]:.2f}",
        )
        return Point(closest_point_map[0], closest_point_map[1], closest_point_map[2])

    def trim_scan_data_to_front(self, scan_data: LaserScan):
        range_len = len(scan_data.ranges)
        eachDegree = range_len // 360
        return list(
            np.concatenate(
                (
                    np.array(
                        scan_data.ranges[range_len - (23 * eachDegree) : range_len]
                    ),
                    np.array(scan_data.ranges[0 : eachDegree * 23]),
                )
            )
        )

    def place_rviz_marker(self, marker_location: Point, label):
        marker = Marker()
        marker.id = self.__HAZARD_TO_PUB_ID[label]
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        # Type.
        marker.ns = "hazard"
        marker.type = marker.CUBE
        marker.action = marker.ADD
        # Location.
        marker.pose.position = marker_location
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        # Scale.
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 2.0
        # Colours.
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.8
        marker.color.b = 0.3
        # Properties.
        marker.lifetime = rospy.Duration(0)
        marker.frame_locked = True
        # Mark the hazard.
        self.rviz_marker_pub.publish(marker)
        self.marked_hazards.append(label)


def main():
    hazard_detector = HazardMarker()
    rospy.spin()


if __name__ == "__main__":
    main()
