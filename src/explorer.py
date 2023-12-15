#!/usr/bin/env python

import numpy as np
import sys

from enum import Enum

import rospy
import actionlib

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from search_navigation_challenge.msg import (
    ExploreFeedback,
    ExploreResult,
    ExploreAction,
    ExploreGoal,
)


class Explorer:
    STATES_ = {
        0: "Find the wall",
        1: "Turn left",
        2: "Follow the wall",
        3: "Diagonally right",
        4: "Stop",
    }

    REGIONS_ = {
        "right": 0,
        "fright": 0,
        "front": 0,
        "fleft": 0,
        "left": 0,
    }

    def LOG_MESSAGE(self, message, type="info"):
        if type == "info":
            rospy.loginfo(message)
        elif type == "warn":
            rospy.logwarn(message)
        elif type == "err":
            rospy.logerr(message)

    def __set_properties(self):
        self.max_wall_dist = 0.6  # Desired wall distance threshold

        self.curr_state = 0
        self.wall_found = False
        self.EXPLORATION_STATE = 0

    def __init__(self):
        rospy.init_node("wall_follower")
        self.__set_properties()
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        self.LOG_MESSAGE("Subscribed - /scan")
        self.controller_listener = actionlib.SimpleActionServer(
            "/explorer_action_server", ExploreAction, self.controller_list, False
        )
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.controller_listener.start()
        self.LOG_MESSAGE("Action Server initiated - /explorer_action_server")
        self.LOG_MESSAGE("Initiating Main Loop...")
        self.run()

    def find_wall(self):
        vel_msg = Twist()
        vel_msg.linear.x = 0.15
        vel_msg.angular.z = 0.0
        return vel_msg

    def go_straight(self):
        vel_msg = Twist()
        vel_msg.linear.x = 0.12
        vel_msg.angular.z = -0.02
        return vel_msg

    def turn_left(self):
        vel_msg = Twist()
        vel_msg.linear.x = 0.02
        vel_msg.angular.z = 0.25
        return vel_msg

    def turn_right_diagonal(self):
        vel_msg = Twist()
        vel_msg.linear.x = 0.05
        vel_msg.angular.z = -0.45
        return vel_msg

    def controller_list(self, msg):
        self.EXPLORATION_STATE = int(msg.goal)
        # Set action
        result = True
        if result:
            self.controller_listener.set_succeeded()
            return ExploreResult(result=True)
        else:
            self.server.set_aborted()
            return ExploreResult(result=False)

    def take_action(self, REGIONS):
        vel_cmd = Twist()
        state_description = ""

        if self.EXPLORATION_STATE == 1:
            if not self.wall_found:  # When wall is not found
                if (
                    REGIONS["front"] > self.max_wall_dist
                    and REGIONS["right"] > self.max_wall_dist
                    and REGIONS["left"] > self.max_wall_dist
                ):
                    self.curr_state = 0
                    state_description = "Finding Wall."
                elif REGIONS["front"] < self.max_wall_dist:
                    state_description = "Wall Found...Following Wall now."
                    self.curr_state = 1
                    self.wall_found = True

            elif self.wall_found:  # When wall is found
                if (
                    REGIONS["right"] > self.max_wall_dist
                    and REGIONS["front"] > self.max_wall_dist
                ):
                    state_description = "case 1 - Moving Diagonally Right"
                    self.curr_state = 3
                elif (
                    REGIONS["right"] > self.max_wall_dist
                    and REGIONS["fright"] > self.max_wall_dist
                    and REGIONS["front"] > self.max_wall_dist
                ):
                    state_description = "case 2 - Turning Right"
                    self.curr_state = 3
                elif REGIONS["front"] > self.max_wall_dist:
                    state_description = "case 3 - Going Straight"
                    self.curr_state = 2
                elif REGIONS["front"] < self.max_wall_dist:
                    state_description = "case 4 - Turning Left"
                    self.curr_state = 1
                elif (
                    REGIONS["front"] < self.max_wall_dist
                    and REGIONS["fleft"] < self.max_wall_dist
                    and REGIONS["fright"] < self.max_wall_dist
                ):
                    state_description = "case 5 - Found a dead end...Turning Left"
                    self.curr_state = 1
                else:
                    state_description = "No State Matched"

    def scan_callback(self, msg):
        length = int(len(msg.ranges))
        eachDiv = length // 8
        right = 2 * eachDiv
        left = length - 2 * eachDiv
        front = np.concatenate(
            (
                np.array(msg.ranges[length - (eachDiv // 2) : length]),
                np.array(msg.ranges[0 : eachDiv // 2]),
            )
        )

        Explorer.REGIONS_ = {
            "front": min(front),
            "left": min(msg.ranges[eachDiv:right]),
            "fleft": min(msg.ranges[eachDiv // 2 : eachDiv]),
            "fright": min(msg.ranges[length - eachDiv : length - (eachDiv // 2)]),
            "right": min(msg.ranges[left : length - eachDiv]),
        }

        self.take_action(Explorer.REGIONS_)

    def run(self):
        self.LOG_MESSAGE("Main Loop Started.")
        pause = True
        HALT_EXPLORATION = 0
        ENABLE_EXPLORATION = 1
        while not rospy.is_shutdown():
            if self.EXPLORATION_STATE == ENABLE_EXPLORATION:
                vel_cmd = Twist()
                if self.curr_state == 0:
                    vel_cmd = self.find_wall()
                elif self.curr_state == 1:
                    vel_cmd = self.turn_left()
                elif self.curr_state == 2:
                    vel_cmd = self.go_straight()
                elif self.curr_state == 3:
                    vel_cmd = self.turn_right_diagonal()
                elif self.curr_state == 4:
                    vel_cmd = Twist()
                else:
                    self.LOG_MESSAGE("Unknown state - MAIN LOOP.", "err")

                self.pub.publish(vel_cmd)
                if pause:
                    pause = False
                    self.LOG_MESSAGE("EXPLORATION ENABLED.", "warn")

            elif self.EXPLORATION_STATE == HALT_EXPLORATION:
                if not pause:
                    self.LOG_MESSAGE("EXPLORATION IS IN HALT STATE.", "warn")
                    self.pub.publish(Twist())
                    pause = True

        rospy.spin()


if __name__ == "__main__":
    explorer = Explorer()
