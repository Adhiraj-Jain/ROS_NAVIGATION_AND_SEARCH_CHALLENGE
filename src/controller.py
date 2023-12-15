import enum

from typing import NoReturn

import rospy
import actionlib

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Path
from std_msgs.msg import Bool

from search_navigation_challenge.msg import ExploreAction, ExploreGoal


class ControllerState(enum.Enum):
    WAITING_FOR_START = 0
    INVESTIGATING = 1
    EXPLORING = 2
    RETURNING = 3


class Controller:
    def __init__(self) -> NoReturn:
        rospy.loginfo("Initializing controller...")
        self.node = rospy.init_node("search_nav_challenge:controller", anonymous=False)
        # Note: Callbacks may change self.state.
        self.all_hazards_detected_sub = rospy.Subscriber(
            "/all_hazards_detected", Bool, self.all_hazards_found
        )
        self.explorer_client = actionlib.SimpleActionClient(
            "/explorer_action_server", ExploreAction
        )
        self.investigation_pub = rospy.Publisher("/investigate", Bool, queue_size=1)
        self.state = ControllerState.WAITING_FOR_START
        self.previous_state = ControllerState.WAITING_FOR_START

    def run(self) -> NoReturn:
        self.control_loop()

    def control_loop(self) -> NoReturn:
        program_strategy = {
            ControllerState.WAITING_FOR_START: self.wait_for_start,
            ControllerState.EXPLORING: self.initiate_exploration,
            ControllerState.INVESTIGATING: self.initiate_investigation,
            ControllerState.RETURNING: self.return_to_starting_position,
        }
        state_exit_strategy = {
            ControllerState.WAITING_FOR_START: lambda: None,
            ControllerState.EXPLORING: self.halt_exploration,
            ControllerState.INVESTIGATING: self.halt_investigation,
            ControllerState.RETURNING: lambda: None,
        }

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.state != self.previous_state:
                rospy.loginfo(
                    f"State changed from {self.previous_state} to {self.state}"
                )
                state_exit_strategy[self.previous_state]()
                self.previous_state = self.state
            # Note: Functions may change self.state.
            program_strategy[self.state]()
            rate.sleep()

    # Program strategies.
    def wait_for_start(self) -> NoReturn:
        try:
            rospy.wait_for_message("/start_challenge", Bool, timeout=rospy.Duration(5))
            self.state = ControllerState.EXPLORING
        except rospy.exceptions.ROSException:
            rospy.logwarn("No start message received from vision node.")

    def initiate_exploration(self):
        exploreGoal = ExploreGoal()
        exploreGoal.goal = 1
        self.explorer_client.send_goal(exploreGoal)

    def initiate_investigation(self):
        self.investigation_pub.publish(True)

    def return_to_starting_position(self):
        x, y, w, z = self.get_starting_position()
        rospy.loginfo(
            f"Starting position: x={x}, y={y}, w={w}; moving to start position..."
        )
        self.move(x, y, w, z)

    # State exit strategies.
    def halt_exploration(self):
        exploreGoal = ExploreGoal()
        exploreGoal.goal = 0
        self.explorer_client.send_goal(exploreGoal)

    def halt_investigation(self):
        self.investigation_pub.publish(False)

    # Callback functions.
    def all_hazards_found(self, _) -> NoReturn:
        rospy.loginfo("All hazards detected. Returning to starting position...")
        self.state = ControllerState.RETURNING

    # Helper functions.
    def get_starting_position(self) -> MoveBaseGoal:
        path_model = rospy.wait_for_message("/path", Path, timeout=rospy.Duration(15))
        start_location = path_model.poses[0]
        x, y = start_location.pose.position.x, start_location.pose.position.y
        w, z = start_location.pose.orientation.w, start_location.pose.orientation.z
        return x, y, w, z

    def move(self, x, y, w, z) -> MoveBaseGoal:
        client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        client.wait_for_server()
        # Add goal metadata.
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        # Add goal position.
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.w = w
        goal.target_pose.pose.orientation.z = z

        client.send_goal(goal)
        wait = client.wait_for_result()
        if not wait:
            rospy.logerr("Unable to query move_base.")


def main():
    controller = Controller()
    controller.run()


if __name__ == "__main__":
    main()
