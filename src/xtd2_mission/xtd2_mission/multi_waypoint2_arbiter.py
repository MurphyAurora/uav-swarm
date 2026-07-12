# DEPRECATED: compatibility wrapper. Use command_arbiter as the only final command owner.

#!/usr/bin/env python3
"""Run multi_waypoint2 as a cruise-proposal producer.

The original multi_waypoint2 publishes directly to /cmd_vel_ned. For the
command-arbiter architecture, it should only provide the low-priority cruise
proposal. This wrapper redirects only Twist publishers whose topic ends with
/cmd_vel_ned to /cruise_cmd_vel_ned, then calls the original multi_waypoint2
main function unchanged.
"""

from geometry_msgs.msg import Twist
from rclpy.node import Node

from xtd2_mission import multi_waypoint2


_ORIGINAL_CREATE_PUBLISHER = Node.create_publisher


def _redirect_cmd_vel_publisher(self, msg_type, topic, qos_profile, *args, **kwargs):
    redirected_topic = topic
    if msg_type is Twist and isinstance(topic, str) and topic.endswith('/cmd_vel_ned'):
        redirected_topic = topic[:-len('/cmd_vel_ned')] + '/cruise_cmd_vel_ned'
        try:
            self.get_logger().info(
                f'command arbiter mode: redirect publisher {topic} -> {redirected_topic}'
            )
        except Exception:
            pass
    return _ORIGINAL_CREATE_PUBLISHER(
        self, msg_type, redirected_topic, qos_profile, *args, **kwargs
    )


def main():
    Node.create_publisher = _redirect_cmd_vel_publisher
    return multi_waypoint2.main()


if __name__ == '__main__':
    main()
