#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Control Watchdog
- Compare primitive / safe / final cmd velocities in real time
- Diagnose which layer is controlling UAV
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import json
import time


class ControlWatchdog(Node):
    def __init__(self):
        super().__init__('control_watchdog')

        # ===== topics (can be remapped) =====
        self.declare_parameter('primitive_topic', '/primitive_cmd_vel_ned')
        self.declare_parameter('safe_topic', '/safe_cmd_vel_ned')
        self.declare_parameter('final_topic', '/cmd_vel_ned')
        self.declare_parameter('drone_id', 1)

        self.primitive_topic = self.get_parameter('primitive_topic').value
        self.safe_topic = self.get_parameter('safe_topic').value
        self.final_topic = self.get_parameter('final_topic').value
        self.drone_id = self.get_parameter('drone_id').value

        # ===== state =====
        self.primitive = None
        self.safe = None
        self.final = None

        # ===== subscribers =====
        self.create_subscription(Twist, self.primitive_topic, self.cb_primitive, 10)
        self.create_subscription(Twist, self.safe_topic, self.cb_safe, 10)
        self.create_subscription(Twist, self.final_topic, self.cb_final, 10)

        self.timer = self.create_timer(0.1, self.report)

    def cb_primitive(self, msg):
        self.primitive = msg

    def cb_safe(self, msg):
        self.safe = msg

    def cb_final(self, msg):
        self.final = msg

    def extract(self, msg):
        if msg is None:
            return (0.0, 0.0, 0.0)
        return (msg.linear.x, msg.linear.y, msg.linear.z)

    def report(self):
        pv = self.extract(self.primitive)
        sv = self.extract(self.safe)
        fv = self.extract(self.final)

        # compute deltas
        def diff(a, b):
            return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

        ps = diff(pv, sv)
        sf = diff(sv, fv)

        self.get_logger().info(
            f"DRONE {self.drone_id} | "
            f"P{pv} S{sv} F{fv} | "
            f"P-S{ps} S-F{sf}"
        )


def main():
    rclpy.init()
    node = ControlWatchdog()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
