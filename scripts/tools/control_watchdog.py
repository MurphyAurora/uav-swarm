#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Control Watchdog (robust)
- Detects whether primitive / safe / final streams are actually alive
- Helps debug why P/S/F = 0 or missing
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time


class ControlWatchdog(Node):
    def __init__(self):
        super().__init__('control_watchdog')

        # ===== topics =====
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

        # last receive time
        self.t_p = 0.0
        self.t_s = 0.0
        self.t_f = 0.0

        # ===== subscribers =====
        self.create_subscription(Twist, self.primitive_topic, self.cb_p, 10)
        self.create_subscription(Twist, self.safe_topic, self.cb_s, 10)
        self.create_subscription(Twist, self.final_topic, self.cb_f, 10)

        self.timer = self.create_timer(0.1, self.report)

        self.get_logger().info(
            f"watchdog started | drone={self.drone_id} | "
            f"P={self.primitive_topic} S={self.safe_topic} F={self.final_topic}"
        )

    def cb_p(self, msg):
        self.primitive = msg
        self.t_p = time.time()

    def cb_s(self, msg):
        self.safe = msg
        self.t_s = time.time()

    def cb_f(self, msg):
        self.final = msg
        self.t_f = time.time()

    def v(self, msg):
        if msg is None:
            return (0.0, 0.0, 0.0)
        return (msg.linear.x, msg.linear.y, msg.linear.z)

    def alive(self, t):
        return (time.time() - t) < 1.0

    def report(self):
        pv = self.v(self.primitive)
        sv = self.v(self.safe)
        fv = self.v(self.final)

        p_alive = self.alive(self.t_p)
        s_alive = self.alive(self.t_s)
        f_alive = self.alive(self.t_f)

        def diff(a, b):
            return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

        self.get_logger().info(
            f"DRONE {self.drone_id} | "
            f"P{pv} S{sv} F{fv} | "
            f"alive P={p_alive} S={s_alive} F={f_alive}"
        )

        if not p_alive:
            self.get_logger().warn("primitive stream missing or not connected")
        if not s_alive:
            self.get_logger().warn("safe stream missing or not connected")
        if not f_alive:
            self.get_logger().warn("final stream missing or not connected")


def main():
    rclpy.init()
    node = ControlWatchdog()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
