#!/usr/bin/env python3
"""Priority-based command arbiter for XTDrone2 swarm missions.

This node implements the mature planner pattern discussed in the project notes:
multiple modules may publish command *proposals*, but only this arbiter publishes
the final ``/cmd_vel_ned`` setpoint for each UAV.

Default input topics per UAV:
  /xtdrone2/x500_{id}/cruise_cmd_vel_ned      low priority mission/waypoint cmd
  /xtdrone2/x500_{id}/orca_cmd_vel_ned        formation/multi-agent proposal
  /xtdrone2/x500_{id}/primitive_cmd_vel_ned   local primitive proposal
  /xtdrone2/x500_{id}/safe_cmd_vel_ned        LiDAR/TTC safety-filtered proposal
  /xtdrone2/x500_{id}/escape_cmd_vel_ned      emergency/fallback proposal

Default output topic per UAV:
  /xtdrone2/x500_{id}/cmd_vel_ned

Important: to avoid command races, other nodes should publish to the proposal
topics above and stop publishing directly to /cmd_vel_ned.
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String


@dataclass
class Proposal:
    source: str
    vx: float
    vy: float
    vz: float
    yaw_rate: float
    stamp: float


class CommandArbiter(Node):
    """Single-output command arbiter with priority, latch and hysteresis."""

    PRIORITY = {
        'failsafe': 100,
        'escape': 90,
        'safe_primitive': 80,
        'primitive': 60,
        'orca': 40,
        'cruise': 10,
    }

    def __init__(
        self,
        num_drones: int,
        max_age_sec: float = 0.6,
        rate_hz: float = 20.0,
        escape_hold_sec: float = 0.8,
        recovery_hold_sec: float = 0.8,
        switch_margin: float = 0.10,
    ):
        super().__init__('command_arbiter')
        self.num_drones = int(num_drones)
        self.max_age_sec = float(max_age_sec)
        self.escape_hold_sec = float(escape_hold_sec)
        self.recovery_hold_sec = float(recovery_hold_sec)
        self.switch_margin = float(switch_margin)

        self.proposals: Dict[int, Dict[str, Proposal]] = {
            i: {} for i in range(1, self.num_drones + 1)
        }
        self.state: Dict[int, str] = {i: 'CRUISE' for i in range(1, self.num_drones + 1)}
        self.state_until: Dict[int, float] = {i: 0.0 for i in range(1, self.num_drones + 1)}
        self.last_source: Dict[int, str] = {i: 'none' for i in range(1, self.num_drones + 1)}
        self.last_cmd: Dict[int, Tuple[float, float, float, float]] = {
            i: (0.0, 0.0, 0.0, 0.0) for i in range(1, self.num_drones + 1)
        }
        self.last_debug_t = 0.0

        self.cmd_pubs = {}
        self.state_pubs = {}
        for drone_id in range(1, self.num_drones + 1):
            self.cmd_pubs[drone_id] = self.create_publisher(
                Twist, f'/xtdrone2/x500_{drone_id}/cmd_vel_ned', 10
            )
            self.state_pubs[drone_id] = self.create_publisher(
                String, f'/xtdrone2/x500_{drone_id}/arbiter_state', 10
            )
            self._subscribe_source(drone_id, 'cruise', 'cruise_cmd_vel_ned')
            self._subscribe_source(drone_id, 'orca', 'orca_cmd_vel_ned')
            self._subscribe_source(drone_id, 'primitive', 'primitive_cmd_vel_ned')
            self._subscribe_source(drone_id, 'safe_primitive', 'safe_cmd_vel_ned')
            self._subscribe_source(drone_id, 'escape', 'escape_cmd_vel_ned')

        period = 1.0 / max(float(rate_hz), 1.0)
        self.timer = self.create_timer(period, self._tick)
        self.get_logger().info(
            f'command_arbiter started: num_drones={self.num_drones}, '
            f'max_age={self.max_age_sec:.2f}s, escape_hold={self.escape_hold_sec:.2f}s, '
            f'recovery_hold={self.recovery_hold_sec:.2f}s'
        )

    def _subscribe_source(self, drone_id: int, source: str, topic_suffix: str) -> None:
        topic = f'/xtdrone2/x500_{drone_id}/{topic_suffix}'
        self.create_subscription(
            Twist,
            topic,
            lambda msg, did=drone_id, src=source: self._proposal_cb(did, src, msg),
            10,
        )

    @staticmethod
    def _finite(value: float, default: float = 0.0) -> float:
        try:
            v = float(value)
        except Exception:
            return default
        return v if math.isfinite(v) else default

    def _proposal_cb(self, drone_id: int, source: str, msg: Twist) -> None:
        self.proposals[int(drone_id)][source] = Proposal(
            source=source,
            vx=self._finite(msg.linear.x),
            vy=self._finite(msg.linear.y),
            vz=self._finite(msg.linear.z),
            yaw_rate=self._finite(msg.angular.z),
            stamp=time.time(),
        )

    def _fresh(self, p: Optional[Proposal], now: float) -> bool:
        return p is not None and (now - p.stamp) <= self.max_age_sec

    def _score_switch(self, drone_id: int, source: str) -> int:
        priority = self.PRIORITY.get(source, 0)
        if source == self.last_source.get(drone_id):
            # Small hysteresis bonus: keep the current source unless a higher
            # priority source is clearly active. This reduces source flicker.
            priority += int(round(self.switch_margin * 100.0))
        return priority

    def _select_source(self, drone_id: int, now: float) -> Tuple[str, str]:
        props = self.proposals[drone_id]
        current_state = self.state[drone_id]

        if self._fresh(props.get('escape'), now):
            self.state[drone_id] = 'ESCAPE'
            self.state_until[drone_id] = max(
                self.state_until.get(drone_id, 0.0), now + self.escape_hold_sec
            )
            return 'escape', 'escape proposal active'

        if current_state == 'ESCAPE' and now < self.state_until.get(drone_id, 0.0):
            # During the escape latch, prefer a still-fresh safe primitive;
            # otherwise publish zero velocity instead of letting low-priority
            # cruise commands immediately take over.
            if self._fresh(props.get('safe_primitive'), now):
                return 'safe_primitive', 'escape hold, safe primitive available'
            return 'failsafe', 'escape hold, no safe proposal'

        if current_state == 'ESCAPE':
            self.state[drone_id] = 'RECOVERY'
            self.state_until[drone_id] = now + self.recovery_hold_sec

        if self.state[drone_id] == 'RECOVERY' and now < self.state_until.get(drone_id, 0.0):
            if self._fresh(props.get('safe_primitive'), now):
                return 'safe_primitive', 'recovery hold'
            if self._fresh(props.get('primitive'), now):
                return 'primitive', 'recovery hold'
            return 'failsafe', 'recovery hold, no planner proposal'

        candidates = []
        for source in ('safe_primitive', 'primitive', 'orca', 'cruise'):
            p = props.get(source)
            if self._fresh(p, now):
                candidates.append((self._score_switch(drone_id, source), source))
        if not candidates:
            self.state[drone_id] = 'FAILSAFE'
            return 'failsafe', 'no fresh proposal'

        candidates.sort(reverse=True)
        selected = candidates[0][1]
        if selected in ('safe_primitive', 'primitive'):
            self.state[drone_id] = 'AVOID'
        elif selected == 'orca':
            self.state[drone_id] = 'FORMATION'
        else:
            self.state[drone_id] = 'CRUISE'
        return selected, 'highest priority fresh proposal'

    def _proposal_to_twist(self, p: Optional[Proposal]) -> Twist:
        tw = Twist()
        if p is not None:
            tw.linear.x = float(p.vx)
            tw.linear.y = float(p.vy)
            tw.linear.z = float(p.vz)
            tw.angular.z = float(p.yaw_rate)
        return tw

    def _tick(self) -> None:
        now = time.time()
        for drone_id in range(1, self.num_drones + 1):
            source, reason = self._select_source(drone_id, now)
            if source == 'failsafe':
                msg = Twist()
            else:
                msg = self._proposal_to_twist(self.proposals[drone_id].get(source))

            self.cmd_pubs[drone_id].publish(msg)
            self.last_source[drone_id] = source
            self.last_cmd[drone_id] = (
                float(msg.linear.x), float(msg.linear.y), float(msg.linear.z), float(msg.angular.z)
            )

            state_msg = String()
            state_msg.data = (
                f'state={self.state[drone_id]}, source={source}, reason={reason}, '
                f'cmd=({msg.linear.x:.2f},{msg.linear.y:.2f},{msg.linear.z:.2f})'
            )
            self.state_pubs[drone_id].publish(state_msg)

        if now - self.last_debug_t >= 1.0:
            self.last_debug_t = now
            summary = []
            for drone_id in range(1, self.num_drones + 1):
                vx, vy, vz, _ = self.last_cmd[drone_id]
                summary.append(
                    f'x500_{drone_id}:{self.state[drone_id]}/{self.last_source[drone_id]}'
                    f' v=({vx:.2f},{vy:.2f},{vz:.2f})'
                )
            self.get_logger().info('command arbiter: ' + '; '.join(summary))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-drones', type=int, default=1)
    parser.add_argument('--max-age-sec', type=float, default=0.6)
    parser.add_argument('--rate', type=float, default=20.0)
    parser.add_argument('--escape-hold-sec', type=float, default=0.8)
    parser.add_argument('--recovery-hold-sec', type=float, default=0.8)
    parser.add_argument('--switch-margin', type=float, default=0.10)
    args, ros_args = parser.parse_known_args(argv)

    rclpy.init(args=ros_args)
    node = CommandArbiter(
        num_drones=args.num_drones,
        max_age_sec=args.max_age_sec,
        rate_hz=args.rate,
        escape_hold_sec=args.escape_hold_sec,
        recovery_hold_sec=args.recovery_hold_sec,
        switch_margin=args.switch_margin,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
