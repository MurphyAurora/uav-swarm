#!/usr/bin/env python3
import argparse
import json
import math
import os
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String


class UavLocalController(Node):
    def __init__(self, drone_id):
        super().__init__(f'x500_{drone_id}_local_controller')
        self.drone_id = int(drone_id)
        self.latest_view = None
        self.latest_ref = None
        self.latest_view_stamp = 0.0
        self.latest_ref_stamp = 0.0
        self.last_log_t = 0.0

        self.consensus_enable = bool(int(self._finite_float(os.environ.get('SEMI_DISTRIBUTED_ENABLE', 1), 1)))
        self.consensus_gain = max(0.0, self._finite_float(os.environ.get('CONSENSUS_CORRECTION_GAIN', 0.25), 0.25))
        self.consensus_max_correction = max(
            0.0,
            self._finite_float(os.environ.get('CONSENSUS_MAX_CORRECTION', 0.35), 0.35),
        )
        self.rear_constraint_enable = bool(int(self._finite_float(os.environ.get('REAR_SPEED_CONSTRAINT', 1), 1)))
        self.rear_safe_gap = max(0.0, self._finite_float(os.environ.get('REAR_SAFE_GAP', 1.45), 1.45))
        self.rear_lateral_window = max(0.0, self._finite_float(os.environ.get('REAR_LATERAL_WINDOW', 0.9), 0.9))
        self.rear_max_brake = max(0.0, self._finite_float(os.environ.get('REAR_MAX_BRAKE', 0.55), 0.55))
        self.ref_timeout_sec = max(0.1, self._finite_float(os.environ.get('LOCAL_CONTROLLER_REF_TIMEOUT', 0.6), 0.6))
        self.view_timeout_sec = max(0.1, self._finite_float(os.environ.get('LOCAL_CONTROLLER_VIEW_TIMEOUT', 1.0), 1.0))

        self.pub = self.create_publisher(Twist, f'/xtdrone2/x500_{self.drone_id}/cmd_vel_ned', 10)
        self.create_subscription(String, f'/xtdrone2/x500_{self.drone_id}/local_view', self._local_view_cb, 10)
        self.create_subscription(String, '/xtdrone2/swarm/formation_reference', self._ref_cb, 10)
        self.timer = self.create_timer(0.05, self._timer_cb)
        self.get_logger().info(
            f'local controller started: id={self.drone_id}, consensus={int(self.consensus_enable)}, '
            f'gain={self.consensus_gain:.2f}, max_correction={self.consensus_max_correction:.2f}, '
            f'rear_constraint={int(self.rear_constraint_enable)}'
        )

    def _finite_float(self, value, default=0.0):
        try:
            v = float(value)
        except Exception:
            return default
        if not math.isfinite(v):
            return default
        return v

    def _normalize_state(self, state):
        out = {
            'id': int(state.get('id', 0)),
            'x': self._finite_float(state.get('x', 0.0)),
            'y': self._finite_float(state.get('y', 0.0)),
            'z': self._finite_float(state.get('z', 0.0)),
            'vx': self._finite_float(state.get('vx', 0.0)),
            'vy': self._finite_float(state.get('vy', 0.0)),
            'vz': self._finite_float(state.get('vz', 0.0)),
            'heading': self._finite_float(state.get('heading', 0.0)),
            'stamp': self._finite_float(state.get('stamp', 0.0)),
        }
        if 'world_x' in state:
            out['world_x'] = self._finite_float(state.get('world_x', state.get('x', 0.0)))
        if 'world_y' in state:
            out['world_y'] = self._finite_float(state.get('world_y', state.get('y', 0.0)))
        return out

    def _local_view_cb(self, msg):
        try:
            data = json.loads(msg.data)
            if int(data.get('id', -1)) != self.drone_id:
                return
            self.latest_view = {
                'self': self._normalize_state(data.get('self') or {}),
                'neighbors': [self._normalize_state(s) for s in data.get('neighbors', []) or []],
                'neighbor_ids': [int(i) for i in data.get('neighbor_ids', []) or []],
                'graph_mode': str(data.get('graph_mode', '')),
            }
            self.latest_view_stamp = time.time()
        except Exception as exc:
            self.get_logger().warn(f'local_view parse failed: {exc}')

    def _ref_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.latest_ref = data
            self.latest_ref_stamp = time.time()
        except Exception as exc:
            self.get_logger().warn(f'formation_reference parse failed: {exc}')

    def _state_fresh(self, st, max_age_sec):
        stamp = float(st.get('stamp', 0.0))
        return stamp > 0.0 and (time.time() - stamp) <= max_age_sec

    def _limit_vector(self, vx, vy, vz, max_norm):
        norm = math.sqrt(vx * vx + vy * vy + vz * vz)
        if norm > max_norm and norm > 1e-6:
            scale = max_norm / norm
            return vx * scale, vy * scale, vz * scale
        return vx, vy, vz

    def _distance_xyz(self, a, b):
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        dz = float(a[2]) - float(b[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _rotate_offset(self, dx, dy, heading):
        c = math.cos(heading)
        s = math.sin(heading)
        return dx * c - dy * s, dx * s + dy * c

    def _state_xy_for_safety(self, st):
        if 'world_x' in st and 'world_y' in st:
            return float(st['world_x']), float(st['world_y'])
        return float(st['x']), float(st['y'])

    def _publish_vel(self, vx, vy, vz):
        tw = Twist()
        tw.linear.x = float(vx)
        tw.linear.y = float(vy)
        tw.linear.z = float(vz)
        self.pub.publish(tw)

    def _offset_for(self, ref, drone_id):
        offsets = ref.get('follower_offsets', {}) or {}
        raw = offsets.get(str(drone_id), [0.0, 0.0, 0.0])
        return float(raw[0]), float(raw[1]), float(raw[2])

    def _estimate_consensus_ref(self, ref, view, state_timeout_sec):
        samples = [view.get('self')] + list(view.get('neighbors', []) or [])
        centers = []
        seen = set()
        heading = self._finite_float(ref.get('heading', 0.0))
        use_heading_offsets = bool(ref.get('use_heading_offsets', False))
        formation_ref = ref.get('formation_ref', [0.0, 0.0, 0.0])
        for st in samples:
            if st is None or not self._state_fresh(st, state_timeout_sec):
                continue
            sample_id = int(st.get('id', 0))
            if sample_id <= 0 or sample_id in seen:
                continue
            seen.add(sample_id)
            dx, dy, dz = self._offset_for(ref, sample_id)
            if use_heading_offsets:
                rdx, rdy = self._rotate_offset(dx, dy, heading)
            else:
                rdx, rdy = dx, dy
            centers.append((float(st['x']) - rdx, float(st['y']) - rdy, float(st['z']) - dz))
        if not centers:
            return formation_ref, 0, 0.0
        inv_n = 1.0 / len(centers)
        consensus = (
            sum(p[0] for p in centers) * inv_n,
            sum(p[1] for p in centers) * inv_n,
            sum(p[2] for p in centers) * inv_n,
        )
        dx = (consensus[0] - float(formation_ref[0])) * self.consensus_gain
        dy = (consensus[1] - float(formation_ref[1])) * self.consensus_gain
        dz = (consensus[2] - float(formation_ref[2])) * self.consensus_gain
        dx, dy, dz = self._limit_vector(dx, dy, dz, self.consensus_max_correction)
        local_ref = (float(formation_ref[0]) + dx, float(formation_ref[1]) + dy, float(formation_ref[2]) + dz)
        return local_ref, len(centers), self._distance_xyz((dx, dy, dz), (0.0, 0.0, 0.0))

    def _apply_rear_constraint(self, desired_vel, ref, view):
        if not self.rear_constraint_enable or self.rear_safe_gap <= 0.0:
            return desired_vel
        formation_ff = ref.get('formation_ff', [0.0, 0.0, 0.0])
        leader_target = ref.get('leader_target', [0.0, 0.0, 0.0])
        formation_ref = ref.get('formation_ref', [0.0, 0.0, 0.0])
        ax = float(formation_ff[0])
        ay = float(formation_ff[1])
        if math.hypot(ax, ay) <= 1e-6:
            ax = float(leader_target[0]) - float(formation_ref[0])
            ay = float(leader_target[1]) - float(formation_ref[1])
        norm = math.hypot(ax, ay)
        if norm <= 1e-6:
            return desired_vel
        ax /= norm
        ay /= norm
        own = view.get('self')
        if own is None:
            return desired_vel
        own_x, own_y = self._state_xy_for_safety(own)
        vx, vy, vz = desired_vel
        own_axis_v = vx * ax + vy * ay
        capped_axis_v = own_axis_v
        for other in view.get('neighbors', []) or []:
            other_x, other_y = self._state_xy_for_safety(other)
            rel_x = other_x - own_x
            rel_y = other_y - own_y
            forward_gap = rel_x * ax + rel_y * ay
            if forward_gap <= 0.0 or forward_gap >= self.rear_safe_gap:
                continue
            lateral_x = rel_x - forward_gap * ax
            lateral_y = rel_y - forward_gap * ay
            if math.hypot(lateral_x, lateral_y) > self.rear_lateral_window:
                continue
            other_axis_v = float(other.get('vx', 0.0)) * ax + float(other.get('vy', 0.0)) * ay
            gap_ratio = (self.rear_safe_gap - forward_gap) / max(self.rear_safe_gap, 1e-6)
            capped_axis_v = min(capped_axis_v, other_axis_v - self.rear_max_brake * gap_ratio)
        delta = capped_axis_v - own_axis_v
        if delta < -1e-6:
            vx += delta * ax
            vy += delta * ay
        return vx, vy, vz

    def _timer_cb(self):
        now = time.time()
        if self.latest_ref is None:
            return
        if now - self.latest_ref_stamp > self.ref_timeout_sec:
            self._publish_vel(0.0, 0.0, 0.0)
            return
        if self.latest_view is None or now - self.latest_view_stamp > self.view_timeout_sec:
            self._publish_vel(0.0, 0.0, 0.0)
            return

        ref = self.latest_ref
        view = self.latest_view
        state_timeout_sec = max(0.1, self._finite_float(ref.get('state_timeout_sec', self.view_timeout_sec), self.view_timeout_sec))
        own = view.get('self')
        if own is None or not self._state_fresh(own, state_timeout_sec):
            self._publish_vel(0.0, 0.0, 0.0)
            return

        formation_ref = ref.get('formation_ref', [0.0, 0.0, 0.0])
        local_ref = formation_ref
        sample_count = 0
        delta_norm = 0.0
        if self.consensus_enable:
            local_ref, sample_count, delta_norm = self._estimate_consensus_ref(ref, view, state_timeout_sec)

        heading = self._finite_float(ref.get('heading', 0.0))
        use_heading_offsets = bool(ref.get('use_heading_offsets', False))
        dx, dy, dz = self._offset_for(ref, self.drone_id)
        if use_heading_offsets:
            rdx, rdy = self._rotate_offset(dx, dy, heading)
        else:
            rdx, rdy = dx, dy
        target = (float(local_ref[0]) + rdx, float(local_ref[1]) + rdy, float(local_ref[2]) + dz)
        ff = ref.get('formation_ff', [0.0, 0.0, 0.0])
        kp = self._finite_float(ref.get('formation_kp', 0.45), 0.45)
        max_speed = max(0.0, self._finite_float(ref.get('max_follower_speed', 0.85), 0.85))
        ex = target[0] - float(own['x'])
        ey = target[1] - float(own['y'])
        ez = target[2] - float(own['z'])
        vx = float(ff[0]) + kp * ex
        vy = float(ff[1]) + kp * ey
        vz = float(ff[2]) + kp * ez
        vx, vy, vz = self._limit_vector(vx, vy, vz, max_speed)
        vx, vy, vz = self._apply_rear_constraint((vx, vy, vz), ref, view)
        self._publish_vel(vx, vy, vz)

        if now - self.last_log_t >= 1.0:
            self.last_log_t = now
            self.get_logger().info(
                f'local cmd stage={ref.get("stage", "")} samples={sample_count} '
                f'consensus_delta={delta_norm:.3f} err=({ex:.2f},{ey:.2f},{ez:.2f}) '
                f'vel=({vx:.2f},{vy:.2f},{vz:.2f}) graph={view.get("graph_mode", "")}'
            )


def main():
    parser = argparse.ArgumentParser(description='Per-UAV local semi-distributed controller')
    parser.add_argument('--id', type=int, required=True)
    args, _ = parser.parse_known_args()
    rclpy.init()
    node = UavLocalController(args.id)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
