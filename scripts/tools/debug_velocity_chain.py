#!/usr/bin/env python3
import csv
import math
import os
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String


class VelocityChainDebug(Node):
    def __init__(self):
        super().__init__('velocity_chain_debug')
        self.drone_id = int(os.environ.get('DRONE_ID', '1'))
        self.rate_hz = float(os.environ.get('DEBUG_RATE_HZ', '2.0'))
        self.out_path = os.environ.get(
            'DEBUG_VEL_CSV',
            os.path.expanduser('~/ws_xtd2/scripts/runs/velocity_chain_debug.csv'),
        )
        os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
        self.latest = {}
        self.state = {}
        self.start_t = time.time()

        topics = {
            'primitive': f'/xtdrone2/x500_{self.drone_id}/primitive_cmd_vel_ned',
            'safe': f'/xtdrone2/x500_{self.drone_id}/safe_cmd_vel_ned',
            'cmd': f'/xtdrone2/x500_{self.drone_id}/cmd_vel_ned',
        }
        for name, topic in topics.items():
            self.create_subscription(Twist, topic, self._make_twist_cb(name), 10)
        self.create_subscription(String, '/xtdrone2/swarm/state_exchange', self._state_cb, 10)

        self.csv_file = open(self.out_path, 'w', newline='', buffering=1)
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            't', 'x', 'y', 'z', 'vx_state', 'vy_state', 'vz_state',
            'primitive_vx', 'primitive_vy', 'primitive_vz', 'primitive_age',
            'safe_vx', 'safe_vy', 'safe_vz', 'safe_age',
            'cmd_vx', 'cmd_vy', 'cmd_vz', 'cmd_age',
            'safe_minus_primitive_xy', 'cmd_minus_safe_xy',
        ])

        period = 1.0 / max(self.rate_hz, 0.1)
        self.create_timer(period, self._tick)
        self.get_logger().info(
            f'velocity chain debug started for x500_{self.drone_id}, '
            f'rate={self.rate_hz:.1f}Hz, csv={self.out_path}'
        )

    def _make_twist_cb(self, name):
        def cb(msg):
            self.latest[name] = {
                'vx': float(msg.linear.x),
                'vy': float(msg.linear.y),
                'vz': float(msg.linear.z),
                'stamp': time.time(),
            }
        return cb

    def _state_cb(self, msg):
        try:
            import json
            data = json.loads(msg.data)
            for st in data.get('states', []):
                if int(st.get('id', -1)) == self.drone_id:
                    self.state = {
                        'x': float(st.get('x', float('nan'))),
                        'y': float(st.get('y', float('nan'))),
                        'z': float(st.get('z', float('nan'))),
                        'vx': float(st.get('vx', float('nan'))),
                        'vy': float(st.get('vy', float('nan'))),
                        'vz': float(st.get('vz', float('nan'))),
                    }
                    break
        except Exception:
            return

    @staticmethod
    def _fmt(v):
        if v is None or not math.isfinite(v):
            return 'nan'
        return f'{v:.2f}'

    def _vals(self, name, now):
        item = self.latest.get(name)
        if item is None:
            return None, None, None, float('inf')
        return item['vx'], item['vy'], item['vz'], now - item['stamp']

    def _tick(self):
        now = time.time()
        t = now - self.start_t
        pvx, pvy, pvz, page = self._vals('primitive', now)
        svx, svy, svz, sage = self._vals('safe', now)
        cvx, cvy, cvz, cage = self._vals('cmd', now)

        safe_delta = float('nan')
        cmd_delta = float('nan')
        if pvx is not None and svx is not None:
            safe_delta = math.hypot(svx - pvx, svy - pvy)
        if svx is not None and cvx is not None:
            cmd_delta = math.hypot(cvx - svx, cvy - svy)

        st = self.state
        self.writer.writerow([
            f'{t:.3f}',
            st.get('x', float('nan')), st.get('y', float('nan')), st.get('z', float('nan')),
            st.get('vx', float('nan')), st.get('vy', float('nan')), st.get('vz', float('nan')),
            pvx, pvy, pvz, page,
            svx, svy, svz, sage,
            cvx, cvy, cvz, cage,
            safe_delta, cmd_delta,
        ])

        print(
            f't={t:6.1f}s '
            f'pos=({self._fmt(st.get("x"))},{self._fmt(st.get("y"))},{self._fmt(st.get("z"))}) '
            f'state_v=({self._fmt(st.get("vx"))},{self._fmt(st.get("vy"))},{self._fmt(st.get("vz"))}) '
            f'primitive=({self._fmt(pvx)},{self._fmt(pvy)},{self._fmt(pvz)}) a={page:.1f}s '
            f'safe=({self._fmt(svx)},{self._fmt(svy)},{self._fmt(svz)}) a={sage:.1f}s '
            f'cmd=({self._fmt(cvx)},{self._fmt(cvy)},{self._fmt(cvz)}) a={cage:.1f}s '
            f'|safe-prim|={self._fmt(safe_delta)} |cmd-safe|={self._fmt(cmd_delta)}',
            flush=True,
        )

    def destroy_node(self):
        try:
            self.csv_file.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = VelocityChainDebug()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
