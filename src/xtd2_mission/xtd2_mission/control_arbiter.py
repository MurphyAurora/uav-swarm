#!/usr/bin/env python3
"""Control arbitration for UAV local avoidance.

Keeps one final velocity authority:
mission reference -> arbiter -> cmd_vel_ned.
Avoidance planners provide candidate commands only.
"""

from enum import Enum
import time


class ControlState(Enum):
    NORMAL = "normal"
    AVOIDING = "avoiding"
    EMERGENCY = "emergency"
    RECOVER = "recover"


class ControlArbiter:
    def __init__(self, recovery_clearance=1.5):
        self.state = {}
        self.recovery_clearance = float(recovery_clearance)
        self.last_switch = {}

    def select(self, drone_id, mission_cmd, primitive_cmd=None,
               safe_cmd=None, clearance=999.0):
        now = time.time()
        drone_id = int(drone_id)
        current = self.state.get(drone_id, ControlState.NORMAL)

        if safe_cmd is not None:
            self.state[drone_id] = ControlState.EMERGENCY
            self.last_switch[drone_id] = now
            return safe_cmd, self.state[drone_id].value

        if primitive_cmd is not None:
            self.state[drone_id] = ControlState.AVOIDING
            self.last_switch[drone_id] = now
            return primitive_cmd, self.state[drone_id].value

        if current in (ControlState.AVOIDING, ControlState.EMERGENCY):
            if clearance >= self.recovery_clearance:
                self.state[drone_id] = ControlState.RECOVER
            else:
                return mission_cmd, current.value

        self.state[drone_id] = ControlState.NORMAL
        return mission_cmd, self.state[drone_id].value
