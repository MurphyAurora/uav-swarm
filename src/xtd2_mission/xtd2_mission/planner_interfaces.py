"""Common command interface definitions for planner modules.

All local planners should publish proposal commands only.
The command_arbiter node owns the final cmd_vel_ned output.
"""

CRUISE_CMD_TOPIC = "cruise_cmd_vel_ned"
ORCA_CMD_TOPIC = "orca_cmd_vel_ned"
PRIMITIVE_CMD_TOPIC = "primitive_cmd_vel_ned"
SAFE_CMD_TOPIC = "safe_cmd_vel_ned"
ESCAPE_CMD_TOPIC = "escape_cmd_vel_ned"
FINAL_CMD_TOPIC = "cmd_vel_ned"

COMMAND_SOURCES = (
    "cruise",
    "orca",
    "primitive",
    "safe_primitive",
    "escape",
)
