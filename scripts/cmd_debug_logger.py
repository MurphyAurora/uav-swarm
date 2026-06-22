import json
import time

class CmdDebugLogger:
    """
    Lightweight logger for UAV command arbitration debugging.
    Logs primitive / safe / final cmd_vel into jsonl.
    """

    def __init__(self, log_path="logs/cmd_debug_log.jsonl"):
        self.log_path = log_path
        self.file = open(self.log_path, "a")

    def log(self, primitive=None, safe=None, cmd=None, mode=None, avoid_source=None):
        def pack(v):
            if v is None:
                return None
            return {
                "vx": float(getattr(v.linear, "x", 0.0)),
                "vy": float(getattr(v.linear, "y", 0.0)),
                "vz": float(getattr(v.linear, "z", 0.0)),
            }

        record = {
            "t": time.time(),
            "primitive": pack(primitive),
            "safe": pack(safe),
            "cmd": pack(cmd),
            "mode": mode,
            "avoid_source": avoid_source,
        }

        self.file.write(json.dumps(record) + "\n")
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()
