"""Trajectory logging utilities for offline planner evaluation."""

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


@dataclass
class TrajectoryRecord:
    time: float
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    clearance: float = 0.0
    mode: str = ""
    command_mode: str = ""


class TrajectoryLogger:
    def __init__(self) -> None:
        self.records: List[TrajectoryRecord] = []

    def record(self, **kwargs) -> None:
        self.records.append(TrajectoryRecord(**kwargs))

    def save_csv(self, path: str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(self.records[0]).keys() if self.records else [])
            writer.writeheader()
            for item in self.records:
                writer.writerow(asdict(item))
