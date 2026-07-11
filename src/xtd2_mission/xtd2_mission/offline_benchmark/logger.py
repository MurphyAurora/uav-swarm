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
            writer = csv.DictWriter(
                f,
                fieldnames=asdict(self.records[0]).keys() if self.records else [],
            )
            writer.writeheader()
            for item in self.records:
                writer.writerow(asdict(item))

    def plot(self, path: str) -> None:
        """Save a lightweight 2D trajectory visualization.

        This function intentionally only depends on matplotlib so that the
        offline benchmark remains independent from ROS2/Gazebo.
        """
        if not self.records:
            return

        import matplotlib.pyplot as plt

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        xs = [item.x for item in self.records]
        ys = [item.y for item in self.records]

        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, marker=".")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Offline planner trajectory")
        plt.axis("equal")
        plt.grid(True)
        plt.savefig(output, bbox_inches="tight")
        plt.close()
