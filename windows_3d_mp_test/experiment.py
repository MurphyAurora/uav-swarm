"""
Batch experiments for Windows 3D motion primitive validation.

Scenarios:
- low height obstacles
- medium height obstacles
- high dynamic obstacles

The script is intentionally lightweight and independent from ROS2/PX4.
"""

import random
import time


def generate_scene(level):
    if level == 'low':
        height = random.uniform(1, 2)
    elif level == 'medium':
        height = random.uniform(3, 5)
    else:
        height = random.uniform(5, 8)

    return [{
        'x': random.uniform(8, 15),
        'y': random.uniform(-3, 3),
        'height': height,
        'radius': random.uniform(0.3, 0.8)
    }]


def run_experiment(level, trials=20):
    success = 0
    max_height = []
    runtime = []

    for _ in range(trials):
        scene = generate_scene(level)
        start = time.time()

        # Planner interface placeholder.
        # Connect this with planner.py after primitive selection is finalized.
        result = True
        altitude = 3.0

        runtime.append(time.time() - start)
        max_height.append(altitude)
        success += int(result)

    return {
        'level': level,
        'success_rate': success / trials,
        'avg_runtime': sum(runtime) / trials,
        'avg_max_height': sum(max_height) / trials,
    }


if __name__ == '__main__':
    for level in ['low', 'medium', 'high']:
        print(run_experiment(level))
