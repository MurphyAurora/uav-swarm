import argparse
import numpy as np

from planner import MotionPrimitivePlanner
from scene_generator import make_scene
from visualize import plot_result


def run_simulation(
    scenario="forest_gap",
    pillar_height=5.0,
    flight_height=3.0,
    max_steps=300,
    dt=0.1,
):
    scene = make_scene(
        scenario,
        pillar_height=pillar_height,
        flight_height=flight_height,
    )
    state = scene["start"].copy()
    goal = scene["goal"].copy()
    obstacles = scene["obstacles"]

    planner = MotionPrimitivePlanner()
    history = [state.copy()]

    print("scenario:", scene["name"])
    print("start:", state, "goal:", goal, "obstacles:", len(obstacles))

    for step in range(max_steps):
        best_traj, cost = planner.plan(state, goal, obstacles, predict_time=2.0)

        if best_traj is None:
            print("no feasible trajectory")
            break

        state = best_traj[1]
        history.append(state.copy())

        for obs in obstacles:
            obs.update(dt)

        print(
            "step:", step,
            "state:", state,
            "cost:", round(cost, 3),
        )

        if np.linalg.norm(state - goal) < 1.0:
            print("goal reached")
            break

    path = np.array(history)
    print("max height:", round(float(np.max(path[:, 2])), 3))
    print("height variation:", round(float(np.sum(np.abs(np.diff(path[:, 2])))), 3))

    plot_result(path, obstacles, goal)
    return path, obstacles, goal


def main():
    parser = argparse.ArgumentParser(description="3D motion primitive validation")
    parser.add_argument("--scenario", default="forest_gap")
    parser.add_argument("--pillar-height", type=float, default=5.0)
    parser.add_argument("--flight-height", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.1)
    args = parser.parse_args()

    run_simulation(
        scenario=args.scenario,
        pillar_height=args.pillar_height,
        flight_height=args.flight_height,
        max_steps=args.max_steps,
        dt=args.dt,
    )


if __name__ == "__main__":
    main()
