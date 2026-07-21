import argparse
import numpy as np

from planner import MotionPrimitivePlanner
from scene_generator import make_scene
from visualize import plot_result


def _format_debug(debug):
    if not debug:
        return "primitive: none"

    velocity = debug["velocity"]
    terms = debug["cost_terms"]
    return (
        f"primitive: vx={velocity[0]:.2f} vy={velocity[1]:.2f} vz={velocity[2]:.2f} "
        f"height={terms['height_cost']:.3f} "
        f"collision={terms['collision_cost']:.3f} "
        f"transition={terms['transition_cost']:.3f} "
        f"risk={terms['risk_cost']:.3f} "
        f"risk_level={terms['obstacle_risk']:.3f}"
    )


def run_simulation(
    scenario="forest_gap",
    pillar_height=5.0,
    flight_height=3.0,
    pillar_radius=None,
    max_steps=300,
    dt=0.2,
    predict_time=2.0,
    show_plot=True,
):
    scene = make_scene(
        scenario,
        pillar_height=pillar_height,
        flight_height=flight_height,
        pillar_radius=pillar_radius,
    )
    state = scene["start"].copy()
    goal = scene["goal"].copy()
    obstacles = scene["obstacles"]

    planner = MotionPrimitivePlanner(primitive_dt=dt)
    history = [state.copy()]

    print("scenario:", scene["name"])
    print("start:", state, "goal:", goal, "obstacles:", len(obstacles))
    print("dt:", dt, "predict_time:", predict_time)
    if obstacles:
        print(
            "first obstacle radius:", round(float(obstacles[0].radius), 3),
            "height:", round(float(obstacles[0].height), 3),
        )

    for step in range(max_steps):
        best_traj, cost = planner.plan(state, goal, obstacles, predict_time=predict_time)

        if best_traj is None:
            print("no feasible trajectory")
            break

        state = best_traj[1]
        history.append(state.copy())

        for obs in obstacles:
            obs.update(dt)

        print(
            "step:", step,
            "state:", np.round(state, 3),
            "cost:", round(cost, 3),
            _format_debug(planner.last_debug),
        )

        if np.linalg.norm(state - goal) < 1.0:
            print("goal reached")
            break

    path = np.array(history)
    print("max height:", round(float(np.max(path[:, 2])), 3))
    print("height variation:", round(float(np.sum(np.abs(np.diff(path[:, 2])))), 3))

    if show_plot:
        plot_result(path, obstacles, goal)
    return path, obstacles, goal


def main():
    parser = argparse.ArgumentParser(description="3D motion primitive validation")
    parser.add_argument("--scenario", default="forest_gap")
    parser.add_argument("--pillar-height", type=float, default=5.0)
    parser.add_argument("--flight-height", type=float, default=3.0)
    parser.add_argument("--pillar-radius", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--predict-time", type=float, default=2.0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    run_simulation(
        scenario=args.scenario,
        pillar_height=args.pillar_height,
        flight_height=args.flight_height,
        pillar_radius=args.pillar_radius,
        max_steps=args.max_steps,
        dt=args.dt,
        predict_time=args.predict_time,
        show_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
