import argparse
import csv
from pathlib import Path

import numpy as np

from planner import MotionPrimitivePlanner
from scene_generator import make_scene
from visualize import plot_result


LOG_COLUMNS = [
    "step",
    "time",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "total_cost",
    "goal_cost",
    "collision_cost",
    "height_cost",
    "smooth_cost",
    "energy_cost",
    "transition_cost",
    "top_clearance_cost",
    "risk_level",
    "status",
]


def _format_debug(debug):
    if not debug:
        return "primitive: none"

    velocity = debug["velocity"]
    terms = debug["cost_terms"]
    return (
        f"primitive: vx={velocity[0]:.2f} vy={velocity[1]:.2f} vz={velocity[2]:.2f} "
        f"goal={terms['goal_cost']:.3f} "
        f"collision={terms['collision_cost']:.3f} "
        f"height={terms['height_cost']:.3f} "
        f"transition={terms['transition_cost']:.3f} "
        f"top_clearance={terms['top_clearance_cost']:.3f} "
        f"risk_level={terms['obstacle_risk']:.3f}"
    )


def _open_log_writer(log_file):
    if log_file is None:
        return None, None

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
    writer.writeheader()
    return handle, writer


def _log_row(writer, step, sim_time, state, cost, debug, status):
    if writer is None:
        return

    row = {
        "step": step,
        "time": round(float(sim_time), 6),
        "x": round(float(state[0]), 6),
        "y": round(float(state[1]), 6),
        "z": round(float(state[2]), 6),
        "vx": "",
        "vy": "",
        "vz": "",
        "total_cost": round(float(cost), 6) if np.isfinite(cost) else "inf",
        "goal_cost": "",
        "collision_cost": "",
        "height_cost": "",
        "smooth_cost": "",
        "energy_cost": "",
        "transition_cost": "",
        "top_clearance_cost": "",
        "risk_level": "",
        "status": status,
    }

    if debug:
        velocity = debug["velocity"]
        terms = debug["cost_terms"]
        row.update(
            {
                "vx": round(float(velocity[0]), 6),
                "vy": round(float(velocity[1]), 6),
                "vz": round(float(velocity[2]), 6),
                "goal_cost": round(float(terms["goal_cost"]), 6),
                "collision_cost": round(float(terms["collision_cost"]), 6),
                "height_cost": round(float(terms["height_cost"]), 6),
                "smooth_cost": round(float(terms["smooth_cost"]), 6),
                "energy_cost": round(float(terms["energy_cost"]), 6),
                "transition_cost": round(float(terms["transition_cost"]), 6),
                "top_clearance_cost": round(float(terms["top_clearance_cost"]), 6),
                "risk_level": round(float(terms["obstacle_risk"]), 6),
            }
        )

    writer.writerow(row)


def run_simulation(
    scenario="forest_gap",
    pillar_height=5.0,
    flight_height=3.0,
    pillar_radius=None,
    max_steps=300,
    dt=0.2,
    predict_time=2.0,
    show_plot=True,
    log_file=None,
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

    planner = MotionPrimitivePlanner(primitive_dt=dt, xy_bounds=scene.get("bounds"))
    history = [state.copy()]
    log_handle, log_writer = _open_log_writer(log_file)

    print("scenario:", scene["name"])
    print("start:", state, "goal:", goal, "obstacles:", len(obstacles))
    print("dt:", dt, "predict_time:", predict_time)
    if log_file is not None:
        print("log file:", str(Path(log_file)))
    if scene.get("bounds") is not None:
        print("xy bounds:", scene["bounds"])
    if obstacles:
        print(
            "first obstacle radius:", round(float(obstacles[0].radius), 3),
            "height:", round(float(obstacles[0].height), 3),
        )

    status = "max_steps"
    try:
        _log_row(log_writer, -1, 0.0, state, 0.0, None, "start")

        for step in range(max_steps):
            best_traj, cost = planner.plan(state, goal, obstacles, predict_time=predict_time)

            if best_traj is None:
                status = "no_feasible"
                print("no feasible trajectory")
                _log_row(log_writer, step, step * dt, state, cost, planner.last_debug, status)
                break

            state = best_traj[1]
            history.append(state.copy())

            for obs in obstacles:
                obs.update(dt)

            status = "running"
            if np.linalg.norm(state - goal) < 1.0:
                status = "goal_reached"

            print(
                "step:", step,
                "state:", np.round(state, 3),
                "cost:", round(cost, 3),
                _format_debug(planner.last_debug),
            )
            _log_row(log_writer, step, (step + 1) * dt, state, cost, planner.last_debug, status)

            if status == "goal_reached":
                print("goal reached")
                break
    finally:
        if log_handle is not None:
            log_handle.close()

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
    parser.add_argument("--log-file", default=None)
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
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()
