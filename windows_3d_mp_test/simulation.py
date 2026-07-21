import argparse
import csv
import json
import time
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
    "altitude_deviation_cost",
    "height_cost",
    "smooth_cost",
    "energy_cost",
    "vertical_motion_cost",
    "transition_cost",
    "top_clearance_cost",
    "minimum_clearance",
    "risk_level",
    "computation_time_ms",
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
        f"altitude={terms['altitude_deviation_cost']:.3f} "
        f"vertical={terms['vertical_motion_cost']:.3f} "
        f"transition={terms['transition_cost']:.3f} "
        f"top_clearance={terms['top_clearance_cost']:.3f} "
        f"min_clearance={terms['minimum_clearance']:.3f} "
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


def _round_or_blank(value):
    if value == "" or value is None:
        return ""
    if isinstance(value, str):
        return value
    if np.isinf(value):
        return "inf"
    return round(float(value), 6)


def _log_row(writer, step, sim_time, state, cost, debug, status, computation_time=0.0):
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
        "altitude_deviation_cost": "",
        "height_cost": "",
        "smooth_cost": "",
        "energy_cost": "",
        "vertical_motion_cost": "",
        "transition_cost": "",
        "top_clearance_cost": "",
        "minimum_clearance": "",
        "risk_level": "",
        "computation_time_ms": round(float(computation_time) * 1000.0, 6),
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
                "goal_cost": _round_or_blank(terms["goal_cost"]),
                "collision_cost": _round_or_blank(terms["collision_cost"]),
                "altitude_deviation_cost": _round_or_blank(terms["altitude_deviation_cost"]),
                "height_cost": _round_or_blank(terms["height_cost"]),
                "smooth_cost": _round_or_blank(terms["smooth_cost"]),
                "energy_cost": _round_or_blank(terms["energy_cost"]),
                "vertical_motion_cost": _round_or_blank(terms["vertical_motion_cost"]),
                "transition_cost": _round_or_blank(terms["transition_cost"]),
                "top_clearance_cost": _round_or_blank(terms["top_clearance_cost"]),
                "minimum_clearance": _round_or_blank(terms["minimum_clearance"]),
                "risk_level": _round_or_blank(terms["obstacle_risk"]),
            }
        )

    writer.writerow(row)


def _path_length(path):
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def _minimum_clearance(path, obstacles, cost_model, dt):
    if not obstacles:
        return float("inf")
    times = np.arange(len(path), dtype=float) * float(dt)
    return float(cost_model.minimum_clearance(path, obstacles, times))


def _collision_rate(path, obstacles, cost_model, dt):
    if not obstacles or len(path) == 0:
        return 0.0
    times = np.arange(len(path), dtype=float) * float(dt)
    collisions = 0
    for point, stamp in zip(path, times):
        if cost_model.minimum_clearance(np.asarray([point]), obstacles, np.asarray([stamp])) < 0.0:
            collisions += 1
    return float(collisions / len(path))


def _build_metrics(path, obstacles, planner, dt, status, computation_times):
    min_clearance = _minimum_clearance(path, obstacles, planner.cost, dt)
    collision_rate = _collision_rate(path, obstacles, planner.cost, dt)
    success = status == "goal_reached" and collision_rate == 0.0
    total_comp = float(np.sum(computation_times)) if computation_times else 0.0
    avg_comp = float(np.mean(computation_times)) if computation_times else 0.0
    max_comp = float(np.max(computation_times)) if computation_times else 0.0

    return {
        "success_rate": 1.0 if success else 0.0,
        "collision_rate": collision_rate,
        "path_length": _path_length(path),
        "flight_time": max(0.0, (len(path) - 1) * float(dt)),
        "minimum_clearance": min_clearance,
        "max_altitude": float(np.max(path[:, 2])) if len(path) else 0.0,
        "height_variation": float(np.sum(np.abs(np.diff(path[:, 2])))) if len(path) > 1 else 0.0,
        "computation_time": total_comp,
        "avg_computation_time": avg_comp,
        "max_computation_time": max_comp,
        "status": status,
        "steps": max(0, len(path) - 1),
    }


def _print_metrics(metrics):
    print("metrics:")
    for key in [
        "success_rate",
        "collision_rate",
        "path_length",
        "flight_time",
        "minimum_clearance",
        "max_altitude",
        "height_variation",
        "computation_time",
        "avg_computation_time",
        "max_computation_time",
        "status",
    ]:
        value = metrics[key]
        if isinstance(value, float):
            if np.isinf(value):
                print(f"  {key}: inf")
            else:
                print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


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
    summary_file=None,
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
    computation_times = []
    log_handle, log_writer = _open_log_writer(log_file)

    print("scenario:", scene["name"])
    print("start:", state, "goal:", goal, "obstacles:", len(obstacles))
    print("dt:", dt, "predict_time:", predict_time)
    if log_file is not None:
        print("log file:", str(Path(log_file)))
    if summary_file is not None:
        print("summary file:", str(Path(summary_file)))
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
            tic = time.perf_counter()
            best_traj, cost = planner.plan(state, goal, obstacles, predict_time=predict_time)
            comp_time = time.perf_counter() - tic
            computation_times.append(comp_time)

            if best_traj is None:
                status = "no_feasible"
                print("no feasible trajectory")
                _log_row(log_writer, step, step * dt, state, cost, planner.last_debug, status, comp_time)
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
                "comp_ms:", round(comp_time * 1000.0, 3),
                _format_debug(planner.last_debug),
            )
            _log_row(log_writer, step, (step + 1) * dt, state, cost, planner.last_debug, status, comp_time)

            if status == "goal_reached":
                print("goal reached")
                break
    finally:
        if log_handle is not None:
            log_handle.close()

    path = np.array(history)
    metrics = _build_metrics(path, obstacles, planner, dt, status, computation_times)
    _print_metrics(metrics)

    if summary_file is not None:
        summary_path = Path(summary_file)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if show_plot:
        plot_result(path, obstacles, goal)
    return path, obstacles, goal, metrics


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
    parser.add_argument("--summary-file", default=None)
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
        summary_file=args.summary_file,
    )


if __name__ == "__main__":
    main()
