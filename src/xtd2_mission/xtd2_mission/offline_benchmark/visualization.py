"""Simple trajectory visualization for offline benchmark results."""

from pathlib import Path


def plot_trajectory(records, output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not records:
        return

    xs = [item.x for item in records]
    ys = [item.y for item in records]

    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, label="uav trajectory")
    plt.scatter([xs[0]], [ys[0]], label="start")
    plt.scatter([xs[-1]], [ys[-1]], label="end")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
