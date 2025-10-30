from local_map_management import OdometryWrapper
from kitti360_dataloader import Kitti360LidarData
from pipeline import OptimizationPipeline, OptimizationPipelineConfig
from kitti_file_format import save_poses_as_kitti
import numpy as np
from tqdm import tqdm
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn
from rich.panel import Panel
from rich.layout import Layout

max_frames = 1000


opc = OptimizationPipelineConfig(
    max_points_per_voxel=20,
    alpha=1.0,
    max_hamming_distance=35
)
T_odom = np.eye(4)
T_slam = np.eye(4)
loop_closures = []
status_text = "Initializing..."

# --- tqdm-style progress bar ---
progress = Progress(
    TextColumn("[bold cyan]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("{task.percentage:>3.0f}%"),
)
task = progress.add_task("Optimizing Pose Graph", total=max_frames)


def matrix_panel(matrix, title, color="green"):
    """Make a Rich Panel showing a 4x4 matrix."""
    table = Table.grid(expand=True)
    for row in matrix:
        table.add_row("  ".join(f"{v: .3f}" for v in row))
    return Panel(table, title=f"[bold {color}]{title}[/bold {color}]")


def make_layout():
    """Construct the full UI layout."""
    layout = Layout()
    layout.split_column(
        Layout(Panel(progress), name="progress", size=3),
        Layout(name="main", ratio=1),
    )

    # bottom row split left/right
    layout["main"].split_row(
        Layout(name="transforms", ratio=1),
        Layout(name="loops", ratio=1),
    )

    # --- Left: odom + slam stacked vertically ---
    left = Layout()
    left.split_column(
        Layout(matrix_panel(T_odom, "Odom (Raw)", color="cyan"), size=9),
        Layout(matrix_panel(T_slam, "SLAM (Position)", color="green"), size=9),
    )
    layout["transforms"].update(left)

    # --- Right: loop closures + status ---
    loop_table = Table(title="[bold yellow]Recent Loop Closures[/bold yellow]")
    loop_table.add_column("ID1", justify="center")
    loop_table.add_column("ID2", justify="center")
    loop_table.add_column("Distance", justify="right")

    for (i, j, d) in loop_closures[-10:]:
        color = "green" if d < 1.0 else "yellow" if d < 2.0 else "red"
        loop_table.add_row(str(i), str(j), f"[{color}]{d:.3f}[/{color}]")

    status_panel = Panel(Text(status_text, style="bold magenta"))
    right = Layout()
    right.split_column(
        Layout(Panel(loop_table, title="Loop Closures"), ratio=4),
        Layout(status_panel, size=3),
    )
    layout["loops"].update(right)

    return layout

def main():
    movement_threshold: float = 25.0
    kiss_pipeline = OdometryWrapper()
    data_loader = Kitti360LidarData()
    pipeline = OptimizationPipeline(opc)
    last_density_map_pose = kiss_pipeline.get_current_position()

    #with Live(make_layout(),refresh_per_second=4) as live:
    for _ in tqdm(range(max_frames)):
            progress.update(task, advance=1)
            if not data_loader.has_next():
                return
            cloud = data_loader.retrieve_next_frame()
            cloud_xyz = cloud[:, :3]
            kiss_pipeline.register_frame(cloud_xyz)
            odom_position = kiss_pipeline.get_current_position()
            new_loop_closures = pipeline.update(cloud_xyz, odom_position)
            slam_position = pipeline.pgo_position()

            T_odom[:] = odom_position[:]
            T_slam[:] = slam_position[:]

            for closure in new_loop_closures:
                id = closure[0]
                query_id = closure[1]
                loop_closures.append((id, query_id, 5.0))

    save_poses_as_kitti(pipeline.get_vertices_np(), filename="./kitti360_00_with_pose_graph.txt")


if __name__ == "__main__":
    main()
