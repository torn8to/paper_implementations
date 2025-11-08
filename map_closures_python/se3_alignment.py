import o3d
import numpy as np


def cloud_alignment(source_np: np.ndarray, target_np: np.ndarray, initial_guess: np.ndarray, threshold=0.02) -> np.ndarry:
    source_cloud = o3d.t.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(source_np)
    target_cloud = o3d.t.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target_np)
    source_cloud.paint_uniform_color([1.0, 0.7, 0.0])
    source_cloud.paint_uniform_color([0.7, 1.0, 0.0])
    o3d.visualization.draw_geometries([source_cloud, target_cloud])
    result: o3d.piplelines.registration.RegistrationResult = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, threshold, initial_guess
    )
    print(result)
