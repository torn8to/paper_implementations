import open3d as o3d
import numpy as np

def registrationICP(source_np,
                    target_np,
                    trans_init,
                    threshold=0.02,
                    max_iterations = 500):
    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = source_np
    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = target_np

    result = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iterations=500)
    )
    return result.transformation

