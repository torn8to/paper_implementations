import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.spatial.transform import Rotation

EulerXYZ = namedtuple("EulerXYZ", "roll pitch yaw x y z")

def convert_homogenuous_pose_to_euler_xyz(pose:np.ndarray):
    r = Rotation.from_matrix(pose[:3,:3]).as_euler('xyz')
    return EulerXYZ(r[0], r[1], r[2], pose[0,3], pose[1,3], pose[2,3])

def convert_se2_to_euler_xyz(theta, t, resolution)-> tuple:
    euler_xyz = np.array([0, 0, theta])
    xyz = np.array([t[0]*resolution, t[1]*resolution, 0])
    return (euler_xyz, xyz)

def convert_se2_to_se3(ransac_model, resolution=1.0, z_translation=0.0):
    """
    Convert SE2 transformation from RANSAC result to SE3 transformation matrix.
    
    Args:
        ransac_model: RansacReturnModel containing so2 (2x2 rotation matrix), t (2D translation), theta (rotation angle)
        resolution: Scale factor for translation (default 1.0)
        z_translation: Z-axis translation for the point cloud frame (default 0.0)
    
    Returns:
        np.ndarray: 4x4 SE3 transformation matrix
    """
    # Extract SE2 components
    R_2d = ransac_model.so2  # 2x2 rotation matrix
    t_2d = ransac_model.t    # 2D translation vector
    theta = ransac_model.theta  # rotation angle (for reference)
    
    # Create 3D rotation matrix from 2D rotation
    # The 2D rotation is around Z-axis in 3D space
    R_3d = np.eye(3)
    R_3d[:2, :2] = R_2d
    
    # Create 3D translation vector
    t_3d = np.array([t_2d[0] * resolution, t_2d[1] * resolution, z_translation])
    
    # Construct 4x4 SE3 transformation matrix
    T_se3 = np.eye(4)
    T_se3[:3, :3] = R_3d
    T_se3[:3, 3] = t_3d
    
    return T_se3

def convert_ransac_to_se3_pose(ransac_model, resolution=1.0, z_translation=0.0):
    """
    Convert RansacReturnModel to SE3 pose in point cloud frame.
    
    Args:
        ransac_model: RansacReturnModel from ransac2d
        resolution: Scale factor for translation (default 1.0)
        z_translation: Z-axis translation for the point cloud frame (default 0.0)
    
    Returns:
        tuple: (T_se3, inliers) where T_se3 is 4x4 transformation matrix and inliers is the inlier mask
    """
    T_se3 = convert_se2_to_se3(ransac_model, resolution, z_translation)
    return T_se3, ransac_model.inliers

def plot_transformation_matrices(transforms, connections=None, figsize=(10, 8), title="Transformation Matrices in XY Plane"):
    """
    Plot homogeneous transformation matrices in the xy plane and optionally draw lines between pairs.
    
    Args:
        transforms (list): List of 4x4 homogeneous transformation matrices
        connections (list, optional): List of tuples (i, j) representing pairs of matrix indices to connect with lines.
                                     If None, draws green lines between consecutive points.
        figsize (tuple): Figure size for matplotlib (default (10, 8))
        title (str): Title for the plot (default "Transformation Matrices in XY Plane")
    
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract x, y coordinates from transformation matrices
    x_coords = []
    y_coords = []
    
    for transform in transforms:
        x_coords.append(transform[0, 3])  # x translation
        y_coords.append(transform[1, 3])  # y translation
    
    # Plot the transformation positions as points
    ax.scatter(x_coords, y_coords, c='blue', s=50, alpha=0.7, label='Transformations')
    
    # Add index labels for each point
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', 
                   fontsize=8, alpha=0.8)
    
    # Always draw green trajectory lines between consecutive points
    if len(transforms) > 1:
        ax.plot(x_coords, y_coords, color='green', linewidth=2, alpha=0.7, label='Trajectory')
    
    # Draw custom connections if provided
    if connections is not None:
        for idx, (i, j) in enumerate(connections):
            if i < len(transforms) and j < len(transforms):
                x1, y1 = transforms[i][0, 3], transforms[i][1, 3]
                x2, y2 = transforms[j][0, 3], transforms[j][1, 3]
                ax.plot([x1, x2], [y1, y2], color='orange', linewidth=2, 
                       alpha=0.7, label='Loop Closure' if idx == 0 else "")
    
    # Set equal aspect ratio and add grid
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(title)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig


def visualize_hbst_matches(match_object, img1, img2):

    """
    Visualize matches between two images with connecting lines.
    Works with pyhbst match objects.
    
    Parameters:
    -----------
    img1 : numpy.ndarray
        First image (grayscale or RGB)
    img2 : numpy.ndarray
        Second image (grayscale or RGB)
    kp1 : list
        Keypoints in first image (e.g., cv2.KeyPoint objects)
    kp2 : list
        Keypoints in second image (e.g., cv2.KeyPoint objects)
    matches : list
        List of match objects from pyhbst (e.g., cv2.DMatch objects)
        Each match has .queryIdx and .trainIdx attributes
    max_matches : int
        Maximum number of matches to display
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the visualization
    """
    # Get image dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create side-by-side image
    h = max(h1, h2)
    combined = np.zeros((h, w1 + w2, 3), dtype=img1.dtype)
    combined[:h1, :w1] = img1
    combined[:h2, w1:w1+w2] = img2
    
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(combined, cmap='gray' if combined.shape[2] == 1 else None)
    
    # Draw matches
    colors = plt.cm.hsv(np.linspace(0, 1, len(matches_to_show)))
    
    for match, color in zip(matches, colors):
        # Get keypoint indices from match object
        kpt1 = match.object_query
        kpt2 = match.matchable_query[0]
        
        # Offset x2 by width of first image
        x2_offset = x2 + w1
        
        # Draw line
        ax.plot([x1, x2_offset], [y1, y2], '-', color=color, linewidth=1.5, alpha=0.7)
        
        # Draw keypoints
        #ax.plot(x1, y1, 'o', color=color, markersize=6, markeredgecolor='white', markeredgewidth=1)
        #ax.plot(x2_offset, y2, 'o', color=color, markersize=6, markeredgecolor='white', markeredgewidth=1)
    
    ax.axis('off')
    ax.set_title(f'Feature Matches ({len(matches_to_show)} shown)', fontsize=14, pad=10)
    plt.tight_layout()
    
    return fig

def cloud_visualization(source_np: np.ndarray, 
                        target_np: np.ndarray):
    print(source_np.shape, target_np.shape)
    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(source_np)
    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target_np)
    source_cloud.paint_uniform_color([1.0, 0.7, 0.0])
    target_cloud.paint_uniform_color([0.7, 1.0, 0.0])
    o3d.visualization.draw_geometries([source_cloud, target_cloud])
    

