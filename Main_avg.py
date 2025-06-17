import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import time

# variant which filters and then fix tilt
def level_and_filter(pcd: o3d.geometry.PointCloud,
                     distance_thresh=0.002,
                     num_iters=1000,
                     z_margin=0.005) -> o3d.geometry.PointCloud:
    """
    1) RANSAC‐fit the dominant plane (table) and rotate it horizontal.
    2) Remove all points whose Z is below (table_z + z_margin).
    """
    # Fit table and build leveling transform ---
    (a, b, c, d), inliers = pcd.segment_plane(distance_thresh, # max distance point can be to be considered inlier
                                              ransac_n=3, # Must be at least 3 random sample for plane
                                              num_iterations=num_iters) # higher = more accurate but slower
    # pick one table inlier as pivot 
    centroid = np.asarray(pcd.points)[inliers[0]]

    # normal → unit
    n = np.array([a, b, c])
    n /= np.linalg.norm(n)

    # build Rodrigues rotation to send n → [0,0,1]
    z = np.array([0, 0, 1.0])
    v = np.cross(n, z)
    s = np.linalg.norm(v)
    if s < 1e-6:
        R = np.eye(3)
    else:
        c_dot = np.dot(n, z)
        vx = np.array([[   0, -v[2],  v[1]],
                       [ v[2],    0, -v[0]],
                       [-v[1],  v[0],   0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c_dot) / (s**2))

    # apply centering → rotation → decentering
    pcd = pcd.translate(-centroid, relative=False)
    pcd = pcd.rotate(R, center=(0, 0, 0))
    pcd = pcd.translate(centroid, relative=False)

    # Height filter in leveled frame
    # after leveling, the table plane goes through z = centroid[2]
    table_z = centroid[2]
    pts = np.asarray(pcd.points)
    keep = pts[:, 2] <= (table_z + z_margin)

    return pcd.select_by_index(np.where(keep)[0])

# Fit a plane to the point cloud using RANSAC, similar to level_cloud using RANSAC
def plane_fit(cloud, thresh=0.005, n_iter=1000): # thresdd is 0.005 by default
    # Fit with RANSAC
    (a, b, c, d), inliers = cloud.segment_plane(thresh, ransac_n=3, num_iterations=n_iter)
     # Normalize the normal vector and d value
    n = np.array([a, b, c])
    norm = np.linalg.norm(n)
    n /= norm
    d /= norm
    return n, d, inliers

def compute_2d_obb_from_lid(lid_pc: o3d.geometry.PointCloud):
    # Get XY coordinates of lid points (project to 2D)
    points = np.asarray(lid_pc.points)
    xy = points[:, :2]  # Ignore Z

    # Compute convex hull, get smallest rectangle
    hull = ConvexHull(xy)
    hull_pts = xy[hull.vertices]

    # Rotating calipers: test all edges of the hull
    min_area = float("inf")
    best_rect = None

    for i in range(len(hull_pts)):
        # Edge vector
        edge = hull_pts[(i + 1) % len(hull_pts)] - hull_pts[i]
        edge /= np.linalg.norm(edge)  # normalize

        # Get orthogonal vector (rotate 90°)
        ortho = np.array([-edge[1], edge[0]])

        # Build rotation matrix to align edge with X-axis
        R = np.stack([edge, ortho]).T

        # Rotate all hull points
        rot_pts = hull_pts @ R

        # Get bounding box in this frame
        min_xy = rot_pts.min(axis=0)
        max_xy = rot_pts.max(axis=0)
        extent = max_xy - min_xy
        area = extent[0] * extent[1]

        # Update if this is the smallest area
        if area < min_area:
            min_area = area
            best_rect = (R, min_xy, max_xy)

    # Recover the best rectangle in world coords
    R, min_xy, max_xy = best_rect
    center_2d = (min_xy + max_xy) / 2
    corners_2d = np.array([
        [min_xy[0], min_xy[1]],
        [max_xy[0], min_xy[1]],
        [max_xy[0], max_xy[1]],
        [min_xy[0], max_xy[1]],
    ])
    world_corners = (corners_2d @ R.T)

    # Estimate average Z height of lid for 3D placement
    z_mean = np.mean(points[:, 2])
    corners_3d = np.column_stack([world_corners, np.full(4, z_mean)])

    # Return rectangle corners and length/width
    length, width = np.abs(max_xy - min_xy)
    return corners_3d, length, width

if __name__ == '__main__':
    from statistics import mean

    MM_TO_M = 1 / 1000.0
    pcd_path = "PointCloud/PointCloud_box_a1.pcd"

    dims_list = []

    for run in range(3): # change range for number of runss
        print(f"\nRun {run + 1}/3")

        pcd = o3d.io.read_point_cloud(pcd_path)

        # Convert units
        points = np.asarray(pcd.points, dtype=np.float64) * MM_TO_M
        pcd.points = o3d.utility.Vector3dVector(points)

        filtered_pcd = level_and_filter(pcd,
                                distance_thresh=0.002,
                                num_iters=1000,
                                z_margin=0.0) # 0.005 is default but might be bad for clustering segmentation

        n_tab, d_tab, table_inliers = plane_fit(filtered_pcd)
        table_pc = filtered_pcd.select_by_index(table_inliers)
        obj_pc = filtered_pcd.select_by_index(table_inliers, invert=True)

        labels = np.array(obj_pc.cluster_dbscan(eps=0.02, min_points=20))
        largest = np.bincount(labels[labels >= 0]).argmax()
        box_pc = obj_pc.select_by_index(np.where(labels == largest)[0])

        n_lid, d_lid, lid_inliers = plane_fit(box_pc)
        lid_pc = box_pc.select_by_index(lid_inliers)

        corners, length, width = compute_2d_obb_from_lid(lid_pc)
        height_m = abs(d_lid - d_tab)

        dims_cm = np.array([length, width, height_m]) * 100.0
        dims_list.append(dims_cm)

        print(f"  Box ≈ {dims_cm[0]:.2f} × {dims_cm[1]:.2f} × {dims_cm[2]:.2f} cm")

        if run < 4:
            time.sleep(1)  # Pause between runs

    # Compute average
    dims_array = np.array(dims_list)
    avg_dims = dims_array.mean(axis=0)
    print("\nAverage box dimensions over 3 runs:")
    print(f"  ≈ {avg_dims[0]:.2f} × {avg_dims[1]:.2f} × {avg_dims[2]:.2f} cm (L×W×H)")

    # Visualization for the last run only
    lines = [[0,1],[1,2],[2,3],[3,0]]
    colors = [[1, 0.5, 0] for _ in lines]  # orange
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    table_obj = table_pc.get_oriented_bounding_box()
    table_obj.color = (1.0, 0.0, 0.0)
    table_pc.paint_uniform_color([1.0, 0.0, 0.0])  # red
    lid_pc.paint_uniform_color([0.0, 1.0, 0.0])    # green
    box_pc.paint_uniform_color([0.2, 0.8, 1.0])    # cyan

    o3d.visualization.draw_geometries(
        [table_pc, lid_pc, box_pc, line_set, table_obj],
        window_name="Final Visualization: Table, Lid, Box, OBB",
        width=800, height=600
    )

    