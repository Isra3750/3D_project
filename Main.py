import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.path import Path

def box_location(pcd: o3d.geometry.PointCloud, 
                      bbox_pix) -> o3d.geometry.PointCloud:
    eps = 1e-9
    u = (intrinsics["fx"] * pts[:,0]) / (pts[:,2] + eps) + intrinsics["cx"]
    v = (intrinsics["fy"] * pts[:,1]) / (pts[:,2] + eps) + intrinsics["cy"]
    uv = np.c_[u, v]      

    x1, y1, x2, y2 = bbox_pix
    inside = (uv[:,0] >= x1) & (uv[:,0] <= x2) & \
            (uv[:,1] >= y1) & (uv[:,1] <= y2)

    print(f"Total points in cloud : {len(pcd.points):,}")
    print(f"Points that fall in 2-D BBox : {inside.sum():,}")

    col = np.zeros((len(pcd.points),3))
    col[:] = [0.7,0.7,0.7]          # grey
    col[inside] = [1, 0.706, 0]   # Yellow
    pcd.colors = o3d.utility.Vector3dVector(col)

    return pcd

def level_and_filter(pcd: o3d.geometry.PointCloud,
                     distance_thresh=0.002,
                     num_iters=1000,
                     z_min=0.005,
                     z_max=0.025) -> o3d.geometry.PointCloud:
    # Fit table and build leveling transform
    (a, b, c, d), inliers = pcd.segment_plane(distance_thresh,
                                              ransac_n=3,
                                              num_iterations=num_iters)
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

    # Initialize mask to keep all points first
    mask = np.ones(len(pts), dtype=bool)

    if z_min is not None:
        mask &= pts[:, 2] <= (table_z + z_min) # filter out all points below the segmented table + certain height
    if z_max is not None:
        mask &= pts[:, 2] >= (table_z - z_max) # filter out all points above the segmented table + certain height

    return pcd.select_by_index(np.where(mask)[0])

def plane_fit(cloud, thresh=0.005, n_iter=1000):
    (a, b, c, d), inliers = cloud.segment_plane(thresh, ransac_n=3, num_iterations=n_iter)
    
    # Normalize
    n = np.array([a, b, c])
    norm = np.linalg.norm(n)
    n /= norm
    d /= norm

    return n, d, inliers

def plane_fit_horizontal(cloud, thresh=0.005, n_iter=1000, max_tilt_deg=15, visualize=True):
    (a, b, c, d), inliers = cloud.segment_plane(thresh, ransac_n=3, num_iterations=n_iter)
    
    # Normalize
    n = np.array([a, b, c])
    norm = np.linalg.norm(n)
    n /= norm
    d /= norm

    # Check angle between plane normal and Z-axis
    cos_theta = np.abs(np.dot(n, [0, 0, 1]))  # |cos(θ)| where θ is angle with Z
    angle_deg = np.arccos(cos_theta) * (180.0 / np.pi)

    if angle_deg > max_tilt_deg:
        print(f"Rejected plane: tilt angle {angle_deg:.2f}° > {max_tilt_deg}° (not horizontal enough)")
        return None, None, None

    return n, d, inliers

def extract_largest_yellow_cluster(
    obj_pc: o3d.geometry.PointCloud,
    *,
    eps: float = 0.01,
    min_points: int = 20,
    yellow_rgb: tuple[float, float, float] = (1.0, 0.706, 0.0),
    threshold: float = 0.50,
    visualize: bool = False,
    window_name: str | None = None,
) -> o3d.geometry.PointCloud:
    # DBSCAN cluster
    labels = np.asarray(obj_pc.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

    # Build a boolean mask for yellow points
    yellow_rgb = np.asarray(yellow_rgb)
    col_obj    = np.asarray(obj_pc.colors)
    yellow_mask = np.all(np.isclose(col_obj, yellow_rgb, atol=1e-3), axis=1)

    # Select the best cluster
    best_lbl, best_size = -1, -1
    for lbl in np.unique(labels[labels >= 0]):  # iterate over real clusters
        idx   = labels == lbl                   # mask for this cluster
        ratio = yellow_mask[idx].sum() / idx.sum()

        if ratio >= threshold and idx.sum() > best_size:
            best_lbl, best_size = lbl, idx.sum()

    if best_lbl == -1:
        raise RuntimeError(
            f"No cluster had ≥{int(threshold * 100)} % of its points in the yellow ROI."
        )

    # Extract and show the cluster if needed
    box_pc = obj_pc.select_by_index(np.where(labels == best_lbl)[0])
    if visualize:
        if window_name is None:
            window_name = f"Largest cluster ≥{int(threshold * 100)} % yellow"
        box_pc.paint_uniform_color([0.2, 0.8, 1.0])  # cyan
        o3d.visualization.draw_geometries([box_pc], window_name=window_name)

    return box_pc

def compute_2d_obb_from_lid(lid_pc):
    # Get XY coordinates of lid points (project to 2D)
    points = np.asarray(lid_pc.points)
    xy = points[:, :2]  # Ignore Z

    # Compute convex hull
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

def lid_coverage(box_pc):
    # RANSAC to find dominant plane (lid) + hollow check
    n_lid, d_lid, lid_inliers = plane_fit_horizontal(box_pc)

    # if plane is not horizontal enough, return None
    if lid_inliers is None: 
        return None, None, None, None, 0
    
    lid_pc = box_pc.select_by_index(lid_inliers)

    # Project inliers to XY and compute their 2D convex hull
    pts2d = np.asarray(lid_pc.points)[:, :2]
    hull = ConvexHull(pts2d)
    hull_pts = pts2d[hull.vertices]

    # Build a 2D polygon for point-in-hull tests
    poly = Path(hull_pts)

    # Create a grid covering the hull's bounding box
    grid_size = 50
    min_xy = hull_pts.min(axis=0)
    max_xy = hull_pts.max(axis=0)
    cell_size = (max_xy - min_xy) / grid_size

    # Which grid‐cell centers are inside the hull?
    x_centers = min_xy[0] + (np.arange(grid_size) + 0.5) * cell_size[0]
    y_centers = min_xy[1] + (np.arange(grid_size) + 0.5) * cell_size[1]
    XX, YY = np.meshgrid(x_centers, y_centers)
    centers = np.vstack([XX.ravel(), YY.ravel()]).T
    inside = poly.contains_points(centers).reshape(grid_size, grid_size)

    # Mark which cells have at least one lid point
    filled = np.zeros_like(inside)
    idx = ((pts2d - min_xy) / cell_size).astype(int)
    idx[:, 0] = np.clip(idx[:, 0], 0, grid_size - 1)
    idx[:, 1] = np.clip(idx[:, 1], 0, grid_size - 1)
    # Note: idx rows are [i, j] = [x_index, y_index]
    filled[idx[:, 1], idx[:, 0]] = True

    # Compute coverage = filled cells inside hull / total hull cells
    coverage = filled[inside].sum() / inside.sum()
    print(f"Lid coverage: {coverage:.2%}")

    return n_lid, d_lid, lid_inliers, lid_pc, coverage

def top_percentile_obb(box_pc, percentile=10):
    # Pull out raw points
    pts = np.asarray(box_pc.points)
    zs  = pts[:, 2]

    # Threshold at the desired percentile
    z_thr   = np.percentile(zs, percentile)
    top_idx = np.where(zs <= z_thr)[0]
    top_pts = pts[top_idx]               # shape (M, 3)

    # Build a temp Open3D cloud of just those top points
    temp_pc = o3d.geometry.PointCloud()
    temp_pc.points = o3d.utility.Vector3dVector(top_pts)

    # Now call your existing plane‐to‐OBB function
    corners, length, width = compute_2d_obb_from_lid(temp_pc)

    return corners, length, width

def estimate_box_dimensions(box_pc, coverage, lid_inliers, d_lid, d_tab, lid_pc):
    # If RANSAC hull is too hollow or no inliers (if angle is too tilted for RANSAC), fall back to top‐percentile outline
    if (coverage < 0.7) or (lid_inliers == None):
        corners, length, width = top_percentile_obb(box_pc, percentile=20) # top 20 percent of box is used to project to 2D then find XY
        # Compute height from the top 5% of points
        pts = np.asarray(box_pc.points)
        zs  = pts[:, 2]

        # want the top 1%, so use 99th percentile threshold, for height level
        z_thr   = np.percentile(zs, 1)
        top_idx = np.where(zs <= z_thr)[0]
        top_pts = pts[top_idx]

        # Use the mean (or max) Z of those as your lid height
        z_lid_est = top_pts[:, 2].mean()

        # Get diff between top of box and table
        height_m = abs(z_lid_est + d_tab)

    else:
        corners, length, width = compute_2d_obb_from_lid(lid_pc) # standard method of projecting to 2D then find XY

        # Get diff between top of box and table
        height_m = abs(d_lid - d_tab)

    print(f"Box ≈ {length*100:.2f} × {width*100:.2f} × {height_m*100:.2f} cm  (L×W×H)")

if __name__ == "__main__":
    # Camera parameters
    intrinsics = dict(fx=617.0, fy=617.0, cx=319.5, cy=239.5) # <- camera dimensions

    # Load in data and box input coordinate base on 5 samples
    box_num = input("Enter the experiment number (e.g. e1, e2, e3, e4 big, e4 small, e5): ")
    if box_num == "e1":
        pcd_path = "Phase_2_PCD/PointCloud_e1.pcd" # <- full path to your .pcd file
        Box_corners = [214.18213, 253.65614, 435.97366, 366.13275]
    elif box_num == "e2":
        pcd_path = "Phase_2_PCD/PointCloud_e2.pcd"
        Box_corners = [338.89346, 268.44052, 519.0965 , 439.19873]
    elif box_num == "e3":
        pcd_path = "Phase_2_PCD/PointCloud_e3.pcd"
        Box_corners = [338.1808 , 250.50539, 519.58405, 478.66837]
    elif box_num == "e4 big":
        pcd_path = "Phase_2_PCD/PointCloud_e4.pcd"
        Box_corners = [79.322464, 192.35791 , 254.25697 , 366.5183]
    elif box_num == "e4 small":
        pcd_path = "Phase_2_PCD/PointCloud_e4.pcd"
        Box_corners = [385.79095 , 286.39618 , 460.3058  , 370.37234]
    elif box_num == "e5":
        pcd_path = "Phase_2_PCD/PointCloud_e5.pcd"
        Box_corners = [182.18625, 223.96574, 460.8647 , 412.31375]
    else:
        raise ValueError("Invalid experiment number")
    
    bbox_pix = np.array(Box_corners) 

    # load the point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points) # shape = (N,3)

    # Find box location
    pcd = box_location(pcd, bbox_pix)

    # Convert units (assumes points are in mm)
    MM_TO_M = 1 / 1000.0
    points = np.asarray(pcd.points, dtype=np.float64) * MM_TO_M
    pcd.points = o3d.utility.Vector3dVector(points)

    # Level and filter
    pcd = level_and_filter(pcd,
                        distance_thresh=0.003, # how tightly the plane fitting should hug the data, 2mm
                        num_iters=1000,
                        z_min=0, # 25mm (0.025) default, extra margin at bottom of table for filtering below
                        z_max=0.250) # 250mm (0.250) default, filter all points above the table

    # Find table plane
    n_tab, d_tab, table_inliers = plane_fit_horizontal(pcd, thresh=0.005) # Finds the table plane

    # Error check (i.e. no table founded)
    if table_inliers == None:
        raise Exception("No table detected")

    # Find table centroid
    table_pc = pcd.select_by_index(table_inliers) # table object (dominant flat plane)
    obj_pc = pcd.select_by_index(table_inliers, invert=True) # all other object, found by using invert=True

    # Find box cluster
    box_pc = extract_largest_yellow_cluster(obj_pc, eps=0.015, min_points=30, threshold=0.6, visualize=True)

    # RANSAC to find lid (dominant plane), this will be top of box (or side, or hollow), also get lid coverage
    n_lid, d_lid, lid_inliers, lid_pc, coverage = lid_coverage(box_pc)

    # Estimate box dimensions
    estimate_box_dimensions(box_pc, coverage, lid_inliers, d_lid, d_tab, lid_pc)

    # visualize
    obj_pc.paint_uniform_color([1, 0, 0]) # red
    box_pc.paint_uniform_color([0.2, 0.8, 1.0]) # cyan
    lid_pc.paint_uniform_color([0, 1, 0]) # green
    o3d.visualization.draw_geometries([lid_pc, box_pc, table_pc, obj_pc])