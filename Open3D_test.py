import open3d as o3d
import numpy as np
import matplotlib.colors as mcolors

"""Segment a point cloud into depth layers and visualize each layer.

Usage:
    python Open3D_segment.py [path_to_pcd] [num_levels]

- path_to_pcd: path to the PointCloud.pcd file (default: 'PointCloud.pcd')
- num_levels: number of depth levels to segment (default: 5)

The script colors each depth band differently and prints the bounding
box size for each level.
"""

import sys


def color_map(index: int, total: int) -> tuple:
    """Return a distinct RGB color for the given index."""
    hue = index / float(max(total, 1))
    return tuple(mcolors.hsv_to_rgb((hue, 1.0, 1.0)))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "PointCloud/PointCloud_box_a1.pcd"
    num_levels = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_points():
        print(f"No points loaded from {path}")
        return

    points = np.asarray(pcd.points)
    min_z, max_z = points[:, 2].min(), points[:, 2].max()
    z_step = (max_z - min_z) / num_levels if num_levels > 0 else (max_z - min_z)

    layers = []
    geometries = []

    for i in range(num_levels):
        z_low = min_z + i * z_step
        z_high = z_low + z_step
        mask = (points[:, 2] >= z_low) & (points[:, 2] < z_high)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        layer = pcd.select_by_index(idx)
        color = color_map(i, num_levels)
        layer.paint_uniform_color(color)

        bbox = layer.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        print(f"Level {i + 1} ({z_low:.2f}-{z_high:.2f}) bbox: {extent}")

        layers.append(layer)
        geometries.append(layer)
        bbox.color = (0, 0, 0)
        geometries.append(bbox)

    if geometries:
        o3d.visualization.draw_geometries(geometries)
    else:
        print("No layers to display.")


if __name__ == "__main__":
    main()