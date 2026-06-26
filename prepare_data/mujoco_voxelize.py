"""
Convert a MuJoCo XML scene file to an occupancy voxel grid.

Parses box, cylinder, and plane geom elements from a MuJoCo XML file and
voxelizes them into a binary occupancy grid at a configurable resolution
(default 8 cm). The output is a .pkl file in the standard format used by
the Motion-Occupancy-Base pipeline: (occu_g, unit, llb).

Coordinate convention:
    MuJoCo X = right,  Y = forward,  Z = up
    Codebase X = right, Y = forward, Z = up
    Voxel grid is indexed as [X, Y, Z].
    World point (wx, wy, wz) -> voxel index ((wx - llb_x) / unit, ...)

Margin:
    Free space is added around the room geometry so that the local 25x25x25
    egocentric query grid does not fall out-of-bounds when the character is
    outside the walls. Out-of-bounds queries are clamped to 1 (occupied) in
    query_occu_batched(), which would incorrectly block motion.

Usage:
    python prepare_data/mujoco_voxelize.py assets/room_hanyi/room.xml output.pkl
    python prepare_data/mujoco_voxelize.py assets/room_zixuan/room.xml output.pkl --margin 2.0
"""

import argparse
import os
import pickle
import xml.etree.ElementTree as ET

import numpy as np


def parse_float_list(s):
    """Parse a MuJoCo whitespace-separated float list."""
    return [float(x) for x in s.strip().split()]


def compute_voxel_bounds(geoms, unit, margin_m):
    """
    Compute the global occupancy grid bounds from all finite geoms.

    Args:
        geoms: list of parsed geom dicts
        unit: voxel side length in meters
        margin_m: free-space margin in meters added around all sides

    Returns:
        (llb, grid_shape) where llb is [3] and grid_shape is [3] int array
    """
    all_mins = []
    all_maxs = []

    for g in geoms:
        gtype = g["type"]
        pos = g["pos"]
        size = g["size"]

        if gtype == "plane":
            # Infinite in XY; excluded from bounds computation
            continue

        if gtype == "box":
            gmin = np.array([pos[i] - size[i] for i in range(3)])
            gmax = np.array([pos[i] + size[i] for i in range(3)])
        elif gtype == "cylinder":
            r, h = size[0], size[1]
            gmin = np.array([pos[0] - r, pos[1] - r, pos[2] - h])
            gmax = np.array([pos[0] + r, pos[1] + r, pos[2] + h])
        else:
            raise ValueError(f"Unknown geom type: {gtype}")

        all_mins.append(gmin)
        all_maxs.append(gmax)

    if not all_mins:
        raise ValueError("No finite geoms found in XML — cannot determine grid bounds")

    all_mins = np.stack(all_mins, axis=0)
    all_maxs = np.stack(all_maxs, axis=0)

    # Margin only in XY plane — Z stays tight so the ground plane
    # does not get pushed deep underground, which would break the
    # visualization's hard-coded :18 Z-slice (hides ceiling).
    margin = np.array([margin_m, margin_m, 0.0])
    llb = all_mins.min(axis=0) - margin
    rub = all_maxs.max(axis=0) + margin

    grid_shape = np.ceil((rub - llb) / unit).astype(int)
    return llb, grid_shape


def world_to_voxel_bounds(pos, half_extents, llb, unit, grid_shape):
    """
    Convert a world-space AABB to voxel index ranges, clipped to grid bounds.

    Args:
        pos: [3] center position in world coords
        half_extents: [3] half-extents in world coords
        llb: [3] lower-left-back corner
        unit: voxel side length
        grid_shape: [3] grid dimensions

    Returns:
        (vox_min, vox_max): [3] integer index ranges (start inclusive, end exclusive)
    """
    world_min = pos - half_extents
    world_max = pos + half_extents

    vox_min = np.floor((world_min - llb) / unit).astype(int)
    vox_max = np.ceil((world_max - llb) / unit).astype(int)

    vox_min = np.clip(vox_min, 0, grid_shape)
    vox_max = np.clip(vox_max, 0, grid_shape)

    return vox_min, vox_max


def voxelize_box(occu, llb, unit, grid_shape, pos, size):
    """Fill a box geom in the occupancy grid."""
    half_extents = np.array(size)
    vmin, vmax = world_to_voxel_bounds(np.array(pos), half_extents, llb, unit, grid_shape)
    occu[vmin[0]:vmax[0], vmin[1]:vmax[1], vmin[2]:vmax[2]] = True


def voxelize_cylinder(occu, llb, unit, grid_shape, pos, size):
    """
    Fill a cylinder geom in the occupancy grid.

    MuJoCo cylinders are oriented along the Z axis.
    size[0] = radius, size[1] = half-height.
    """
    r, half_h = size[0], size[1]
    half_extents = np.array([r, r, half_h])
    vmin, vmax = world_to_voxel_bounds(np.array(pos), half_extents, llb, unit, grid_shape)

    # Iterate over the bounding box of voxels
    xs = np.arange(vmin[0], vmax[0])
    ys = np.arange(vmin[1], vmax[1])
    zs = np.arange(vmin[2], vmax[2])

    if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
        return

    # Compute world-space centers of candidate voxels
    cx = llb[0] + (xs + 0.5) * unit
    cy = llb[1] + (ys + 0.5) * unit
    cz = llb[2] + (zs + 0.5) * unit

    # Distance from cylinder axis in XY plane
    dx = cx[:, None, None] - pos[0]
    dy = cy[None, :, None] - pos[1]
    dz = cz[None, None, :] - pos[2]

    dist_sq = dx ** 2 + dy ** 2
    mask_xy = dist_sq <= r ** 2
    mask_z = np.abs(dz) <= half_h

    mask = mask_xy & mask_z
    xi, yi, zi = np.where(mask)  # dim0->xs(X), dim1->ys(Y), dim2->zs(Z)
    occu[xs[xi], ys[yi], zs[zi]] = True


def voxelize_plane(occu, llb, unit, grid_shape, pos, size):
    """
    Fill a plane geom as a finite-height slab within the grid XY extent.

    MuJoCo plane: size="sx sy thickness" where sx, sy are render half-extents
    and thickness is the Z half-thickness. The plane is infinite in collision
    but we limit it to the grid extent.
    """
    thickness = size[2]  # Z half-thickness
    z_center = pos[2]

    half_extents = np.array([0, 0, thickness])
    vmin_z, vmax_z = world_to_voxel_bounds(
        np.array([0, 0, z_center]), half_extents, llb, unit, grid_shape
    )

    # Fill the full XY extent of the grid at these Z levels
    occu[:, :, vmin_z[2]:vmax_z[2]] = True


def parse_mujoco_xml(xml_path):
    """
    Parse a MuJoCo XML file and extract all geom elements.

    Returns:
        list of dicts with keys: type, pos (np.array[3]), size (np.array[3])
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    geoms = []
    for geom_elem in root.iter("geom"):
        gtype = geom_elem.get("type", "box")  # default MuJoCo geom type is 'box'
        pos_str = geom_elem.get("pos", "0 0 0")
        size_str = geom_elem.get("size", "1 1 1")

        pos = np.array(parse_float_list(pos_str), dtype=np.float64)
        size = np.array(parse_float_list(size_str), dtype=np.float64)

        # Handle geom types with 2-element sizes
        if gtype == "cylinder" and len(size) == 2:
            size = np.array([size[0], size[1], size[1]], dtype=np.float64)
        elif gtype == "plane" and len(size) == 2:
            # plane size defaults: first two are render half-extents,
            # third (thickness) defaults to a small value
            size = np.array([size[0], size[1], 0.1], dtype=np.float64)
        elif gtype == "plane" and len(size) == 3:
            size = np.array(size, dtype=np.float64)

        geoms.append({"type": gtype, "pos": pos, "size": size})

    return geoms


def mujoco_xml_to_occupancy(xml_path, unit=0.08, margin_m=2.0):
    """
    Convert a MuJoCo XML scene file to an occupancy voxel grid.

    Args:
        xml_path: path to the MuJoCo .xml file
        unit: voxel side length in meters (default 0.08 = 8 cm)
        margin_m: free-space margin in meters added around all sides of the
                  room geometry (default 2.0 m)

    Returns:
        (occu_g, unit, llb) where:
        - occu_g: np.ndarray of bool, shape [X, Y, Z], 1 = occupied
        - unit: float, voxel side length in meters
        - llb: np.ndarray of shape [1, 3], lower-left-back corner in world coords
    """
    unit = float(unit)
    geoms = parse_mujoco_xml(xml_path)

    # Compute grid bounds from finite geoms (excludes planes)
    llb, grid_shape = compute_voxel_bounds(geoms, unit, margin_m)
    llb = np.asarray(llb, dtype=np.float64)

    # Allocate occupancy grid
    occu = np.zeros(grid_shape, dtype=bool)
    print(f"Allocated occupancy grid: shape={tuple(grid_shape)}, "
          f"llb={llb}, rub={llb + grid_shape * unit}")

    # Voxelize each geom
    for g in geoms:
        gtype = g["type"]
        pos = g["pos"]
        size = g["size"]

        if gtype == "box":
            voxelize_box(occu, llb, unit, grid_shape, pos, size)
        elif gtype == "cylinder":
            voxelize_cylinder(occu, llb, unit, grid_shape, pos, size)
        elif gtype == "plane":
            voxelize_plane(occu, llb, unit, grid_shape, pos, size)
        else:
            raise ValueError(f"Unknown geom type: {gtype}")

    return occu, unit, llb.reshape(1, 3)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a MuJoCo XML scene to an occupancy voxel grid (.pkl)"
    )
    parser.add_argument("xml_path", type=str, help="Path to the MuJoCo .xml file")
    parser.add_argument("output_pkl", type=str, help="Output .pkl file path")
    parser.add_argument(
        "--unit", type=float, default=0.08,
        help="Voxel side length in meters (default: 0.08)"
    )
    parser.add_argument(
        "--margin", type=float, default=2.0,
        help="Free-space margin in meters added around all sides of the room (default: 2.0)"
    )
    args = parser.parse_args()

    occu_g, unit, llb = mujoco_xml_to_occupancy(
        args.xml_path, unit=args.unit, margin_m=args.margin
    )

    os.makedirs(os.path.dirname(args.output_pkl) or ".", exist_ok=True)
    with open(args.output_pkl, "wb") as f:
        pickle.dump((occu_g, unit, llb), f)

    occu_ratio = occu_g.sum() / occu_g.size
    print(f"\nSaved occupancy to {args.output_pkl}")
    print(f"  Geoms processed: {len(parse_mujoco_xml(args.xml_path))}")
    print(f"  Shape:           {occu_g.shape}")
    print(f"  Unit:            {unit} m")
    print(f"  llb:             {llb[0]}")
    print(f"  rub:             {llb[0] + np.array(occu_g.shape) * unit}")
    print(f"  Occupied:        {occu_g.sum():,} / {occu_g.size:,} voxels ({occu_ratio:.2%})")


if __name__ == "__main__":
    main()
