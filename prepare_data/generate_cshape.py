"""
Generate the C-shaped room occupancy as a .pkl file.

This replicates the hard-coded C-shape from training/dataset.py lines 110-131
and saves it in the standard (occu_g, unit, llb) pickle format used by the
original per-mid loading pipeline.

Usage:
    python prepare_data/generate_cshape.py --output datasets/occu_g_25/cshape.pkl
    python prepare_data/generate_cshape.py --output datasets/occu_g_25/cshape.pkl --grid_size 25 --grid_unit 0.08
"""

import argparse
import os
import pickle

import numpy as np


def generate_cshape_occupancy(grid_size=25, grid_unit=0.08):
    """
    Generate the C-shaped room occupancy grid.

    Args:
        grid_size: base grid size (doubled for the global grid), default 25
        grid_unit: voxel side length in meters, default 0.08

    Returns:
        (occu_g, unit, llb) tuple matching the original pipeline format:
        - occu_g: np.ndarray of bool, shape [2*grid_size, 2*grid_size, 2*grid_size]
        - unit: float, voxel side length in meters
        - llb: np.ndarray of shape [1, 3], lower-left-back corner in world coords
    """
    if isinstance(grid_size, int):
        grid_size = [grid_size] * 3
    grid_size = [int(s) * 2 for s in grid_size]  # [50, 50, 50] for default 25

    grid_unit = float(grid_unit)
    custom_shape = tuple(grid_size)
    custom_llb = np.array(
        [[-grid_size[0] * grid_unit / 2,
          -grid_size[1] * grid_unit / 4,
          -grid_size[2] * grid_unit / 8]],
        dtype=np.float32,
    )

    occu_g = np.zeros(grid_size, dtype=bool)
    occu_g[..., :6] = 1          # ground (z indices 0-5)
    occu_g[..., 40:] = 1         # ceiling (z indices 40-49)
    occu_g[:10, ...] = 1         # left wall (x indices 0-9)
    occu_g[-2:, ...] = 1         # right wall (x indices 48-49)
    occu_g[:, :6, ...] = 1       # back wall (y indices 0-5)
    occu_g[:, -16:, ...] = 1     # front wall (y indices 34-49)
    occu_g[25:, 20:24, ...] = 1  # inner wall (x >= 25, y in [20, 23])

    return occu_g, grid_unit, custom_llb


def main():
    parser = argparse.ArgumentParser(
        description="Generate C-shaped room occupancy as a .pkl file"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output .pkl file path"
    )
    parser.add_argument(
        "--grid_size", type=int, default=25,
        help="Base grid size (doubled for global grid, default: 25 -> 50x50x50)"
    )
    parser.add_argument(
        "--grid_unit", type=float, default=0.08,
        help="Voxel side length in meters (default: 0.08)"
    )
    args = parser.parse_args()

    occu_g, unit, llb = generate_cshape_occupancy(
        grid_size=args.grid_size, grid_unit=args.grid_unit
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump((occu_g, unit, llb), f)

    occu_ratio = occu_g.sum() / occu_g.size
    print(f"C-shape occupancy saved to {args.output}")
    print(f"  Shape:       {occu_g.shape}")
    print(f"  Unit:        {unit} m")
    print(f"  llb:         {llb[0]}")
    print(f"  rub:         {llb[0] + np.array(occu_g.shape) * unit}")
    print(f"  Occupied:    {occu_g.sum():,} / {occu_g.size:,} voxels ({occu_ratio:.2%})")


if __name__ == "__main__":
    main()
