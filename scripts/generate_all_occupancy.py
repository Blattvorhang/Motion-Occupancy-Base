#!/usr/bin/env python3
"""
Generate all occupancy .pkl files in one go.

Produces:
    assets/occupancy/cshape.pkl        — hard-coded C-shaped room
    assets/occupancy/room_hanyi.pkl    — room_hanyi from MuJoCo XML
    assets/occupancy/room_zixuan.pkl   — room_zixuan from MuJoCo XML

Usage:
    python scripts/generate_all_occupancy.py
    python scripts/generate_all_occupancy.py --output_dir assets/occupancy
"""

import argparse
import os
import sys

# Allow importing from prepare_data
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prepare_data.generate_cshape import generate_cshape_occupancy
from prepare_data.mujoco_voxelize import mujoco_xml_to_occupancy
import pickle


def main():
    parser = argparse.ArgumentParser(
        description="Generate all occupancy .pkl files"
    )
    parser.add_argument(
        "--output_dir", type=str, default="assets/occupancy",
        help="Output directory for .pkl files"
    )
    parser.add_argument(
        "--margin", type=float, default=2.0,
        help="Free-space margin for XML rooms in meters"
    )
    args = parser.parse_args()

    project_root = os.path.join(os.path.dirname(__file__), "..")
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 1. C-shape room
    print("=" * 60)
    print("Generating C-shape room occupancy...")
    cshape_path = os.path.join(output_dir, "cshape.pkl")
    occu_g, unit, llb = generate_cshape_occupancy(grid_size=25, grid_unit=0.08)
    with open(cshape_path, "wb") as f:
        pickle.dump((occu_g, unit, llb), f)
    print(f"  Saved: {cshape_path}")
    print(f"  Shape: {occu_g.shape}, Occupied: {occu_g.sum():,}/{occu_g.size:,} ({occu_g.sum()/occu_g.size:.2%})")

    # 2. room_hanyi
    print("=" * 60)
    print("Generating room_hanyi occupancy...")
    hanyi_xml = os.path.join(project_root, "assets", "room_hanyi", "room.xml")
    hanyi_path = os.path.join(output_dir, "room_hanyi.pkl")
    occu_g, unit, llb = mujoco_xml_to_occupancy(hanyi_xml, unit=0.08, margin_m=args.margin)
    with open(hanyi_path, "wb") as f:
        pickle.dump((occu_g, unit, llb), f)
    print(f"  Saved: {hanyi_path}")
    print(f"  Shape: {occu_g.shape}, Occupied: {occu_g.sum():,}/{occu_g.size:,} ({occu_g.sum()/occu_g.size:.2%})")

    # 3. room_zixuan
    print("=" * 60)
    print("Generating room_zixuan occupancy...")
    zixuan_xml = os.path.join(project_root, "assets", "room_zixuan", "room.xml")
    zixuan_path = os.path.join(output_dir, "room_zixuan.pkl")
    occu_g, unit, llb = mujoco_xml_to_occupancy(zixuan_xml, unit=0.08, margin_m=args.margin)
    with open(zixuan_path, "wb") as f:
        pickle.dump((occu_g, unit, llb), f)
    print(f"  Saved: {zixuan_path}")
    print(f"  Shape: {occu_g.shape}, Occupied: {occu_g.sum():,}/{occu_g.size:,} ({occu_g.sum()/occu_g.size:.2%})")

    print("=" * 60)
    print("All occupancy files generated successfully.")


if __name__ == "__main__":
    main()
