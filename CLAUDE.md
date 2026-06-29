# Paper & Code Mapping

This repo implements **"Revisit Human-Scene Interaction via Space Occupancy"** (ECCV 2024, paper PDF at paper/). The core insight: static Human-Scene Interaction is fundamentally about interacting with **space occupancy** — a chair is just chair-shaped solid space. This allows training on motion-only data by treating the empty space around the human as "pseudo-scene occupancy."

## Key Contributions (Paper → Code)

### 1. Motion Occupancy Base (MOB) — Section 3 → `prepare_data/`
- Aggregates 13 datasets (98k motion clips) into paired human-occupancy data
- Pipeline: `number_data.py` → `extract_data_snippets.py` → `generate_data.py`
- Pseudo-scene occupancy: `O = 1 - Ô` where `Ô` is the voxelized human mesh
- MOB creates **harder** occupancy layouts than real rooms; training on hard cases generalizes to easier real scenes

### 2. Motion State Representation — Section 4.2 → `prepare_data/generate_data.py`
Each frame is stored as a 270-dim flat vector = `[132(j6d) + 66(jego) + 66(jabs) + 3(traj) + 3(fdir)]`:
- `j_6d [22×6]`: SMPL joint rotations in 6D continuous representation (Zhou et al. 2019)
- `j_ego [22×3]`: Joint positions in **egocentric** canonical frame (root-relative, facing-aligned)
- `j_abs [22×3]`: Joint positions in **global** coordinate (for computing target offsets)
- `traj [3]`: Root trajectory in egocentric canonical
- `fdir [3]`: Forward direction vector in egocentric canonical

All positions are canonicalized via `Cano(·, t)` — translated so root XY = (0,0) at frame t, rotated so facing = +Y.

### 3. Canonical Occupancy — Section 4.3 → `training/utils/occu.py`
- 25×25×25 binary voxel grid (8cm/unit → 2m×2m×2m coverage)
- Grid center offset s/4 forward (Y+6 cells): `grid[..., 1] += (size-1)/4.0` in `get_grid()`
- Queried at each frame in the human's current egocentric frame via `query_occu_batched()`
- Occupational grid is recomputed per frame, enabling dynamic scene handling even though trained on static data

### 4. CtrlTransf Architecture — Section 4.4 → `training/models/ctrl_transf.py`
- **Input**: n control groups projected to 512-dim via separate linear layers → stacked into [bs, n_ctrl, 512]
- **Encoder**: `TransformerBlock` × `MID_LAYERS` (2 by default, 8 heads, ELU activation)
- **Decoder**: `ToVecTransfBlock` — learnable query cross-attends to encoder output, outputs [bs, 512]
- **Output**: Linear → 298-dim: `[132(j6d) + 66(jego) + 66(jevel) + 3(nxt_traj) + 3(vel) + 2(fut_fdir_xy) + 2(fdir_vel_cs) + 12(nxt_limb_traj) + 12(limb_vel)]`
- Optional random noise injection (RANDOM config): concatenates noise to latent before output projection

### 5. Field Regulation — Section 4.4 Eq.4-6 → `training/utils/occu.py` + `ctrl_transf.py`
Novel differentiable collision avoidance:
- `project()` computes velocity projection onto each occupied-voxel direction
- `dist_func()` applies inverse-distance penalty: `max(0, 1/|v_ij|^0.85 - 1.6)`
- `get_delta_v()` sums all corrections → added to predicted root velocity in `ctrl_transf.py` forward pass
- Only operates in the **horizontal plane** (`d_vecs[..., 2] = 0.0`) — collision avoidance for walking
- Ignores voxels too low (<0.1m) or too high (>1.6m) to reduce noise
- α = ALPHA_COEFF / grid_dim (6 / 15625 = 0.000384) — stiffness factor `k` from Eq.6

### 6. Training — Section 4.5 → `training/train.py`
- **L_mix** (Eq.7): L1 on rotations (j6d, fdir) + L2 on positions (jego, traj, vel)
- **L_pen** (Eq.8): Penalizes predicted joints falling in occupied voxels
- **L_field** (Eq.9): `(|Δṗ|/|ṗ|)² + |ṗ|²` computed inside the model (floss + vloss), α=2, β=1
- **Scheduled Sampling**: teacher forcing ratio linearly increases 0→0.8 over first 30 epochs
- **Optimizer**: AdamW + CosineAnnealingWarmRestarts, lr=1e-4, 75k iterations, ~6h on Titan Xp
- **Augmentation**: random masking of control signals (target, voxel) during training

### 7. Auto-regressive Inference — Section 4.5 → `training/infer.py`
- Operates at **10 FPS** (downsampled from 30 FPS mocap)
- `PAST_KF=1, FUTURE_KF=1` — empirically more stable than finer granularity
- Loop: query occupancy → predict next frame → update global pose → roll state forward
- Tracks global root position and facing direction; network only sees canonical inputs
- Supports early stopping when target distance < threshold

## Data Pipeline Overview
```
Raw AMASS .npz/.pkl
  → number_data.py      (assign numeric IDs)
  → extract_data_snippets.py  (find valid frame ranges → mid_snip_dict.pkl)
  → generate_data.py    (canonicalize, extract 5-channel → .npy files)
  → split_data.py       (90/10 train/test split)
  → train.py --calc_norm (compute mean/std for normalization)
  → train.py            (train CtrlTransf on MOB)
  → infer.py            (autoregressive rollout + MP4 visualization)
```

## Key Joint Indices (ELIMBS)
`(0, 10, 11, 20, 21)` = root, left foot, right foot, left wrist, right wrist — used for target specification and metric computation.

## Joint Count: 22 vs 23 vs 24

The project uses **22 SMPL joints** internally (missing hand joints 22=left_hand, 23=right_hand from the full 24-joint SMPL), but the SMPL model itself expects **23 body joints** (69 dims for body_pose = 23×3). This mismatch is a frequent source of bugs:

| Context | Joints | body_pose dims | Usage |
|---------|--------|---------------|-------|
| Data (.npy) | 22 | 66 (22×3) | Motion state, network input/output |
| SMPL model | 23 | 69 (23×3) | `posedirs` expects 207 dims (23×3×3) |
| AMASS export | 24 | 72 (24×3) | Standard SMPL format with hand joints |

**Critical fix**: Always pass `pad_bpose=True` to `smpl_forward()` when feeding 22-joint poses — it zero-pads missing joints to identity rotation. Without this, `poses[..., 3:72]` silently truncates and SMPL's matrix multiply fails with shape mismatch (`1×189` vs `207×20670`).

## Reproduction Bug Fixes (Han Yi's Contributions)

These are issues found and fixed during independent reproduction of the paper:

### 1. SMPL Dimension Mismatch (`32ca7a2`)
- **File**: `training/utils/infer_utils.py` — `get_pose()`
- **Fix**: Added `pad_bpose=True` to the `smpl_forward()` call
- **Root cause**: Poses from the network have 22 joints → 66 dims body_pose, but SMPL's `posedirs` expects 69 dims (23 joints). The `pad_bpose=True` flag pads the missing joint(s) to zero (identity rotation).

### 2. FileNotFoundError & Variable Name Errors (`41380d5`)
- **Files**: `prepare_data/extract_data_snippets.py`, `prepare_data/generate_data.py`
- **Fixes**: Corrected file paths and resolved variable naming inconsistencies in the data preparation pipeline.

### 3. KeyError When USE_VOX=False (`815b89c`)
- **File**: `training/infer.py` — `draw_batch()`
- **Fix**: Made `occug`/`llb`/`unit` keys conditional — they were unconditionally accessed but only exist when `USE_VOX=True`.

### 4. Missing Import (`f02df4c`)
- **File**: `training/utils/infer_utils.py`
- **Fix**: Added `from matplotlib.lines import Line2D` — needed for the legend feature.

### 5. Legend in Visualization (`c09b46a`)
- **File**: `training/utils/infer_utils.py` — `draw_seq()`
- **Changes**: Added color-coded legend (green=Start, red=End/Target, blue=Predicted). Fixed `global update_lines` → `nonlocal update_lines` (a Python scoping bug in the nested `update()` function).

## Reproduction Feature Additions

### 6. AMASS-format NPZ Export (`a19cf53`)
- **File**: `training/infer.py`
- **`build_full_aa_pose(j6d, fdir, device)`**: Converts 22-joint 6D canonical rotations → 24-joint SMPL axis-angle (72 dims). Joints 22 (left_hand) and 23 (right_hand) are zero-padded since the model does not predict hand motion.
- **`save_npz(draw_dict, save_dir)`**: Saves generated motions as standard AMASS `.npz` files with fields: `poses [frames, 72]`, `trans [frames, 3]`, `betas [16]`, `gender`, `mocap_framerate=10.0`. Naming: `gen_no{idx}_{mid}_{st_fid}_{end_fid}.npz`.
- AA poses and global translations are accumulated frame-by-frame during the autoregressive loop, then stacked and (optionally) truncated for early stopping.

### 7. File-based Global Occupancy Loading (`ad13ba7`)
- **File**: `training/dataset.py` — `MphaseDataset.__init__()`
- **Two modes** controlled by `ASSETS.OCCU_FILE` in config:
  1. **Single-file broadcast**: `OCCU_FILE=cshape.pkl` → one `.pkl` loaded and broadcast to all motion IDs
  2. **Per-mid with fallback**: Looks for `{mid:08d}.pkl`, falls back to `default.pkl` if missing
- **New pipeline**: MuJoCo XML scenes → voxel → `.pkl` occupancy files
  - `prepare_data/mujoco_voxelize.py`: Parses MuJoCo XML (box/cylinder/plane geoms), voxelizes at 8cm unit with XY margin for free space outside walls
  - `prepare_data/generate_cshape.py`: Standalone C-shaped room `.pkl` generator
  - `scripts/generate_all_occupancy.py`: Batch script for all room XML files

### 8. Target Offset Inference (`b7e5f0c`)
- **File**: `training/infer.py` — `infer()`
- **`tgt_offset` parameter**: Translates the target pose along the character's facing direction by `tgt_offset` meters. The target translation is applied in global space: `jabs_tgt = jabs_tgt + crt_fdir * tgt_offset`.
- **Sweep**: The main block runs inference across `np.arange(0.5, 1.5, 0.5)` offsets, saving each to a separate subdirectory (`offset_0_5/`, `offset_1_0/`).
- **Debug logging**: First iteration of the autoregressive loop prints a complete summary of network I/O dimensions, ctrl_dict keys/shapes, and output tensor layout — critical for verifying data flow correctness when debugging.

### 9. Condensed Environment YAML (`afd7d84`)
- **File**: `environment.yaml` (repo root)
- Single conda environment for the entire project: `python=3.11`, `pytorch-cuda=12.1`, `pytorch3d`, `omegaconf`, `matplotlib`, `scipy`, `trimesh`, `joblib`, `chumpy`.

## Autoregressive Inference Data Flow

Understanding the per-frame loop in `infer()` is essential for debugging:

```
Frame t:
  1. Prepare canonical inputs (j6d, jego, jevel, past_traj, past_vel, past_fdir, tgt_limbs, voxel)
  2. Normalize inputs
  3. vec_to_ctrl() → ctrl_dict with keys: joints, traj, [tgt], [vox]
  4. network(ctrl_dict) → pred_vec [bs, 298]
  5. Denormalize output
  6. Extract: nxt_j6d, nxt_jego, nxt_jevel, nxt_traj, vel, fdir_vel from pred_vec
  7. Update global state:
     - nxt_pos = crt_pos + qrot(fdir_quat, nxt_traj)  (canonical→global)
     - nxt_fdir via delta rotation from fdir_vel
     - nxt_jabs via SMPL forward pass (if USE_SMPL_UPDATE)
  8. Roll state forward: crt_* ← nxt_*, past_* ← updated with new frame
  9. (Optional) Early stop if distance to target < threshold
```

Key insight: the network always operates in **egocentric canonical space** — the global→canonical conversion happens at input assembly (step 3), and the canonical→global conversion happens at state update (step 7). The network never sees global coordinates.

## Manual Mode — Quick Start

Dataset-free inference with customizable start/target. Body pose from file; position/orientation from CLI.

```bash
# Single run: walk from origin to (0, 3, 0.9)
python training/infer.py manual \
    -c checkpoints/dvox_realf_6alpha_all_wodrop/dvox_realf_6alpha_all_wodrop.yaml \
    --init_path pose_data/stand.npz \
    --init_pos 0.0,0.0,0.90 \
    --tgt_path pose_data/stand.npz \
    --tgt_root 0.0,3.0,0.90 \
    --occu_path datasets/occu_g_25/room_hanyi.pkl \
    --name step_001

# Closed-loop planning (--no_video for speed)
python training/infer.py manual \
    -c checkpoints/.../config.yaml \
    --init_path step_001.npz \
    --init_pos <new_root> \
    --tgt_path pose_data/stand.npz \
    --tgt_root <next_target> \
    --occu_path datasets/occu_g_25/room.pkl \
    --name step_002 --no_video
```

Output: `{output_dir}/{name}.npz` (SMPL + internal fast-path fields) + `.mp4`.
See `python training/infer.py manual --help` for all options. Defaults in `INFER.MANUAL.*`.