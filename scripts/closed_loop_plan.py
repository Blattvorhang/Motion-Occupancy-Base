#!/usr/bin/env python3
"""MOB planner wrapper for closed-loop control.

Thin CLI → ``infer_manual()`` adapter.  Accepts standard args from the
orchestrator and writes an AMASS-format .npz with project-internal fields
(``_j6d``, ``_jego``, ``_jabs``, ``_fdir``) for fast-path state passing.

Usage::

    conda run -n hoi python scripts/closed_loop_plan.py \
        --init_path /tmp/start.npz --init_frame 7 \
        --init_pos 1.2,3.4,0.78 --init_fdir 0.0,1.0,0.0 \
        --tgt_limb /tmp/tgt_limb_abs.npy \
        --occ_grid room_hanyi \
        --output_npz /tmp/seq_new.npz
"""

import argparse
import os
import sys
import pickle as _pickle
import numpy as np
import torch

# Ensure training/ is on sys.path so that ``from models import ...`` works.
_project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_training_dir = os.path.join(_project_root, "training")
if _training_dir not in sys.path:
    sys.path.insert(0, _training_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from omegaconf import OmegaConf
from models import get_model
from utils.data_misc import trunc_norm
from utils.infer_utils import get_pose       # noqa: F401 (used inside infer_manual)
from utils.occu import get_grid
from utils.train_utils import (set_io_dims, get_from_outputs,
                               get_foot_contact, get_ss_past_traj,
                               get_ss_past_vel, get_ss_tgt,
                               pad_fcontact_norm, vec_to_ctrl,
                               update_pos, update_fdir,
                               get_limb_from_outputs, get_ss_past_limb_traj)
from utils.quaternion import *
import smplx1 as smplx
from dataset import ELIMBS
import infer as _infer_mod
from infer import infer_manual, load_smpl_npz, save_npz
from utils.utils import create_logger


def _parse_vec(s: str):
    """Parse comma-separated float string → tensor [3]."""
    return torch.tensor([float(x.strip()) for x in s.split(",")])


def main():
    parser = argparse.ArgumentParser(
        description="MOB planner — closed-loop wrapper")
    parser.add_argument("--init_path", type=str, required=True,
                        help="Path to start-state .npz (single-frame or multi-frame)")
    parser.add_argument("--init_frame", type=int, default=0,
                        help="Frame index within init_path (0-indexed)")
    parser.add_argument("--init_pos", type=str, required=True,
                        help="Start root position x,y,z (Z-up)")
    parser.add_argument("--init_fdir", type=str, required=True,
                        help="Start forward direction x,y,z (Z-up)")
    parser.add_argument("--tgt_limb", type=str, required=True,
                        help="Path to target ELIMBS .npy [5,3] (Z-up world)")
    parser.add_argument("--occ_grid", type=str, required=True,
                        help="Occupancy scene name (room_zixuan, room_hanyi, cshape)")
    parser.add_argument("--output_npz", type=str, required=True,
                        help="Output .npz path")
    parser.add_argument("--history_path", type=str, default="",
                        help="Path to _history.npy from previous cycle")
    parser.add_argument("--infer_len", type=int, default=None,
                        help="Frames to generate (default: from config or calculated)")
    parser.add_argument("--device", type=int, default=None,
                        help="CUDA device index")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Config
    # ------------------------------------------------------------------
    config_path = os.path.join(_training_dir, "configs", "config.yml")
    config = OmegaConf.load(config_path)
    config.STAGE = "INFER"

    if config.INFER.get("IGNORE_WARNINGS", False):
        import warnings
        warnings.filterwarnings("ignore")

    # GRID_SIZE: int → list (matching infer.py __main__)
    if isinstance(config.TRAIN.GRID_SIZE, int):
        config.TRAIN.GRID_SIZE = [config.TRAIN.GRID_SIZE] * 3

    # Override checkpoint and split dir for this deployment
    # Resolve all relative ASSETS paths against the MOB project root
    # (required because subprocess CWD may differ from MOB project root)
    config.ASSETS.CHECKPOINT = os.path.join(
        _project_root, "checkpoints/dvox_realf_6alpha_all_wodrop/epoch_150.pt")
    for key in ["SPLIT_DIR", "OCCUG_DIR", "SMPL_DIR", "OCCUG_REF_DIR",
                "RESULT_DIR", "NPY_DIR", "BASIS_PATH"]:
        if key in config.ASSETS:
            val = config.ASSETS[key]
            if not os.path.isabs(val):
                config.ASSETS[key] = os.path.join(_project_root, val)
    # Ensure SPLIT_DIR points to the correct split
    config.ASSETS.SPLIT_DIR = os.path.join(_project_root, "datasets/splits/SSM")

    # Setup logger (infer_manual/infer/save_npz reference infer.logger as a global)
    _infer_mod.logger = create_logger(config, to_file=False)

    # Device
    device_idx = args.device if args.device is not None else config.DEVICE
    device_str = f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    config.DEVICE_STR = device_str
    print(f"[MOB] Device: {device_str}")

    # ------------------------------------------------------------------
    # 2. Network & norm
    # ------------------------------------------------------------------
    PAST_KF = config.TRAIN.PAST_KF
    FUTURE_KF = config.TRAIN.FUTURE_KF
    IN_DIM, OUT_DIM = set_io_dims(config)

    print("[MOB] Loading network...")
    network = get_model(config)
    ckpt = torch.load(config.ASSETS.CHECKPOINT, map_location="cpu")
    load_res = network.load_state_dict(ckpt, strict=False)
    print(f"[MOB] Checkpoint loaded | missing={len(load_res.missing_keys)} "
          f"unexpected={len(load_res.unexpected_keys)}")
    network.to(device)
    network.eval()
    torch.set_grad_enabled(False)

    mean_std = torch.load(
        os.path.join(config.ASSETS.SPLIT_DIR, config.ASSETS.MEAN_STD_NAME),
        map_location="cpu")
    input_mean, input_std, output_mean, output_std = [
        v.to(device) for k, v in list(mean_std.items())]
    if config.TRAIN.get("USE_FCONTACT", False):
        input_mean, input_std = pad_fcontact_norm(input_mean, input_std)
    input_mean, input_std, output_mean, output_std = trunc_norm(
        config, input_mean, input_std, output_mean, output_std)
    norm = (input_mean, input_std, output_mean, output_std)

    # ------------------------------------------------------------------
    # 3. SMPL body model
    # ------------------------------------------------------------------
    bm = smplx.create(config.ASSETS.SMPL_DIR, model_type="smpl",
                      gender="male", num_betas=16).to(device)

    # ------------------------------------------------------------------
    # 4. Start state
    # ------------------------------------------------------------------
    init_pos = _parse_vec(args.init_pos).to(device)
    init_fdir = _parse_vec(args.init_fdir).to(device)
    print(f"[MOB] Start: pos={init_pos.tolist()}, fdir={init_fdir.tolist()}")

    init_j6d, init_jego, init_jabs, betas = load_smpl_npz(
        args.init_path, init_pos, init_fdir, bm, device, frame=args.init_frame)
    print(f"[MOB] Loaded init from {args.init_path}[{args.init_frame}]: "
          f"j6d={init_j6d.shape}, jego={init_jego.shape}")

    # For closed-loop state passing: reset root rotation to identity.
    # --init_fdir defines the facing direction.  The body-joint rotations
    # (indices 1..21) are preserved for pose continuity; only the root is
    # reset so that SMPL global orientation at frame 0 = init_fdir exactly.
    _identity_root_6d = torch.tensor([1., 0., 0.,  0., 1., 0.],
                                     device=device, dtype=init_j6d.dtype)
    init_j6d[:, 0, :] = _identity_root_6d

    # ------------------------------------------------------------------
    # 4a. History state (autoregressive continuity)
    # ------------------------------------------------------------------
    history = None
    if args.history_path and os.path.exists(args.history_path):
        raw = np.load(args.history_path, allow_pickle=True).item()
        old_fdir = torch.from_numpy(raw['_final_fdir']).float().to(device)  # [1,3]
        g1_fdir = init_fdir.unsqueeze(0)  # [1,3]

        # Compute rotation from old canonical frame → new (G1) canonical frame
        old_quat = fdir_to_quat(old_fdir)
        new_quat = fdir_to_quat(g1_fdir)
        delta_quat = qmul(new_quat, qinv(old_quat))  # [1, 4]

        # Re-canonicalise root-related buffers to G1's actual frame
        for key in ['past_traj', 'past_vel', 'past_fdir', 'past_fdir_vel']:
            if key in raw:
                t = torch.from_numpy(raw[key]).float().to(device)   # [1, N, 3]
                raw[key] = qrot(delta_quat[:, None, :], t).cpu().numpy()
        # Limb buffers are also in canonical frame, need the same rotation.
        # Handle arbitrary dimensionality with explicit expand.
        for key in ['past_limb_traj', 'past_limb_vel']:
            if key in raw:
                t = torch.from_numpy(raw[key]).float().to(device)   # e.g. [1,2,4,3]
                n_extra = t.dim() - 2  # 2 for N=1 (bs=1, then spatial dims)
                dq = delta_quat.view(1, *([1] * n_extra), 4).expand(*t.shape[:-1], 4)
                raw[key] = qrot(dq, t).cpu().numpy()

        # crt_jevel stays untouched (joint-local velocity)
        del raw['_final_fdir']
        history = raw
        print(f"[MOB] History loaded from {args.history_path} "
              f"(old_fdir→G1_fdir rotation applied)")

    # ------------------------------------------------------------------
    # 5. Target
    # ------------------------------------------------------------------
    tgt_limb_np = np.load(args.tgt_limb)
    tgt_limb_abs = torch.from_numpy(tgt_limb_np).float().to(device).unsqueeze(0)  # [1, 5, 3]
    print(f"[MOB] Target: limbs={tgt_limb_abs.shape} from {args.tgt_limb}")

    # ------------------------------------------------------------------
    # 6. Occupancy
    # ------------------------------------------------------------------
    occu_path = os.path.join(config.ASSETS.OCCUG_DIR, f"{args.occ_grid}.pkl")
    print(f"[MOB] Loading occupancy: {occu_path}")
    with open(occu_path, "rb") as f:
        occu_g_np, unit, llb = _pickle.load(f)
    occu_g = torch.from_numpy(occu_g_np).float()
    llb = torch.from_numpy(llb).float()
    print(f"[MOB] Occupancy: shape={occu_g_np.shape}, unit={unit}")

    # ------------------------------------------------------------------
    # 7. Inference length
    # ------------------------------------------------------------------
    infer_len = args.infer_len
    if infer_len is None:
        infer_len = config.INFER.get("INFER_LEN", 100)
    print(f"[MOB] infer_len={infer_len} @ {config.TRAIN.GRID_SIZE[0]}³ grid")

    # ------------------------------------------------------------------
    # 8. Infer
    # ------------------------------------------------------------------
    print(f"[MOB] Calling infer_manual...")
    t0 = __import__("time").perf_counter()
    draw_dict = infer_manual(
        bm, network, config, norm,
        init_pos.unsqueeze(0), init_fdir.unsqueeze(0),
        init_j6d, init_jego,
        tgt_limb_abs, occu_g, llb, unit,
        infer_len, device, betas=betas, history=history,
        save_history_frame=args.init_frame)
    elapsed = __import__("time").perf_counter() - t0
    n_frames = draw_dict["aa_poses"].shape[1]
    print(f"[MOB] infer_manual done: {n_frames} frames in {elapsed:.1f}s "
          f"({n_frames / elapsed:.1f} fps)")

    # ------------------------------------------------------------------
    # 9. Save output
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output_npz) or ".", exist_ok=True)
    aa_poses = draw_dict["aa_poses"][0].cpu().numpy().astype(np.float64)  # [frames, 72]
    trans = draw_dict["trans"][0].cpu().numpy().astype(np.float64)         # [frames, 3]
    poses_out = np.zeros((n_frames, 156), dtype=np.float64)
    poses_out[:, :72] = aa_poses  # root(3) + 21 body(63) + 2 hands(6) = 72
    betas_np = betas[0].cpu().numpy().astype(np.float64) if betas is not None else np.zeros(16, dtype=np.float64)

    out_dict = {
        "poses": poses_out.astype(np.float32),
        "trans": trans.astype(np.float32),
        "betas": betas_np.astype(np.float32),
        "gender": np.array("male"),
        "mocap_framerate": np.float32(10.0),
        # Internal fields for fast-path state passing (Z-up, canonical)
        "_j6d": draw_dict["_j6d"][0].cpu().numpy().astype(np.float64),
        "_jego": draw_dict["_jego"][0].cpu().numpy().astype(np.float64),
        "_jabs": draw_dict["seq"][0].cpu().numpy().astype(np.float64),
        "_fdir": draw_dict["_fdir"][0].cpu().numpy().astype(np.float64),
    }
    np.savez(args.output_npz, **out_dict)
    size_kb = os.path.getsize(args.output_npz) / 1024
    print(f"[MOB] Saved: {args.output_npz} ({n_frames} fr, {size_kb:.0f} KB)")

    # Save autoregressive history for next cycle
    if '_history' in draw_dict and '_final_fdir' in draw_dict:
        history_path = args.output_npz.replace('.npz', '_history.npy')
        save_hist = {}
        for k, v in draw_dict['_history'].items():
            save_hist[k] = v.cpu().numpy() if hasattr(v, 'numpy') else v
        save_hist['_final_fdir'] = draw_dict['_final_fdir'].cpu().numpy()
        np.save(history_path, save_hist)
        print(f"[MOB] History saved: {history_path}")


if __name__ == "__main__":
    main()
