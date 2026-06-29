import argparse
import sys
from functools import partial
# from multiprocessing import Pool, set_start_method
import os
import pickle
import random
import time
import numpy as np
from omegaconf import OmegaConf
import torch
from models import get_model
from utils.data_misc import trunc_norm
from utils.infer_utils import get_pose
from utils.metric_utils import calculate_distances
from utils.occu import get_grid, query_bps_batched, query_occu, query_bps, get_bps, query_occu_batched, query_occu_batched_field, project, get_delta_v, query_bps_batched_field
from utils.train_utils import calc_jegor, get_foot_contact, get_from_outputs, get_in_vec, get_limb_from_outputs, get_ss_past_limb_traj, get_ss_past_traj, get_ss_past_vel, get_ss_tgt, pad_fcontact_norm, set_io_dims, update_fdir, update_pos, vec_to_ctrl
from utils.infer_utils import *

from utils.quaternion import *
from utils.utils import backup_code, backup_config_file, create_logger
from dataset import ELIMBS, LIMBS, MphaseDataset, collate_fn, normalize, denormalize
import smplx1 as smplx

def build_full_aa_pose(j6d, fdir, device):
	"""
	Convert 22-joint 6D rotation (canonical frame) to 24-joint axis-angle (SMPL format).
	j6d: [bs, 22, 6] canonical 6D rotations
	fdir: [bs, 3] global facing direction
	Returns: [bs, 72] = global_orient[3] + body_pose[21 joints x 3 = 63] + hand_zeros[6]
	Joints 22 (left_hand) and 23 (right_hand) are zero-padded since the model does not predict them.
	"""
	bs = j6d.shape[0]
	aa = cont6d_to_aa(j6d)                          # [bs, 22, 3] axis-angle in canonical
	global_orient = calc_orient_in_global(aa[:, 0], fdir_to_rad(fdir))  # [bs, 3]
	body_pose = aa[:, 1:22].reshape(bs, 63)          # [bs, 63] parent-local, unchanged
	hand_zeros = torch.zeros(bs, 6, device=device)   # joints 22+23
	return torch.cat([global_orient, body_pose, hand_zeros], dim=1)


def _smp_to_project_positions(joints_smpl):
    """Convert joint positions from SMPL Y-up to project Z-up.
    joints_smpl: [..., 3] in SMPL frame (Y-up, -Z forward)
    Returns: [..., 3] in project frame (Z-up, +Y forward)
    Rotation +90° around X: [x, y, z] → [x, -z, y]
    """
    return torch.stack([joints_smpl[..., 0], -joints_smpl[..., 2], joints_smpl[..., 1]], dim=-1)


def _smp_to_project_6d(j6d_smpl):
    """Convert 6D rotations from SMPL Y-up to project Z-up.
    j6d_smpl: [..., J, 6] in SMPL frame
    Returns: [..., J, 6] in project frame
    Each rotation matrix column is transformed by [x, y, z] → [x, -z, y].
    """
    col0 = j6d_smpl[..., :3]  # first rotation matrix column
    col1 = j6d_smpl[..., 3:6]  # second rotation matrix column
    col0_zup = torch.stack([col0[..., 0], -col0[..., 2], col0[..., 1]], dim=-1)
    col1_zup = torch.stack([col1[..., 0], -col1[..., 2], col1[..., 1]], dim=-1)
    return torch.cat([col0_zup, col1_zup], dim=-1)


@torch.no_grad()
def load_smpl_npz(npz_path, init_pos, init_fdir, bm, device):
    """Load SMPL .npz and convert to project internal format.
    If .npz contains _j6d/_jego/_jabs/_fdir fields, uses them directly (fast path).
    Otherwise falls back to SMPL forward (slow path, handles external SMPL .npz).
    """
    data = np.load(npz_path, allow_pickle=True)
    betas_npz = torch.from_numpy(data['betas']).float().to(device).unsqueeze(0)  # [1, 16]

    # Fast path: project-internal fields available (from our own save_npz output)
    if '_j6d' in data and '_jego' in data and '_jabs' in data and '_fdir' in data:
        # Take first frame if multi-frame arrays are stored
        _j6d = torch.from_numpy(data['_j6d']).float().to(device)
        _jego = torch.from_numpy(data['_jego']).float().to(device)
        _jabs = torch.from_numpy(data['_jabs']).float().to(device)
        _fdir = torch.from_numpy(data['_fdir']).float().to(device)
        if _j6d.dim() == 3:  # [frames, 22, 6] → [22, 6]
            _j6d, _jego, _jabs, _fdir = _j6d[0], _jego[0], _jabs[0], _fdir[0]
        j6d = _j6d.unsqueeze(0)        # [1, 22, 6]
        jego_npz = _jego.unsqueeze(0)  # [1, 22, 3]
        jabs_npz = _jabs.unsqueeze(0)  # [1, 22, 3]
        fdir_npz = _fdir.unsqueeze(0)  # [1, 3]

        init_pos_t = init_pos.to(device).view(1, 3)
        init_fdir_t = init_fdir.to(device).view(1, 3)

        # Shift root to init_pos (preserve limb positions relative to root)
        jabs = jabs_npz - jabs_npz[:, 0:1, :] + init_pos_t

        # Canonicalize with init_fdir
        crt_quat = fdir_to_quat(init_fdir_t)
        jego = change_system(jabs, init_pos_t, crt_quat, keep_h=True)
        return j6d, jego, jabs, betas_npz

    # Slow path: standard SMPL .npz (from external sources)
    poses = torch.from_numpy(data['poses']).float().to(device)
    trans_npz = torch.from_numpy(data['trans']).float().to(device)

    aa_72 = poses[0:1]
    aa_24 = aa_72.reshape(1, 24, 3)
    quat_24 = aa_to_quat(aa_24)
    j6d_24 = quat_to_6d(quat_24)
    j6d = j6d_24[:, :22]

    body_pose_66 = aa_72[:, 3:69]
    smpl_out = smpl_forward(bm, orient=aa_72[:, :3], bpose=body_pose_66,
                           betas=betas_npz, rm_offset=True, pad_bpose=True)
    jabs_smpl = smpl_out['joints'] + trans_npz[0:1]
    jabs = _smp_to_project_positions(jabs_smpl)
    j6d = _smp_to_project_6d(j6d)

    # Canonicalize
    init_pos_t = init_pos.to(device).view(1, 3)
    init_fdir_t = init_fdir.to(device).view(1, 3)
    crt_quat = fdir_to_quat(init_fdir_t)
    jego = change_system(jabs, init_pos_t, crt_quat, keep_h=True)

    # Canonicalize root rotation (now in project Z-up)
    l2g_quat = fdir_to_quat(init_fdir_t)
    g2l_quat = qinv(l2g_quat)
    root_6d_zup = j6d[:, 0:1]
    root_quat_zup = cont6d_to_quat(root_6d_zup)
    root_quat_canonical = qmul(g2l_quat[:, None, :], root_quat_zup)
    root_6d_canonical = quat_to_6d(root_quat_canonical)
    j6d = torch.cat([root_6d_canonical.view(1, 1, 6), j6d[:, 1:22]], dim=1)

    return j6d, jego, jabs, betas_npz


def get_tgt_limbs_from_npz(npz_path, tgt_root, tgt_fdir, bm, device):
    """Load target SMPL .npz and extract ELIMBS positions.
    Uses project-internal _jabs field if available (fast path).
    """
    data = np.load(npz_path, allow_pickle=True)

    tgt_root_t = tgt_root.to(device).view(1, 3)
    tgt_fdir_t = tgt_fdir.to(device).view(1, 3)

    # Fast path: use project-internal jabs directly
    if '_jabs' in data and '_fdir' in data:
        _jabs = torch.from_numpy(data['_jabs']).float().to(device)
        _fdir = torch.from_numpy(data['_fdir']).float().to(device)
        if _jabs.dim() == 3:  # [frames, 22, 3] → [22, 3]
            _jabs, _fdir = _jabs[0], _fdir[0]
        jabs_npz = _jabs.unsqueeze(0)   # [1, 22, 3]
        fdir_npz = _fdir.unsqueeze(0)   # [1, 3]

        # Shift root to tgt_root, rotate by tgt_fdir
        root_old = jabs_npz[:, 0:1, :]
        jabs_shifted = jabs_npz - root_old  # root at origin
        # Rotate from old fdir to new fdir
        old_quat = fdir_to_quat(fdir_npz)
        new_quat = fdir_to_quat(tgt_fdir_t)
        delta_quat = qmul(new_quat, qinv(old_quat))
        jabs_rotated = qrot(delta_quat[:, None, :].expand(-1, 22, -1), jabs_shifted)
        jabs_global = jabs_rotated + tgt_root_t

        tgt_limb_abs = jabs_global[:, ELIMBS]
        return tgt_limb_abs, jabs_global

    # Slow path: standard SMPL .npz
    poses = torch.from_numpy(data['poses']).float().to(device)
    betas_npz = torch.from_numpy(data['betas']).float().to(device).unsqueeze(0)

    aa_72 = poses[0:1]
    global_orient = aa_72[:, :3]
    body_pose_66 = aa_72[:, 3:69]

    smpl_out = smpl_forward(bm, orient=global_orient, bpose=body_pose_66,
                           betas=betas_npz, rm_offset=True, pad_bpose=True)
    jabs_local_smpl = smpl_out['joints']
    jabs_local = _smp_to_project_positions(jabs_local_smpl)

    rad = fdir_to_rad(tgt_fdir_t)
    orient_global = calc_orient_in_global(global_orient, rad)
    smpl_out2 = smpl_forward(bm, orient=orient_global, bpose=body_pose_66,
                            betas=betas_npz, rm_offset=True, pad_bpose=True)
    jabs_global_smpl = smpl_out2['joints'] + tgt_root_t
    jabs_global = _smp_to_project_positions(jabs_global_smpl)

    tgt_limb_abs = jabs_global[:, ELIMBS]
    return tgt_limb_abs, jabs_global


@torch.no_grad()
def infer_manual(bm, network, config, norm, init_pos, init_fdir, init_j6d, init_jego,
                 tgt_limb_abs, occu_g, llb, unit, infer_len, device, betas=None):
    """Manual-mode inference with externally specified start and target.

    Zero dependency on MphaseDataset. All inputs passed explicitly.

    Args:
        bm: SMPL body model
        network: CtrlTransf model
        config: OmegaConf config
        norm: (input_mean, input_std, output_mean, output_std)
        init_pos: [1, 3] global root position
        init_fdir: [1, 3] global forward direction
        init_j6d: [1, 22, 6] 6D rotations in canonical
        init_jego: [1, 22, 3] ego-centric joint positions
        tgt_limb_abs: [1, 5, 3] target ELIMBS positions in global
        occu_g: occupancy grid tensor
        llb: lower-left-back corner
        unit: grid unit size
        infer_len: number of frames to generate
        device: torch device
        betas: [16] SMPL shape parameters (default: zeros)

    Returns:
        draw_dict compatible with draw_batch()
    """
    input_mean, input_std, output_mean, output_std = norm
    if config.TRAIN.get('USE_FIELD', False):
        vel_mean = output_mean[264:267].to(device)
        vel_std = output_std[264:267].to(device)
        vel_norm = (vel_mean, vel_std)
    network.eval()
    past_kf = config.TRAIN.PAST_KF
    future_kf = config.TRAIN.FUTURE_KF
    bs = 1  # manual mode always bs=1

    USE_EVEL = config.TRAIN.USE_EVEL
    USE_NORM = config.TRAIN.USE_NORM
    USE_FCONTACT = config.TRAIN.get('USE_FCONTACT', False)
    USE_LIMB_TRAJ = config.TRAIN.USE_LIMB_TRAJ
    USE_TGT = config.TRAIN.USE_TARGET
    USE_VOX = config.TRAIN.USE_VOX
    USE_BPS = config.TRAIN.get('USE_BPS', False)
    USE_SMPL_UPDATE = config.TRAIN.get('USE_SMPL_UPDATE', False) and bm is not None
    USE_FIELD = config.TRAIN.get('USE_FIELD', False)

    if USE_VOX:
        vox_device = device if config.TRAIN.VOX_ON_GPU else 'cpu'
        grid_size = config.TRAIN.GRID_SIZE
        grid, oidx, occu_l_grid = get_grid(unit=unit, size=grid_size, device=vox_device)
        grid = grid[None].expand(bs, -1, -1).contiguous()
        oidx = oidx[None].expand(bs, -1, -1).contiguous()
        occu_l_grid = occu_l_grid[None].expand(bs, -1, -1, -1).contiguous()
        occu_g = occu_g.to(vox_device).unsqueeze(0)  # [1, D, H, W]
        llb = llb.to(vox_device).unsqueeze(0)  # [1, 3]

    # Initialize state
    crt_j6d = init_j6d.to(device)
    crt_jego = init_jego.to(device)
    crt_jevel = torch.zeros(bs, 22, 3, device=device)

    crt_pos = init_pos.to(device).view(bs, 3)
    crt_fdir = init_fdir.to(device).view(bs, 3)

    height = crt_jego[:, 0, 2:3]
    limb_height = crt_jego[:, LIMBS, 2]

    past_traj = torch.zeros(bs, past_kf, 3, device=device)
    past_traj[..., 2] = height[:, None]
    past_vel = torch.zeros(bs, past_kf, 3, device=device)
    past_fdir = torch.zeros(bs, past_kf, 3, device=device)
    past_fdir[..., 1] = 1.0
    past_fdir_vel = torch.zeros(bs, past_kf, 3, device=device)
    past_fdir_vel[..., 0] = 1.0
    tgt_limb_abs = tgt_limb_abs.to(device).view(bs, 5, 3)

    if USE_LIMB_TRAJ:
        past_limb_traj = torch.zeros(bs, past_kf + 1, 4, 3, device=device)
        past_limb_traj[..., 2] = limb_height
        past_limb_vel = torch.zeros(bs, past_kf, 4, 3, device=device)

    pose_seq = [get_pose(cont6d_to_aa(crt_j6d).flatten(1), crt_pos, fdir_to_rad(crt_fdir), bm).cpu()]
    aa_pose_seq = [build_full_aa_pose(crt_j6d, crt_fdir, device)]
    trans_seq = [crt_pos.clone()]
    j6d_seq = [crt_j6d.clone()]
    jego_seq = [crt_jego.clone()]
    fdir_seq = [crt_fdir.clone()]
    if USE_TGT:
        rel_pos_seq = [get_abs_limbs(crt_pos, fdir_to_rad(crt_fdir),
                       rel_limb_pos=get_ss_tgt(tgt_limb_abs, crt_pos, crt_fdir)).cpu()]
    if USE_FIELD:
        ori_v_lst = []
        dv_lst = []

    logger.info(f'>>> INFER manual | bs={bs}, len={infer_len}, vox={USE_VOX}, field={USE_FIELD}, tgt={USE_TGT}')

    for i in range(infer_len):
        in_dict = {'j6d': crt_j6d, 'jego': crt_jego, 'jevel': crt_jevel}
        in_dict.update({'crt_pos': crt_pos})
        in_dict.update({'past_traj': past_traj, 'past_vel': past_vel,
                        'past_fdir': past_fdir, 'past_fdir_vel': past_fdir_vel})
        if USE_FCONTACT:
            fcontact = get_foot_contact(crt_jego)
            in_dict.update({'fcontact': fcontact})
        if USE_LIMB_TRAJ:
            in_dict.update({'past_limb_traj': past_limb_traj, 'past_limb_vel': past_limb_vel})
        if USE_TGT:
            tgt_limbs = get_ss_tgt(tgt_limb_abs, crt_pos, crt_fdir)
            in_dict.update({'tgt_limbs': tgt_limbs})

        if USE_VOX:
            if USE_FIELD:
                query_result = query_occu_batched_field(occu_g, llb, crt_pos, fdir_to_rad(crt_fdir),
                    unit=unit, grid_size=grid_size, device=vox_device, grid=grid, oidx=oidx, occu_l=occu_l_grid,
                    jego=crt_jego, config=config, tgt_limb_abs=tgt_limb_abs)
                occu_l_lst, d_vecs = query_result['occul'], query_result['d_vecs']
            else:
                occu_l_lst = query_occu_batched(occu_g, llb, crt_pos, fdir_to_rad(crt_fdir),
                                                unit, None, vox_device, grid, oidx, occu_l_grid)
            occu_l_lst = occu_l_lst.flatten(1).to(device)
            in_dict['occu_l'] = occu_l_lst

        in_vec = get_in_vec(in_dict)

        if USE_NORM:
            ndims = input_mean.shape[0]
            in_vec[:, :ndims] = normalize(in_vec[:, :ndims], input_mean, input_std)
        if config.TRAIN.SEP_CTRLS:
            ctrl_dict = vec_to_ctrl(config, in_vec)
            if USE_FIELD:
                vel = past_vel.view(-1, 3)
                pred_dict = network(ctrl_dict, d_vecs, vel_norm, vel, return_vel=True)
                pred_vec, ori_v, dv = pred_dict['pred'], pred_dict['vel'], pred_dict['dv']
                ori_v = qrot(fdir_to_quat(crt_fdir), ori_v)
                dv = qrot(fdir_to_quat(crt_fdir), dv)
                ori_v[..., 2] = 0.0
                dv[..., 2] = 0.0
                ori_v_lst.append(ori_v.cpu())
                dv_lst.append(dv.cpu())
            else:
                pred_vec = network(ctrl_dict)
        else:
            pred_vec = network(in_vec)

        if USE_NORM:
            pred_vec = denormalize(pred_vec, output_mean, output_std)
        nxt_j6d, nxt_jego, nxt_jevel, nxt_traj, vel, fdir_vel = get_from_outputs(pred_vec, future_kf)
        if USE_LIMB_TRAJ:
            nxt_limbs, nxt_limb_vel = get_limb_from_outputs(pred_vec, future_kf)

        nxt_pos = update_pos(crt_pos, crt_fdir, nxt_traj)
        nxt_fdir = update_fdir(crt_fdir, fdir_vel)
        abs_pose = get_pose(cont6d_to_aa(nxt_j6d).flatten(1), nxt_pos, fdir_to_rad(nxt_fdir), bm)
        if USE_SMPL_UPDATE:
            nxt_jego = get_pose(cont6d_to_aa(nxt_j6d).flatten(1), bm=bm)
        pose_seq.append(abs_pose.clone().cpu())
        aa_pose_seq.append(build_full_aa_pose(nxt_j6d, nxt_fdir, device))
        trans_seq.append(nxt_pos.clone())
        j6d_seq.append(nxt_j6d.clone())
        jego_seq.append(nxt_jego.clone())
        fdir_seq.append(nxt_fdir.clone())
        if USE_TGT:
            rel_pos_seq.append(get_abs_limbs(crt_pos, fdir_to_rad(crt_fdir), tgt_limbs).cpu())
        past_traj, past_vel = get_ss_past_traj(past_traj, past_vel, crt_pos[:, 2:3], vel, nxt_traj, fdir_vel)
        if USE_LIMB_TRAJ:
            past_limb_traj, past_limb_vel = \
                get_ss_past_limb_traj(past_limb_traj, nxt_traj, nxt_limbs, past_limb_vel, nxt_limb_vel, fdir_vel)
        past_fdir, past_fdir_vel = get_ss_past_vel(past_fdir, past_fdir_vel, fdir_vel)
        crt_j6d = nxt_j6d
        crt_jego = nxt_jego
        crt_jevel = nxt_jevel
        crt_pos = nxt_pos
        crt_fdir = nxt_fdir

    pose_seq = torch.stack(pose_seq, dim=1)  # [1, frames, 22, 3]
    starts = pose_seq[:, 0]
    rel_pos_seq = torch.stack(rel_pos_seq, dim=1)  # [1, frames, 5, 3]
    aa_pose_seq = torch.stack(aa_pose_seq, dim=1)  # [1, frames, 72]
    trans_seq = torch.stack(trans_seq, dim=1)  # [1, frames, 3]

    # Build draw_dict
    draw_dict = {'seq': pose_seq, 'start': starts}
    draw_dict['mid_snip'] = torch.tensor([[0, 0, 0, 0]])  # placeholder
    draw_dict['aa_poses'] = aa_pose_seq
    draw_dict['trans'] = trans_seq
    draw_dict['_j6d'] = torch.stack(j6d_seq, dim=1)  # [1, frames, 22, 6]
    draw_dict['_jego'] = torch.stack(jego_seq, dim=1)  # [1, frames, 22, 3]
    draw_dict['_fdir'] = torch.stack(fdir_seq, dim=1)  # [1, frames, 3]
    if USE_TGT:
        draw_dict['tgt_seq'] = rel_pos_seq
        # Create 22-joint end pose from 5 ELIMBS positions (rest zero)
        end_pose = torch.zeros(1, 22, 3)
        end_pose[:, ELIMBS] = tgt_limb_abs.cpu()
        draw_dict['end'] = end_pose
    if USE_VOX:
        draw_dict['occug'] = [occu_g.cpu().squeeze(0)]
        draw_dict['llb'] = llb
        draw_dict['unit'] = unit
    if USE_FIELD:
        ori_v_lst.append(torch.zeros_like(ori_v_lst[0]))
        dv_lst.append(torch.zeros_like(dv_lst[0]))
        ori_v_lst = torch.stack(ori_v_lst, dim=1)
        dv_lst = torch.stack(dv_lst, dim=1)
        draw_dict['ori_v'] = ori_v_lst
        draw_dict['dv'] = dv_lst
    return draw_dict


def infer(data, dataset, bm, network, config, norm):
	input_mean, input_std, output_mean, output_std = norm
	if config.TRAIN.get('USE_FIELD', False):
		vel_mean = output_mean[264:267].to(device) # get mean of vel
		vel_std = output_std[264:267].to(device) # get std of vel
		vel_norm = (vel_mean, vel_std)
	network.eval()
	past_kf = config.TRAIN.PAST_KF
	future_kf = config.TRAIN.FUTURE_KF
	infer_len = config.INFER.INFER_LEN
	bs = len(data['j_6d'])
	j6d  = data['j_6d'].to(device, non_blocking=True)
	jego = data['j_ego'].to(device, non_blocking=True)
	jabs = data['j_abs'].to(device, non_blocking=True)
	traj  = data['traj'].to(device, non_blocking=True)
	fdir  = data['fdir'].to(device, non_blocking=True)

	USE_EVEL = config.TRAIN.USE_EVEL
	USE_NORM = config.TRAIN.USE_NORM
	USE_FCONTACT = config.TRAIN.get('USE_FCONTACT', False)
	USE_LIMB_TRAJ = config.TRAIN.USE_LIMB_TRAJ
	USE_TGT = config.TRAIN.USE_TARGET
	USE_VOX = config.TRAIN.USE_VOX
	USE_BPS = config.TRAIN.get('USE_BPS', False)
	USE_SMPL_UPDATE = config.TRAIN.get('USE_SMPL_UPDATE', False) and bm is not None
	USE_FIELD = config.TRAIN.get('USE_FIELD', False)
	if config.TRAIN.USE_VOX:
		if config.TRAIN.BATCH_VOX:
			if config.TRAIN.get('PRECREATE_GRID', False):
				vox_device = device if config.TRAIN.VOX_ON_GPU else 'cpu'
				grid_size = config.TRAIN.GRID_SIZE
				grid, oidx, occu_l = get_grid(unit=config.TRAIN.GRID_UNIT, size=grid_size, device=vox_device)
				bs = len(j6d)
				grid = grid[None].expand(bs, -1, -1).contiguous()
				oidx = oidx[None].expand(bs, -1, -1).contiguous()
				occu_l = occu_l[None].expand(bs, -1, -1, -1).contiguous()
			occu_g = data['vox'].to(vox_device)
			llb    = data['llb'].to(vox_device)
			if USE_BPS:
				occu_g_ref = data['ref'].to(vox_device)
				occu_shape = data['shape'].to(vox_device)
				bps = torch.from_numpy(np.load(config.ASSETS.BASIS_PATH)).float().to(vox_device)
				bps = get_bps(bps, config.TRAIN.GRID_UNIT, config.TRAIN.GRID_SIZE)
		else:
			mid = data['mid'].tolist()
			llb = data['llb'].to('cpu')
			occu_g = [dataset.occu_g_dict[i][0].to('cpu') for i in mid]
	if USE_EVEL:
		crt_j6d, crt_jego, crt_jevel = get_g_joints_evel(j6d, jego, past_kf)
	else:
		crt_j6d, crt_jego, crt_jevel = get_g_joints(j6d, jego, jabs, fdir, past_kf, nxt_vel=config.TRAIN.USE_NXT_EVEL)
	crt_pos, crt_fdir = get_pos_dir(jabs, fdir, idx=past_kf)
	if USE_TGT:
		jabs_tgt = data['jabs_tgt'].to(device, non_blocking=True)
		tgt_limb_abs = jabs_tgt[:, ELIMBS]
	else:
		jabs_tgt = None
		tgt_limb_abs = None
	past_traj, past_vel = get_g_past_traj(traj, fdir, past_kf, crt_idx=past_kf)
	past_fdir, past_fdir_vel = get_g_past_fdir(fdir, past_kf, crt_idx=past_kf)
	if USE_LIMB_TRAJ:
		past_limb_traj, past_limb_vel = get_g_past_limb_traj(jabs, fdir, past_kf, crt_idx=past_kf)

	if config.INFER.get('SPECIFY_FIRST', False):
		first_pose = torch.load(config.INFER.FIRST_PATH)
		crt_j6d = first_pose[:22*6].view(1, 22, 6).expand(bs, -1, -1).to(device)
		crt_jego = first_pose[22*6:].view(1, 22, 3).expand(bs, -1, -1).to(device)
		height = crt_jego[:, 0, 2]
		limb_height = crt_jego[:, LIMBS, 2]
		crt_jevel = torch.zeros_like(crt_jevel)
		past_traj = torch.zeros_like(past_traj)
		past_traj[..., 2] = height[:, None]
		past_vel  = torch.zeros_like(past_vel)
		past_fdir = torch.zeros_like(past_fdir)
		past_fdir[..., 1] = 1.0
		past_fdir_vel = torch.zeros_like(past_fdir_vel)
		past_fdir_vel[..., 0] = 1.0
		if USE_LIMB_TRAJ:
			past_limb_traj = torch.zeros_like(past_limb_traj)
			past_limb_traj[..., 2] = limb_height[:, None]
			past_limb_vel = torch.zeros_like(past_limb_vel)
		crt_pos[:, 2] = height

	pose_seq = [get_pose(cont6d_to_aa(crt_j6d).flatten(1), crt_pos, fdir_to_rad(crt_fdir), bm).cpu()]
	aa_pose_seq = [build_full_aa_pose(crt_j6d, crt_fdir, device)]  # frame 0 (ground truth input)
	trans_seq = [crt_pos.clone()]
	if USE_TGT:
		rel_pos_seq = [get_abs_limbs(crt_pos, fdir_to_rad(crt_fdir), rel_limb_pos=get_ss_tgt(tgt_limb_abs, crt_pos, crt_fdir)).cpu()]
	if USE_FIELD:
		ori_v_lst = []
		dv_lst = []

	# ---- Timing accumulators ----
	t_input  = 0.0  # input assembly (in_dict, norm, get_in_vec, vec_to_ctrl)
	t_voxel  = 0.0  # voxel occupancy query
	t_model  = 0.0  # network forward pass
	t_smpl   = 0.0  # SMPL forward (get_pose)
	t_update = 0.0  # state update (pos, fdir, past buffers, roll-forward)
	t_total  = 0.0  # wall-clock for full loop

	# ---- Brisk network I/O summary (once) ----
	vox_str = ''
	if USE_VOX:
		gs = config.TRAIN.GRID_SIZE
		vox_str = f' | vox[{gs[0]*gs[1]*gs[2]}]'
	logger.info(f'>>> INFER bs={bs}, len={infer_len}, '
	            f'in={config.MODEL.IN_DIM}/out={config.MODEL.OUT_DIM}, '
	            f'vox={USE_VOX}, bps={USE_BPS}, field={USE_FIELD}, tgt={USE_TGT}{vox_str}')

	# ---- Main autoregressive loop ----
	loop_t0 = time.perf_counter()
	for i in range(infer_len):
		t0 = time.perf_counter()

		# --- Input preparation ---
		t_prep = time.perf_counter()
		in_dict = {'j6d': crt_j6d, 'jego': crt_jego, 'jevel': crt_jevel}
		in_dict.update({'crt_pos': crt_pos})
		in_dict.update({'past_traj': past_traj, 'past_vel': past_vel, 'past_fdir': past_fdir, 'past_fdir_vel': past_fdir_vel})
		if USE_FCONTACT:
			fcontact = get_foot_contact(crt_jego)
			in_dict.update({'fcontact': fcontact})
		if USE_LIMB_TRAJ:
			in_dict.update({'past_limb_traj': past_limb_traj, 'past_limb_vel': past_limb_vel})
		if USE_TGT:
			tgt_limbs = get_ss_tgt(tgt_limb_abs, crt_pos, crt_fdir)
			in_dict.update({'tgt_limbs': tgt_limbs})

		# --- Voxel query ---
		t_vox_start = time.perf_counter()
		if USE_VOX:
			if USE_BPS:
				if USE_FIELD:
					occu_l_lst, d_vecs = query_bps_batched_field(occu_g, llb, occu_g_ref, occu_shape, crt_pos, fdir_to_rad(crt_fdir), bps, unit=config.TRAIN.GRID_UNIT, device=vox_device)
				else:
					occu_l_lst = query_bps_batched(occu_g, llb, occu_g_ref, occu_shape, crt_pos, fdir_to_rad(crt_fdir), bps, unit=config.TRAIN.GRID_UNIT, device=vox_device)
			else:
				if USE_FIELD:
					query_result = query_occu_batched_field(occu_g, llb, crt_pos, fdir_to_rad(crt_fdir),
						unit=config.TRAIN.GRID_UNIT, grid_size=config.TRAIN.GRID_SIZE, device=vox_device, grid=grid, oidx=oidx, occu_l=occu_l,
						jego=crt_jego, config=config, tgt_limb_abs=tgt_limb_abs)
					occu_l_lst, d_vecs = query_result['occul'], query_result['d_vecs']
					if 'close' in query_result and config.TRAIN.get('CLOSE_LABEL', False):
						close = query_result['close']
				else:
					occu_l_lst = query_occu_batched(occu_g, llb, crt_pos, fdir_to_rad(crt_fdir), config.TRAIN.GRID_UNIT, None, vox_device, grid, oidx, occu_l)
			occu_l_lst = occu_l_lst.flatten(1).to(device)
			in_dict['occu_l'] = occu_l_lst
		t_voxel += time.perf_counter() - t_vox_start

		if USE_TGT and 'close' in locals():
			in_dict['close'] = close
		in_vec = get_in_vec(in_dict)
		t_input += time.perf_counter() - t_prep

		# --- Network forward ---
		t_model_start = time.perf_counter()
		if USE_NORM:
			ndims = input_mean.shape[0]
			in_vec[:, :ndims] = normalize(in_vec[:, :ndims], input_mean, input_std)
		if config.TRAIN.SEP_CTRLS:
			ctrl_dict = vec_to_ctrl(config, in_vec)
			if i == 0:
				logger.info(f'[Net] first frame ctrl keys: {list(ctrl_dict.keys())} | '
				            f'in_vec: {list(in_vec.shape)} | pred_out: {config.MODEL.OUT_DIM}')
			if USE_FIELD:
				vel = past_vel.view(-1, 3)
				pred_dict = network(ctrl_dict, d_vecs, vel_norm, vel, return_vel=True)
				pred_vec, ori_v, dv = pred_dict['pred'], pred_dict['vel'], pred_dict['dv']
				ori_v = qrot(fdir_to_quat(crt_fdir), ori_v)
				dv = qrot(fdir_to_quat(crt_fdir), dv)
				ori_v[..., 2] = 0.0
				dv[..., 2] = 0.0
				ori_v_lst.append(ori_v.cpu())
				dv_lst.append(dv.cpu())
			else:
				pred_vec = network(ctrl_dict)
		else:
			pred_vec = network(in_vec)
		t_model += time.perf_counter() - t_model_start

		# --- Denormalize & parse outputs ---
		if USE_NORM:
			pred_vec = denormalize(pred_vec, output_mean, output_std)
		nxt_j6d, nxt_jego, nxt_jevel, nxt_traj, vel, fdir_vel = get_from_outputs(pred_vec, future_kf)
		if USE_LIMB_TRAJ:
			nxt_limbs, nxt_limb_vel = get_limb_from_outputs(pred_vec, future_kf)

		# --- State update (pos, fdir, SMPL, past buffers, roll-forward) ---
		t_upd_start = time.perf_counter()
		nxt_pos = update_pos(crt_pos, crt_fdir, nxt_traj)
		nxt_fdir = update_fdir(crt_fdir, fdir_vel)

		t_smpl_start = time.perf_counter()
		abs_pose = get_pose(cont6d_to_aa(nxt_j6d).flatten(1), nxt_pos, fdir_to_rad(nxt_fdir), bm)
		if USE_SMPL_UPDATE:
			nxt_jego = get_pose(cont6d_to_aa(nxt_j6d).flatten(1), bm=bm)
		t_smpl += time.perf_counter() - t_smpl_start

		pose_seq.append(abs_pose.clone().cpu())
		aa_pose_seq.append(build_full_aa_pose(nxt_j6d, nxt_fdir, device))
		trans_seq.append(nxt_pos.clone())
		if USE_TGT:
			rel_pos_seq.append(get_abs_limbs(crt_pos, fdir_to_rad(crt_fdir), tgt_limbs).cpu())
		past_traj, past_vel = get_ss_past_traj(past_traj, past_vel, crt_pos[:, 2:3], vel, nxt_traj, fdir_vel)
		if USE_LIMB_TRAJ:
			past_limb_traj, past_limb_vel = \
				get_ss_past_limb_traj(past_limb_traj, nxt_traj, nxt_limbs, past_limb_vel, nxt_limb_vel, fdir_vel)
		past_fdir, past_fdir_vel = get_ss_past_vel(past_fdir, past_fdir_vel, fdir_vel)
		crt_j6d = nxt_j6d
		crt_jego = nxt_jego
		crt_jevel = nxt_jevel
		crt_pos = nxt_pos
		crt_fdir = nxt_fdir
		t_update += time.perf_counter() - t_upd_start

		t_total += time.perf_counter() - t0

	# ---- Timing summary (★ = critical for real-time) ----
	loop_wall = time.perf_counter() - loop_t0
	n_frames = infer_len
	segs = [
		('input_prep',     t_input,  ''),
		('voxel_query',    t_voxel,  ''),
		('model_forward',  t_model,  '★'),
		('smpl_forward',   t_smpl,   ''),
		('state_update',   t_update, ''),
	]
	fps_model = n_frames / max(t_model, 1e-6)
	fps_wall  = n_frames / max(loop_wall, 1e-6)
	logger.info(f'>>> TIMING bs={bs} frames={n_frames} wall={loop_wall:.2f}s '
	            f'| avg={loop_wall/n_frames*1000:.1f}ms/frame | FPS: model={fps_model:.0f} wall={fps_wall:.0f}')
	for name, t, marker in segs:
		logger.info(f'    {marker} {name:<16s} {t:7.3f}s ({t/n_frames*1000:6.2f}ms/f)  {t/loop_wall*100:5.1f}%')
	overhead = t_total - (t_input + t_voxel + t_model + t_smpl + t_update)
	logger.info(f'    loop_overhead     {overhead:7.3f}s  (Python/GPU sync)')

	# ---- Post-process & assemble draw_dict ----
	pose_seq = torch.stack(pose_seq, dim=1) # [bs, frames, 22, 3]
	starts = pose_seq[:, 0]
	rel_pos_seq = torch.stack(rel_pos_seq, dim=1) # [bs, frames, 5, 3]
	if config.INFER.get('ANI_EARLY_STOP', False):
		DIST_THRES = config.INFER.get('DIST_THRES', 4.0)
		INC_LASTS = config.INFER.get('INC_LASTS', 2)
		dists, closest_ids = calculate_distances(pose_seq[:, :, ELIMBS], tgt_limb_abs.cpu(), INC_LASTS, DIST_THRES)
		poses = []
		rel_limbs = []
		for i in range(len(pose_seq)):
			poses.append(pose_seq[i, :closest_ids[i]+1])
			rel_limbs.append(rel_pos_seq[i, :closest_ids[i]+1])
		pose_seq = poses
		rel_limbs = rel_pos_seq
	aa_pose_seq = torch.stack(aa_pose_seq, dim=1)  # [bs, frames, 72]
	trans_seq = torch.stack(trans_seq, dim=1)       # [bs, frames, 3]
	if config.INFER.get('ANI_EARLY_STOP', False):
		aa_poses_trunc = []
		trans_trunc = []
		for i in range(len(aa_pose_seq)):
			aa_poses_trunc.append(aa_pose_seq[i, :closest_ids[i]+1])
			trans_trunc.append(trans_seq[i, :closest_ids[i]+1])
		aa_pose_seq = aa_poses_trunc
		trans_seq = trans_trunc

	draw_dict = {'seq': pose_seq, 'start': starts}
	draw_dict['mid_snip'] = data['mid_snip']
	draw_dict['aa_poses'] = aa_pose_seq
	draw_dict['trans'] = trans_seq
	if USE_TGT:
		draw_dict['tgt_seq'] = rel_pos_seq
		draw_dict['end'] = jabs_tgt
	if USE_VOX:
		draw_dict['occug'] = [dataset.occu_g_dict[i][0] for i in data['mid_snip'][:, 0].tolist()]
		draw_dict['llb'] = llb
		draw_dict['unit'] = config.TRAIN.GRID_UNIT
	if USE_FIELD:
		ori_v_lst.append(torch.zeros_like(ori_v_lst[0]))
		dv_lst.append(torch.zeros_like(dv_lst[0]))
		ori_v_lst = torch.stack(ori_v_lst, dim=1)
		dv_lst = torch.stack(dv_lst, dim=1)
		draw_dict['ori_v'] = ori_v_lst
		draw_dict['dv'] = dv_lst
	return draw_dict

def save_npz(draw_dict, save_dir, name=None):
	"""Save generated motions as AMASS-format NPZ files (24-joint SMPL axis-angle)."""
	os.makedirs(save_dir, exist_ok=True)
	n = len(draw_dict['aa_poses'])
	t0 = time.perf_counter()
	for i in range(n):
		mid, st_fid, end_fid, idx = draw_dict['mid_snip'][i]
		poses = draw_dict['aa_poses'][i].cpu().numpy().astype(np.float64)  # [frames, 72]
		trans = draw_dict['trans'][i].cpu().numpy().astype(np.float64)     # [frames, 3]
		betas = np.zeros(16, dtype=np.float64)
		gender = 'male'
		fps = 10.0
		if name is not None:
			save_name = f'{name}.npz' if n == 1 else f'{name}_{i}.npz'
		else:
			save_name = f'gen_no{idx}_{mid}_{st_fid}_{end_fid}.npz'
		extra = {}
		if '_j6d' in draw_dict:
			extra['_j6d'] = draw_dict['_j6d'][i].cpu().numpy().astype(np.float64)
			extra['_jego'] = draw_dict['_jego'][i].cpu().numpy().astype(np.float64)
			extra['_jabs'] = draw_dict['seq'][i].cpu().numpy().astype(np.float64)  # global joints
			extra['_fdir'] = draw_dict['_fdir'][i].cpu().numpy().astype(np.float64)
		np.savez(os.path.join(save_dir, save_name),
		         poses=poses, trans=trans, betas=betas,
		         gender=gender, mocap_framerate=fps, **extra)
	elapsed = time.perf_counter() - t0
	logger.info(f'>>> NPZ save: {n} files in {elapsed:.2f}s ({elapsed/n:.2f}s/file) → {save_dir}')

def draw_batch(draw_dict, save_dir, name=None, save_video=True):
	os.makedirs(save_dir, exist_ok=True)
	n = len(draw_dict['seq'])
	if save_video:
		t0 = time.perf_counter()
		for i in range(n):
			single_draw_dict = {
				'seq': draw_dict['seq'][i],
				'start': draw_dict['start'][i],
				'end': draw_dict['end'][i],
				'tgt_seq': draw_dict['tgt_seq'][i],
			}
			if "occug" in draw_dict:
				single_draw_dict['occug'] = draw_dict['occug'][i]
				single_draw_dict['llb'] = draw_dict['llb'][i]
				single_draw_dict['unit'] = draw_dict['unit']
			if 'ori_v' in draw_dict:
				single_draw_dict['ori_v'] = draw_dict['ori_v'][i]
				single_draw_dict['dv'] = draw_dict['dv'][i]
			mid, st_fid, end_fid, idx = draw_dict['mid_snip'][i]
			sinp_len = len(single_draw_dict['seq'])
			if name is not None:
				save_name = f'{name}.mp4' if n == 1 else f'{name}_{i}.mp4'
			else:
				save_name = f'vox_len{sinp_len}_no{idx}_{mid}_{st_fid}_{end_fid}.mp4'
			draw_seq(os.path.join(save_dir, save_name), single_draw_dict)
		elapsed = time.perf_counter() - t0
		logger.info(f'>>> MP4 render: {n} videos in {elapsed:.2f}s ({elapsed/n:.2f}s/video) → {save_dir}')
	else:
		logger.info(f'>>> MP4 render: skipped (save_video=False)')
	# Always save NPZ (needed for closed-loop planning).
	if 'aa_poses' in draw_dict and 'trans' in draw_dict:
		save_npz(draw_dict, save_dir, name=name)

if __name__ == '__main__':
    # Extract config path (handle -c/--config BEFORE our argparse, since
    # get_config_path() has its own parse_args that would conflict)
    config_path = None
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] in ('-c', '--config') and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            del sys.argv[i:i+2]  # remove -c value from argv
        else:
            i += 1
    if config_path is None:
        config_path = 'configs/config.yml'

    parser = argparse.ArgumentParser(description='Motion Occupancy Base inference')
    subparsers = parser.add_subparsers(dest='mode', help='Inference mode')

    # --- manual subcommand ---
    manual_parser = subparsers.add_parser('manual', help='Manual start/end specification (no dataset)')
    manual_parser.add_argument('--config', type=str, default=None, help='Path to YAML config')
    manual_parser.add_argument('--init_path', type=str, default=None, help='Path to start SMPL .npz')
    manual_parser.add_argument('--init_pos', type=str, default=None, help='Start root position x,y,z (e.g. 0.0,0.0,0.9)')
    manual_parser.add_argument('--init_fdir', type=str, default=None, help='Start facing direction x,y,z (e.g. 0.0,1.0,0.0)')
    manual_parser.add_argument('--tgt_path', type=str, default=None, help='Path to target SMPL .npz')
    manual_parser.add_argument('--tgt_root', type=str, default=None, help='Target root position x,y,z')
    manual_parser.add_argument('--tgt_fdir', type=str, default=None, help='Target facing direction x,y,z')
    manual_parser.add_argument('--occu_path', type=str, default=None, help='Path to occupancy .pkl')
    manual_parser.add_argument('--name', type=str, default='output', help='Output file name prefix')
    manual_parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    manual_parser.add_argument('--infer_len', type=int, default=None, help='Number of frames to generate')
    manual_parser.add_argument('--no_video', action='store_true', help='Skip MP4 rendering (NPZ always saved)')
    manual_parser.add_argument('--device', type=int, default=None, help='CUDA device index')

    args = parser.parse_args()
    
    # Use CLI config override if provided
    if hasattr(args, 'config') and args.config:
        config_path = args.config
    config = OmegaConf.load(config_path)
    config.STAGE = 'INFER'
    
    if config.INFER.get('IGNORE_WARNINGS', False):
        import warnings
        warnings.filterwarnings("ignore")
    if type(config.TRAIN.GRID_SIZE) is int:
        config.TRAIN.GRID_SIZE = [config.TRAIN.GRID_SIZE] * 3
    
    device_idx = args.device if (hasattr(args, 'device') and args.device is not None) else config.DEVICE
    config.DEVICE_STR = f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu"
    device = torch.device(config.DEVICE_STR)
    logger = create_logger(config, to_file=False)
    
    # Set IO dims
    PAST_KF = config.TRAIN.PAST_KF
    FUTURE_KF = config.TRAIN.FUTURE_KF
    IN_DIM, OUT_DIM = set_io_dims(config)

    logger.info("Initializing network...")
    if config.TRAIN.USE_NORM:
        mean_std = torch.load(os.path.join(config.ASSETS.SPLIT_DIR, config.ASSETS.MEAN_STD_NAME))
        input_mean, input_std, output_mean, output_std = [v.to(device) for k, v in list(mean_std.items())]
        if config.TRAIN.get('USE_FCONTACT', False):
            input_mean, input_std = pad_fcontact_norm(input_mean, input_std)
        input_mean, input_std, output_mean, output_std = trunc_norm(config, input_mean, input_std, output_mean, output_std)

    network = get_model(config)
    load_res = network.load_state_dict(torch.load(config.ASSETS.CHECKPOINT, map_location=torch.device('cpu')), strict=False)
    logger.info(f'>>> Checkpoint loaded | missing={len(load_res.missing_keys)} unexpected={len(load_res.unexpected_keys)}')
    network.to(device)
    network.eval()
    torch.set_grad_enabled(False)

    bm = smplx.create(config.ASSETS.SMPL_DIR, model_type='smpl', gender='male', num_betas=16).to(device)
    norm = (input_mean, input_std, output_mean, output_std)

    if args.mode == 'manual':
        # ---- Manual mode ----
        manual_cfg = config.INFER.get('MANUAL', {})

        # Helper: parse comma-separated float string
        def parse_vec(s, default=None):
            if s is None:
                return default
            return torch.tensor([float(x.strip()) for x in s.split(',')])

        # Resolve init_path
        init_path = args.init_path or manual_cfg.get('INIT_PATH', None)
        if init_path is None:
            logger.error('--init_path (or INFER.MANUAL.INIT_PATH) is required for manual mode')
            sys.exit(1)

        # Resolve tgt_path
        tgt_path = args.tgt_path or manual_cfg.get('TGT_PATH', None)
        if tgt_path is None:
            logger.error('--tgt_path (or INFER.MANUAL.TGT_PATH) is required for manual mode')
            sys.exit(1)

        # Resolve occupancy
        occu_path = args.occu_path or manual_cfg.get('OCCU_PATH', None)
        if occu_path is None:
            logger.error('--occu_path (or INFER.MANUAL.OCCU_PATH) is required for manual mode')
            sys.exit(1)

        # Resolve init_pos
        init_pos = parse_vec(args.init_pos)
        if init_pos is None:
            default_pos = manual_cfg.get('INIT_POS', [0.0, 0.0, 0.9])
            init_pos = torch.tensor(default_pos)

        # Resolve init_fdir
        init_fdir = parse_vec(args.init_fdir)
        if init_fdir is None:
            default_fdir = manual_cfg.get('INIT_FDIR', [0.0, 1.0, 0.0])
            init_fdir = torch.tensor(default_fdir)

        # Resolve tgt_root
        tgt_root = parse_vec(args.tgt_root)
        if tgt_root is None:
            default_tgt_root = manual_cfg.get('TGT_ROOT', None)
            if default_tgt_root is not None:
                tgt_root = torch.tensor(default_tgt_root)
            else:
                # Use init_pos as default for target root
                tgt_root = init_pos.clone()

        # Resolve tgt_fdir
        tgt_fdir = parse_vec(args.tgt_fdir)
        if tgt_fdir is None:
            default_tgt_fdir = manual_cfg.get('TGT_FDIR', None)
            if default_tgt_fdir is not None:
                tgt_fdir = torch.tensor(default_tgt_fdir)
            else:
                tgt_fdir = init_fdir.clone()

        # Resolve output
        output_name = args.name or 'output'
        output_dir = args.output_dir or config.INFER.ANI_SAVE_DIR
        infer_len = args.infer_len or manual_cfg.get('INFER_LEN', config.INFER.INFER_LEN)

        # Load occupancy
        logger.info(f'Loading occupancy from {occu_path}')
        occu_g_np, unit, llb = pickle.load(open(occu_path, 'rb'))
        occu_g = torch.from_numpy(occu_g_np).float()
        llb = torch.from_numpy(llb).float()

        # Load start pose
        logger.info(f'Loading start pose from {init_path}')
        init_j6d, init_jego, init_jabs, _ = load_smpl_npz(init_path, init_pos, init_fdir, bm, device)

        # Load target pose
        logger.info(f'Loading target pose from {tgt_path}')
        tgt_limb_abs, tgt_jabs_full = get_tgt_limbs_from_npz(tgt_path, tgt_root, tgt_fdir, bm, device)

        logger.info(f'Start: pos={init_pos.tolist()}, fdir={init_fdir.tolist()}')
        logger.info(f'Target: root={tgt_root.tolist()}, fdir={tgt_fdir.tolist()}, limbs={tgt_limb_abs.shape}')
        logger.info(f'Output: {output_dir}/{output_name}.[npz|mp4]')

        draw_dict = infer_manual(bm, network, config, norm,
                                 init_pos, init_fdir, init_j6d, init_jego,
                                 tgt_limb_abs, occu_g, llb, unit,
                                 infer_len, device)
        # Override end with full 22-joint target for visualization
        draw_dict['end'] = tgt_jabs_full.cpu()
        save_video = not args.no_video
        draw_batch(draw_dict, output_dir, name=output_name, save_video=save_video)
        logger.info(f'>>> Manual inference done → {output_dir}/{output_name}.npz')

    else:
        # ---- Original dataset mode ----
        logger.info(f'>>> SAVE_DIR: {config.INFER.ANI_SAVE_DIR}')

        dataset = MphaseDataset(config, config.INFER.SPLIT)
        if config.INFER.get('SPECIFY_CASES', False):
            case_config = OmegaConf.load(config.INFER.CASE_CONFIG)
            split_dirname = os.path.basename(config.ASSETS.SPLIT_DIR)
            data_ids = eval(f'case_config.CASES.{split_dirname}')
        else:
            interval = config.INFER.get('SAVE_ID_INTERVAL', 500)
            data_ids = [i for i in range(0, len(dataset), interval)]
        data = collate_fn([dataset[i] for i in data_ids])
        logger.info(f'>>> Loaded {len(data_ids)} snippets (interval={config.INFER.get("SAVE_ID_INTERVAL", 500)})')

        ANI_SAVE_DIR = config.INFER.ANI_SAVE_DIR
        draw_dict = infer(data, dataset, bm, network, config, norm)
        draw_batch(draw_dict, ANI_SAVE_DIR)
        logger.info(f'>>> Inference done → {ANI_SAVE_DIR}')
