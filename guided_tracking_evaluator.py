""" Abdominal Tracking evaluation script.

Takes a volume with all of their centerline points.
Starts tracking from the center of each trace, tracking until the
distance to the trace > threshold.

WARNING: EXISTING CODE CONFUSES VOXSPACE AND MMSPACE.
Thread carefully around unit conversions.
"""

import os
import pathlib
from datetime import datetime
from shutil import copyfile
import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
import nibabel as nib
import torch, numpy as np
from torch.nn.functional import pad
from monai.inferers import sliding_window_inference  # handy utility
import pandas as pd          # optional but handy
import seaborn as sns        # nicer defaults than plain matplotlib
sns.set(style="whitegrid")   # pick any seaborn style you like

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from data.rt_colon_dataset import vox_coords_to_mm, mm_coords_to_vox

from typing import List, Sequence
from scipy.ndimage import binary_dilation

#########################
# Saving Functions
#########################

def save_small_bowel_segmentation_nifti(inf_volume, segmentation, save_path, threshold=0.5):
    """
    Save the full small bowel segmentation output as a NIfTI file.
    """
    seg_mask = (segmentation > threshold).astype(np.uint8)
    affine = np.diag(np.append(inf_volume.spacing, 1))
    nifti_img = nib.Nifti1Image(seg_mask, affine)
    nib.save(nifti_img, save_path)
    print(f"Saved full small bowel segmentation NIfTI to: {save_path}")

def save_original_volume_nifti(inf_volume, save_path):
    """
    Save the original MRI volume as a NIfTI file.
    """
    mri_vol = inf_volume.data.cpu().numpy()[0]
    affine = np.diag(np.append(inf_volume.spacing, 1))
    nifti_img = nib.Nifti1Image(mri_vol, affine)
    nib.save(nifti_img, save_path)
    print(f"Saved original MRI volume NIfTI to: {save_path}")

#########################
# Metrics and Dice
#########################

def mean_surface_distance(predicted_mm: np.ndarray,
                          annotated_vox: np.ndarray,
                          spacing: Sequence[float]) -> float:
    """
    Mean Surface Distance (MSD) between two centre-lines or surface point clouds.

    Parameters
    ----------
    predicted_mm   : (N,3) ndarray – predicted points in **mm space**
    annotated_vox  : (M,3) ndarray – annotated points in **voxel space**
    spacing        : (sx,sy,sz)    – voxel size in mm

    Returns
    -------
    msd_mm         : float         – mean bidirectional distance in **mm**
    """
    annotated_mm = vox_coords_to_mm(annotated_vox, spacing)

    # pair-wise Euclidean distances (N×M)
    D = cdist(predicted_mm, annotated_mm)

    # shortest distance from every point A→B and B→A
    d_pred_to_anno = D.min(axis=1)
    d_anno_to_pred = D.min(axis=0)

    msd_mm = (d_pred_to_anno.sum() + d_anno_to_pred.sum()) / (len(d_pred_to_anno) + len(d_anno_to_pred))
    return float(msd_mm)

def compute_centerline_metrics(predicted_mm, annotated_vox, spacing, threshold=10.0):
    """
    Compute metrics (precision, recall, Dice/F1) between predicted centerline (in mm)
    and annotated centerline (in voxel, converted to mm).
    """
    annotated_mm = vox_coords_to_mm(annotated_vox, spacing)
    predicted_points = np.array(predicted_mm)
    annotated_points = np.array(annotated_mm)
    distances = cdist(predicted_points, annotated_points)
    min_dist_pred = distances.min(axis=1)
    TP_pred = np.sum(min_dist_pred <= threshold)
    precision = TP_pred / len(predicted_points) if len(predicted_points) > 0 else 0
    min_dist_annot = distances.min(axis=0)
    TP_annot = np.sum(min_dist_annot <= threshold)
    recall = TP_annot / len(annotated_points) if len(annotated_points) > 0 else 0
    if precision + recall > 0:
        F1 = 2 * precision * recall / (precision + recall)
    else:
        F1 = 0
    return precision, recall, F1

def dice_coefficient(pred_mask, gt_mask, eps=1e-6):
    intersection = (pred_mask & gt_mask).sum()
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    return (2.0*intersection + eps) / (pred_sum + gt_sum + eps)

#########################
# Helper Functions for Overlays & Masks
#########################

def plot_centerline_overlay(inf_volume, opt, centerlines_mm, styles, save_path, title=None):
    """
    Create and save a PNG overlay image given a list of centerlines (in mm) and style options.
    """
    # Choose a reference centerline for slice selection – use the second one (annotated) if available.
    ref_cl = centerlines_mm[1] if len(centerlines_mm) >= 2 else centerlines_mm[0]
    ref_vox = np.round(mm_coords_to_vox(ref_cl, inf_volume.spacing)).astype(int)
    ref_vox[:,0] -= 1
    ref_vox[:,1] -= 1
    slice_idx = int(np.median(ref_vox[:, 2]))
    mri_vol = inf_volume.data.cpu().numpy()[0]
    mri_slice = mri_vol[:, :, slice_idx]
    plt.figure(figsize=(10, 10))
    plt.imshow(mri_slice.T, cmap='gray', origin='lower')
    for cl_mm, style in zip(centerlines_mm, styles):
        cl_vox = np.round(mm_coords_to_vox(cl_mm, inf_volume.spacing)).astype(int)
        cl_vox[:,0] -= 1
        cl_vox[:,1] -= 1
        plt.plot(cl_vox[:,0], cl_vox[:,1],
                 color=style.get("color", "red"),
                 linestyle=style.get("linestyle", "-"),
                 linewidth=2,
                 label=style.get("label", "Centerline"))
        plt.scatter(cl_vox[:,0], cl_vox[:,1],
                    color=style.get("color", "red"),
                    s=10)
    # Plot seed point from the generated centerline (assumed first element)
    gen_vox = np.round(mm_coords_to_vox(centerlines_mm[0], inf_volume.spacing)).astype(int)
    gen_vox[:,0] -= 1
    gen_vox[:,1] -= 1
    seed_vox = gen_vox[0]
    plt.scatter(seed_vox[0], seed_vox[1], color="cyan", s=50, marker="o", label="Seed Point")
    if title is None:
        title = f"Centerline Overlay on Axial Slice {slice_idx} (Volume {opt.volume_number})"
    plt.title(title)
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved overlay image to: {save_path}")

def create_segmentation_mask(inf_volume, centerlines_mm, labels, dilate=False, dilation_iterations=1):
    """
    Create a segmentation mask from given centerlines (list in mm) and associated labels.
    Optionally perform binary dilation.
    """
    vol_shape = inf_volume.data.cpu().numpy()[0].shape
    mask = np.zeros(vol_shape, dtype=np.uint8)
    for cl_mm, label in zip(centerlines_mm, labels):
        vox = np.round(mm_coords_to_vox(cl_mm, inf_volume.spacing)).astype(int)
        vox[:,0] -= 1
        vox[:,1] -= 1
        for (x, y, z) in vox:
            if 0 <= x < vol_shape[0] and 0 <= y < vol_shape[1] and 0 <= z < vol_shape[2]:
                mask[x, y, z] = label
    if dilate:
        from scipy.ndimage import binary_dilation
        structure = np.ones((3,3,3), dtype=np.uint8)
        new_mask = np.zeros_like(mask)
        for lbl in np.unique(mask):
            if lbl == 0:
                continue
            binary = mask == lbl
            dilated = binary_dilation(binary, structure=structure, iterations=dilation_iterations)
            new_mask[dilated] = lbl
        mask = new_mask
    return mask

def save_nifti_mask(inf_volume, mask, save_path):
    """
    Save a 3D segmentation mask (numpy array) as a NIfTI file.
    """
    affine = np.diag(np.append(inf_volume.spacing, 1))
    nifti_img = nib.Nifti1Image(mask, affine)
    nib.save(nifti_img, save_path)
    print(f"Saved nifti mask to: {save_path}")

#########################
# Helper Functions
#########################

def get_gt_backward_dir(seed_mm: np.ndarray,
                        gt_trace_mm: np.ndarray) -> np.ndarray:
    """
    Return the unit vector from the seed point to the previous GT point.
    If the seed is at index 0 we fall back to the next point (opposite sign).
    """
    # index of seed in the dense GT trace
    idx = np.argmin(np.linalg.norm(gt_trace_mm - seed_mm, axis=1))
    if idx > 0:
        vec = gt_trace_mm[idx-1] - seed_mm          # backward in GT order
    else:                                           # seed at first point
        vec = seed_mm - gt_trace_mm[idx+1]          # use the forward neighbour
    vec /= (np.linalg.norm(vec) + 1e-8)
    return vec.astype(np.float32)

def get_spheredistr_maxprob(probs, vecs, avoid=None, mindist=0, ret_prob=False):
    probs_flat = probs.flatten()
    vecs_flat = np.asarray(vecs).reshape((vecs.shape[0]*vecs.shape[1], 3))
    if mindist > 0 and avoid is not None:
        for i in range(vecs_flat.shape[0]):
            vec = vecs_flat[i, :]
            norm = np.linalg.norm(vec - avoid)
            if norm < mindist:
                probs_flat[i] = 0
    max_prob_index = np.argmax(probs_flat)
    max_prob_value = np.amax(probs_flat)
    max_prob_vec = vecs_flat[max_prob_index, :]
    if ret_prob:
        return max_prob_vec, max_prob_value
    else:
        return max_prob_vec

def show_step_plots(first_direction, inf_volume, model, opt, probs, ss, step, trace, vecs):
    mpatch, _, _ = inf_volume.get_patch(trace[-2], False)
    if len(mpatch.shape) > 3:
        mpatch = mpatch[0, :, :, :]
    tricanvas = model.triplanar_canvas(mpatch)
    rel_coord = (trace[-1] - trace[-2]) / ss
    tricanvas = model.result_painter(tricanvas, [rel_coord], 2)
    pred_rel_coords = []
    probs_flat = probs.flatten()
    vecs_flat = np.asarray(vecs).reshape((probs.shape[0]*probs.shape[1], 3))
    sorted_probs = np.sort(probs_flat)
    for i in range(len(probs_flat)):
        if probs_flat[i] > sorted_probs[-10]:
            pred_rel_coords.append(vecs_flat[i])
    tricanvas = model.result_painter(tricanvas, pred_rel_coords, 6)
    fname = f'/tmp/traceframe_{opt.volume_number}_{abs(first_direction[2]):.3f}_{step:04d}.png'
    plt.imsave(fname, np.clip(tricanvas/255, 0, 1))
    if step % 10 == 0:
        plt.imshow(tricanvas/255)
        plt.show()

def save_patch_overlay_png(patch_np: np.ndarray,
                           mask_np:  np.ndarray,
                           save_path: str,
                           mask_color: str = "red",
                           alpha: float = 0.40,
                           slice_idx: int | None = None):
    """
    patch_np : (P,P,P)   normalised MRI patch   [0-1] float
    mask_np  : (P,P,P)   binary mask            {0,1}
    save_path: *.png
    """
    # choose axial middle slice unless caller specifies one
    z = patch_np.shape[2]//2 if slice_idx is None else slice_idx

    plt.figure(figsize=(4,4))
    plt.imshow(patch_np[:, :, z].T, cmap="gray", origin="lower")
    plt.imshow(mask_np[:, :, z].T,
               cmap=plt.cm.get_cmap(mask_color),
               alpha=alpha, origin="lower")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def save_low_dice_triplet_nifti(patch_img: torch.Tensor,
                                gt_mask: torch.Tensor,
                                pred_mask: torch.Tensor,
                                spacing: Sequence[float],
                                save_path: str):
    """
    Write a 3-channel volume:
        channel-0 : normalised MRI patch (float32)
        channel-1 : GT mask   (uint8)
        channel-2 : Pred mask (uint8)
    so that you can inspect them in ITK-SNAP or 3D Slicer.
    """
    patch_np = patch_img.cpu().numpy().astype(np.float32)
    patch_np = (patch_np - patch_np.min()) / (patch_np.ptp() + 1e-6)  # 0-1

    gt_np   = gt_mask.cpu().numpy().astype(np.uint8)
    pred_np = pred_mask.cpu().numpy().astype(np.uint8)

    triplet = np.stack([patch_np, gt_np, pred_np], axis=0)           # [3,p,p,p]
    affine  = np.diag(np.append(spacing, 1))

    nib.save(nib.Nifti1Image(triplet, affine), save_path)

    png_prefix = save_path.replace(".nii.gz", "")        # reuse same stem
    save_patch_overlay_png(patch_np, gt_np,   png_prefix + "_GT.png",
                           mask_color="Blues")           # annotated mask
    save_patch_overlay_png(patch_np, pred_np, png_prefix + "_PR.png",
                           mask_color="Reds")            # prediction

def get_probs_for_sample(inf_volume, model, key, num):
    im_patch, target, _ = inf_volume.get_sample(key, num, valvol=True)
    patches = [im_patch]
    targets = [target]
    initial_input = {'A': torch.stack(patches, dim=0).cuda()}
    model.set_input(initial_input)
    model.real_B = torch.stack(targets, dim=0).cuda()
    with torch.no_grad():
        model.forward()
    initial_result = model.fake_B
    if inf_volume.opt.independent_dir:
        # result:  [B, S, 500]   with S = 2 × n_shells
        initial_result = initial_result[:, 0::2, :]        # channels 0,2,4,… (forward)
    probs_cuda = merge_multiscale_shells(initial_result, metric=opt.scale_merge_metric)
    probabilities = probs_cuda.cpu().numpy()
    return probabilities

def get_probs_for_aux_traces(inf_volume, model, aux_traces, low_dice_dir: str = None, dice_threshold: float = 0.30):
    rotate = False
    opt = inf_volume.opt
    patches, seggt_patches, rotmats, inv_mats, vecs = [], [], [], [], []
    dummypatch = torch.ones((opt.input_nc, opt.patch_size, opt.patch_size, opt.patch_size))
    for i in range(len(aux_traces)):
        if aux_traces[i].alive:
            cur_point = aux_traces[i].current_point
            valpatch, seggt_patch, rotmat, inv_mat = inf_volume.get_patch(cur_point, rotate, with_seg=True, convert_to_mm=False)
            if opt.independent_dir:
                # add previous direction channels to the patch
                valpatch = add_prev_dir_channels(valpatch, aux_traces[i].last_direction, inv_mat)
        else:
            valpatch = dummypatch
            seggt_patch = dummypatch
            rotmat = np.eye(4)
            inv_mat = np.eye(4)
        patches.append(valpatch)
        seggt_patches.append(seggt_patch)
        rotmats.append(rotmat)
        inv_mats.append(inv_mat)
    initial_input = {'A': torch.stack(patches, dim=0).cuda()}
    model.set_input(initial_input)
    with torch.no_grad():
        model.forward()
    initial_result = model.fake_B
    if opt.independent_dir:
        # result:  [B, S, 500]   with S = 2 × n_shells
        initial_result = initial_result[:, ::2, :]        # channels 0,2,4,… (forward)
    seg_probs = model.fake_C          # [B, C, p, p, p]
    pred_labels = seg_probs.argmax(dim=1)
    probs_cuda = merge_multiscale_shells(initial_result, metric=opt.scale_merge_metric)
    probabilities = probs_cuda.cpu().numpy()
    verts = inf_volume.vertices
    for i in range(len(rotmats)):
        vec = []
        for j in range(verts.shape[0]):
            sphereloc = np.append(verts[j], 0)
            rotpoint = np.dot(rotmats[i], sphereloc)
            vec.append(rotpoint[:3])
        vecs.append(vec)

    dice_list = []
    first = True
    for k in range(len(seggt_patches)):
        if aux_traces[k].alive:
            pred_mask = (pred_labels[k] == 1)
            gt_mask = (seggt_patches[k][0] == 1)       # remove channel dim
            dice = dice_coefficient(pred_mask.cpu().numpy(), gt_mask.cpu().numpy())
            if low_dice_dir and dice < dice_threshold and first:
                os.makedirs(low_dice_dir, exist_ok=True)
                save_low_dice_triplet_nifti(
                    patches[k][0],          # image (channel-0 of patch)
                    gt_mask, pred_mask,
                    inf_volume.spacing,
                    f"{low_dice_dir}/"
                    f"trace{aux_traces[k].__hash__()}_"   # a quick unique id
                    f"step{k}_dice{dice:.2f}.nii.gz"
                )
                first = False
            dice_list.append(dice)
    final_dice = np.asarray(dice_list).mean()
    return probabilities, vecs, final_dice

def get_probs_at_location(inf_volume, model, start_point, point_in_voxspace=False, prev_dir_world=None):
    opt = inf_volume.opt
    patches, rotmats, inv_mats, vecs = [], [], [], []
    for i in range(opt.test_time_augments):
        rotate = (i != 0)
        valpatch, rotmat, inv_mat = inf_volume.get_patch(
                start_point, rotate, convert_to_mm=point_in_voxspace)

        if opt.independent_dir:
            # now pass the caller-supplied vector, may be None
            valpatch = add_prev_dir_channels(valpatch, prev_dir_world, inv_mat)
        patches.append(valpatch)
        rotmats.append(rotmat)
        inv_mats.append(inv_mat)
    initial_input = {'A': torch.stack(patches, dim=0).cuda()}
    model.set_input(initial_input)
    with torch.no_grad():
        model.forward()
    initial_result = model.fake_B
    if opt.independent_dir:
        # result:  [B, S, 500]   with S = 2 × n_shells
        initial_result = initial_result[:, ::2, :]        # channels 0,2,4,… (forward)
    probs_cuda = merge_multiscale_shells(initial_result, metric=opt.scale_merge_metric)
    probabilities = probs_cuda.cpu().numpy()
    verts = inf_volume.vertices
    for i in range(len(rotmats)):
        vec = []
        for j in range(verts.shape[0]):
            sphereloc = np.append(verts[j], 0)
            rotpoint = np.dot(rotmats[i], sphereloc)
            vec.append(rotpoint[:3])
        vecs.append(vec)
    return probabilities, vecs

def merge_multiscale_shells(result, metric='mean'):
    if len(metric) == 1:
        squashed_result = result[:, int(metric), :]
    elif metric == 'mean':
        squashed_result = result.sum(1) / result.shape[1]
    elif metric == 'max':
        squashed_result = result.max(1)[0]
    else:
        print(f'metric {metric} not implemented!!')
        1/0
    return squashed_result

class AuxTrace:
    def __init__(self, start_mm, first_direction, ss):
        self.alive = True
        self.dying = False
        self.ss = ss
        self.current_point = start_mm + ss * first_direction
        self.trace = [start_mm, self.current_point]
        self.confidence_trace = [0.25, 0.25]
        self.last_direction = first_direction

    def __add__(self, other):
        return other + self.alive * 1

    def __radd__(self, other):
        return other + self.alive * 1

    def apply_stochastic_direction_from_spheredistr(self, probs, vecs, mindist):
        probs_flat = probs.flatten()
        vecs = np.asarray(vecs)
        if len(vecs.shape) > 2:
            vecs = vecs.reshape((vecs.shape[0]*vecs.shape[1], 3))
        avoid = -self.last_direction
        for i in range(vecs.shape[0]):
            vec = vecs[i, :]
            norm = np.linalg.norm(vec - avoid)
            if norm < mindist:
                probs_flat[i] = 0
        chosen_ind = np.random.choice(np.arange(0, 500), p=probs_flat / (np.sum(probs_flat)))
        chosen_vec = vecs[chosen_ind]
        self.last_direction = chosen_vec
        max_prob_value = np.amax(probs_flat)
        self.current_point = self.current_point + self.ss * chosen_vec
        self.trace.append(self.current_point)
        self.confidence_trace.append(max_prob_value)
        return self.current_point, max_prob_value

def add_prev_dir_channels(patch, prev_dir_world, inv_mat):
    """
    Append 3 channels that contain the previous *forward* direction,
    rotated into the patch coordinate frame.

    patch          : torch.Tensor [C, p, p, p]
    prev_dir_world : np.ndarray   (3,)         (unit vector, world frame)
                     pass None or np.zeros(3) when no previous step exists
    inv_mat        : np.ndarray   4×4          world→patch matrix

    returns        : torch.Tensor [C+3, p, p, p]  (same dtype/device)
    """
    if prev_dir_world is None:
        prev_dir_world = np.zeros(3, dtype=np.float32)

    # rotate into local coords
    R = inv_mat[:3, :3]                          # 3×3
    prev_local = R @ prev_dir_world              # ↦ patch frame
    prev_local /= (np.linalg.norm(prev_local) + 1e-6)

    # broadcast to full patch
    p = patch.shape[-1]
    dir_map = torch.as_tensor(prev_local, dtype=patch.dtype,
                              device=patch.device).view(3,1,1,1)
    dir_map = dir_map.expand(3, p, p, p)         # [3,p,p,p]

    # concatenate
    return torch.cat([patch, dir_map], dim=0)

def stochastic_track_trace(model, inf_volume, start_point_mm, first_direction, gt_trace_mm, guidance_threshold_mm, low_dice_dir=None):
    opt = inf_volume.opt
    max_steps = opt.n_steps
    ss = opt.step_size
    confidence_thres = opt.confidence_thres
    conformist_threshold = opt.conformist_thres
    start_randomness_mm = opt.start_randomness_mm
    n_traces = opt.n_traces
    moving_conf_average = opt.moving_conf_average
    db_slack = opt.doubleback_slack_steps
    db_mindist = opt.doubleback_mindist
    if opt.disable_oov_slack:
        oov_bounds_z = [0, inf_volume.spacing[-1] * inf_volume.shape[-1]]
    else:
        oov_bounds_z = [-opt.isosample_spacing/2, inf_volume.spacing[-1] * inf_volume.shape[-1] + opt.isosample_spacing/2]
    current_point = start_point_mm + ss * first_direction
    median_trace = [start_point_mm, current_point]
    traces = []
    dicescores = []
    for i in range(n_traces):
        fuzz = [-1, 1, 1]
        fuzzcounter = 0
        while np.linalg.norm(fuzz) > 1:
            fuzzcounter += 1
            fuzz = np.random.uniform(-1, 1, 3)
            fuzzy_start = start_point_mm + fuzz * start_randomness_mm
            if opt.hard_stop_oov:
                if fuzzy_start[2] < oov_bounds_z[0] or fuzzy_start[2] > oov_bounds_z[1]:
                    fuzz = [-1, 1, 1]
            if fuzzcounter > 10000:
                print('bug in fuzzy start system, no valid start points found')
                1/0
        traces.append(AuxTrace(fuzzy_start, first_direction, ss))
    for step in tqdm(range(max_steps)):
        all_probs, all_vecs, dice_coef = get_probs_for_aux_traces(inf_volume, model, traces, low_dice_dir=low_dice_dir)
        dicescores.append(dice_coef)
        all_vecs = np.asarray(all_vecs)
        for i in range(n_traces):
            trace = traces[i]
            if not trace.alive:
                continue
            loc, confidence = trace.apply_stochastic_direction_from_spheredistr(all_probs[i], all_vecs[i], mindist=opt.min_maxprob_dist)
            running_mean = min(len(trace.confidence_trace), moving_conf_average)
            if np.mean(trace.confidence_trace[-running_mean:]) < confidence_thres:
                trace.dying = True
            if db_mindist > 0:
                if len(trace.trace) > db_slack:
                    cur_mindist = np.inf
                    for past_loc in trace.trace[:-db_slack]:
                        cur_mindist = min(cur_mindist, np.linalg.norm(trace.trace[-1] - past_loc))
                    if cur_mindist < db_mindist:
                        trace.dying = True
            if opt.hard_stop_oov:
                if trace.trace[-1][2] < oov_bounds_z[0] or trace.trace[-1][2] > oov_bounds_z[1]:
                    trace.dying = True
        living_traces = []
        living_locs = []
        for i in range(n_traces):
            trace = traces[i]
            if trace.alive:
                living_traces.append(trace)
                living_locs.append(trace.current_point)
        survivors = len(living_locs)
        median_loc = np.median(living_locs, axis=0)
        median_trace.append(median_loc)
        for trace in living_traces:
            dist_to_median = np.linalg.norm(median_loc - trace.current_point)
            if dist_to_median > conformist_threshold or trace.dying:
                trace.alive = False
                survivors -= 1
        if survivors < n_traces/4:
            print(f'stopping, too few survivors ({survivors}/{n_traces})')
            if survivors > 0 and opt.rebuild_median:
                print('rebuilding median trace from survivors...')
                median_trace = median_trace[0:2]
                for substep in range(step + 1):
                    surviving_locs = []
                    for i in range(n_traces):
                        trace = traces[i]
                        if trace.alive:
                            surviving_locs.append(trace.trace[i + 2])
                    median_trace.append(np.median(surviving_locs, axis=0))
            break
    all_traces = [t.trace for t in traces]
    all_trace_confidences = [t.confidence_trace for t in traces]
    print(f'trace done; len {len(median_trace)}')
    return median_trace, all_traces, all_trace_confidences, dicescores

def compute_stochastic_trace(model, inf_volume, key, gtdist_thres=False, low_dice_dir=None):
    if gtdist_thres:
        guidance_thres_mm = 20
    else:
        guidance_thres_mm = 200
    gt_trace = inf_volume.sint_segs_dense_vox[key]
    gt_trace_mm = vox_coords_to_mm(gt_trace, inf_volume.spacing)
    seed_point_vox = gt_trace[gt_trace.shape[0] // 3]
    seed_point_mm = gt_trace_mm[gt_trace_mm.shape[0] // 3]
    print(f'vox_center: {seed_point_vox}, mm_center: {seed_point_mm}')
    init_prev_dir = get_gt_backward_dir(seed_point_mm, gt_trace_mm)
    probs, vecs = get_probs_at_location(
        inf_volume, model, seed_point_mm,
        point_in_voxspace=False,
        prev_dir_world=init_prev_dir
    )
    vecs = np.asarray(vecs)
    dir_A, prob_A = get_spheredistr_maxprob(probs, vecs, ret_prob=True)
    dir_B, prob_B = get_spheredistr_maxprob(probs, vecs, avoid=dir_A, mindist=opt.min_maxprob_dist, ret_prob=True)
    point_bw = gt_trace_mm[gt_trace.shape[0] // 2 - 1]
    point_fw = gt_trace_mm[gt_trace.shape[0] // 2 + 1]
    dir_bw = (point_bw - seed_point_mm) / np.linalg.norm(point_bw - seed_point_mm)
    dir_fw = (point_fw - seed_point_mm) / np.linalg.norm(point_fw - seed_point_mm)
    error_bw = [dir_A - dir_bw, dir_B - dir_bw]
    error_fw = [dir_A - dir_fw, dir_B - dir_fw]
    dist_bw = [np.linalg.norm(error_bw[0]), np.linalg.norm(error_bw[1])]
    dist_fw = [np.linalg.norm(error_fw[0]), np.linalg.norm(error_fw[1])]
    if dist_bw[0] + dist_fw[1] > dist_bw[1] + dist_fw[0]:
        dir_rec_bw = dir_B
        prob_rec_bw = prob_B
        dir_rec_fw = dir_A
        prob_rec_fw = prob_A
    else:
        dir_rec_bw = dir_A
        prob_rec_bw = prob_A
        dir_rec_fw = dir_B
        prob_rec_fw = prob_B
    median_trace_bw, all_traces_bw, all_confs_bw, all_dice = stochastic_track_trace(model, inf_volume, seed_point_mm, dir_rec_bw, gt_trace_mm, guidance_thres_mm, low_dice_dir=low_dice_dir)
    median_trace_fw, all_traces_fw, all_confs_fw, all_dice = stochastic_track_trace(model, inf_volume, seed_point_mm, dir_rec_fw, gt_trace_mm, guidance_thres_mm, low_dice_dir=low_dice_dir)
    median_trace_bw.reverse()
    median_trace = median_trace_bw + median_trace_fw
    all_traces = all_traces_bw + all_traces_fw
    all_confs = all_confs_bw + all_confs_fw
    return median_trace, all_traces, all_confs, all_dice

@torch.no_grad()
def accumulate_patch_segmentation(netG_A: torch.nn.Module,
                                  inf_volume,
                                  trace_mm,
                                  patch_size: int,
                                  stride_mm: float):
    """
    Fuse the soft-max outputs of only those patches the tracker visits back
    into a full-volume probability map.

    Returns
    -------
    probs_accum : [C,X,Y,Z]  – class probabilities (C = net output_nc)
    votes_accum : [  X,Y,Z]  – vote count per voxel
    """
    vol_shape  = inf_volume.data.shape[1:]            # (X,Y,Z)
    device     = next(netG_A.parameters()).device
    probs_accum = None                                # lazy - allocate once we know C
    votes_accum = torch.zeros(vol_shape, dtype=torch.int32, device=device)

    r = patch_size // 2                               # cube radius in voxels

    # ── sample approximately every *stride_mm* along the trace ───────────────
    arc = np.asarray(trace_mm, dtype=np.float32)
    seg_len = np.insert(np.cumsum(np.linalg.norm(np.diff(arc, axis=0), axis=1)), 0, 0.0)
    sample_locs = np.interp(
        np.arange(0, seg_len[-1] + 1e-3, stride_mm), seg_len, np.arange(len(arc))
    ).astype(int)

    for idx in sample_locs:
        loc_mm = arc[idx]

        # crop network‑sized patch;  rotate=False keeps axis‑aligned cube
        patch, _, _ = inf_volume.get_patch(loc_mm, rotate=False, convert_to_mm=True)
        logits, _ = netG_A(patch[None].to(device))        # [1, C, p, p, p]
        probs = logits[0].softmax(0)                      # [C, p, p, p]

        if probs_accum is None:                      # first patch → allocate
            C = probs.shape[0]                       # e.g. 5
            probs_accum = torch.zeros((C, *vol_shape),
                                       dtype=torch.float32,
                                       device=device)

        # voxel centre of the cube
        cx, cy, cz = np.asarray(mm_coords_to_vox(loc_mm, inf_volume.spacing)).astype(int) - 1
        r = patch_size // 2                                      # = 16 for p=32

        # ----- clamp to valid volume coordinates ---------------------------------
        x0, x1 = cx - r, cx + r
        y0, y1 = cy - r, cy + r
        z0, z1 = cz - r, cz + r

        x0v, x1v = max(0, x0), min(vol_shape[0], x1)
        y0v, y1v = max(0, y0), min(vol_shape[1], y1)
        z0v, z1v = max(0, z0), min(vol_shape[2], z1)

        # ----- crop the network output so shapes match ---------------------------
        px0, px1 = x0v - x0, x0v - x0 + (x1v - x0v)
        py0, py1 = y0v - y0, y0v - y0 + (y1v - y0v)
        pz0, pz1 = z0v - z0, z0v - z0 + (z1v - z0v)

        probs_accum[:, x0v:x1v, y0v:y1v, z0v:z1v] += probs[:, px0:px1, py0:py1, pz0:pz1]
        votes_accum[  x0v:x1v, y0v:y1v, z0v:z1v] += 1

    # avoid div‑by‑zero (voxels never covered keep prob = 0)
    probs_accum /= torch.clamp(votes_accum, min=1)
    return probs_accum.cpu(), votes_accum.cpu()

def create_centerline_mask(inf_volume,
                           generated_cls_mm: List[np.ndarray],
                           annotated_cls_mm: List[np.ndarray],
                           dilate: bool = False,
                           dilation_iterations: int = 1):
    """Return a label volume where 1 = generated centre‑line voxels,
    2 = annotated (manual) centre‑line voxels.  Optionally dilate each set.
    """
    vol_shape = inf_volume.data.shape[1:]
    mask = np.zeros(vol_shape, dtype=np.uint8)

    def draw_polyline(pts_mm, label):
        for pt in pts_mm:
            x, y, z = np.asarray(mm_coords_to_vox(pt, inf_volume.spacing)).astype(int) - 1
            if 0 <= x < vol_shape[0] and 0 <= y < vol_shape[1] and 0 <= z < vol_shape[2]:
                mask[x, y, z] = label

    for cl in generated_cls_mm:
        draw_polyline(cl, 1)
    for cl in annotated_cls_mm:
        draw_polyline(cl, 2)

    if dilate and dilation_iterations > 0:
        for lbl in (1, 2):
            mask_lbl = mask == lbl
            mask_lbl = binary_dilation(mask_lbl, iterations=dilation_iterations)
            mask[mask_lbl] = lbl
    return mask

def finalize_volume_outputs(inf_volume,
                            seg_prob_total: torch.Tensor,
                            vote_total: torch.Tensor,
                            gen_cls_mm: List[np.ndarray],
                            ann_cls_mm: List[np.ndarray],
                            save_dir_nifti: str,
                            vol_id):
    """Create and write the three requested NIfTI files for one volume."""

    # ── A. stitched prediction mask ------------------------------------------
    seg_prob_total = seg_prob_total / torch.clamp(vote_total, min=1)
    pred_mask = (seg_prob_total[1] > 0.5).numpy().astype(np.uint8)
    save_nifti_mask(inf_volume, pred_mask,
                    f"{save_dir_nifti}/{vol_id}_pred_seg.nii.gz")

    # ── B. ground‑truth mask (small‑bowel only) ------------------------------
    gt_mask = ((inf_volume.seggt.squeeze(0).permute(0, 1, 2) == 1)
               .numpy().astype(np.uint8))
    save_nifti_mask(inf_volume, gt_mask,
                    f"{save_dir_nifti}/{vol_id}_gt_seg.nii.gz")

    # ── C. combined centre‑lines -------------------------------------------
    cl_mask = create_centerline_mask(inf_volume, gen_cls_mm, ann_cls_mm,
                                     dilate=False)
    save_nifti_mask(inf_volume, cl_mask,
                    f"{save_dir_nifti}/{vol_id}_centerlines_all.nii.gz")

def save_centerline_segmentation_nifti(inf_volume, generated_centerline_mm, annotated_centerline_mm, save_path):
    """
    Create a segmentation volume with the same shape as the MRI volume that encodes:
      - 0: background
      - 1: generated centerline points
      - 2: annotated centerline points.
    """
    from data.rt_colon_dataset import mm_coords_to_vox
    mri_vol = inf_volume.data.cpu().numpy()[0]
    seg_vol = np.zeros(mri_vol.shape, dtype=np.uint8)
    generated_vox = np.round(mm_coords_to_vox(generated_centerline_mm, inf_volume.spacing)).astype(int)
    annotated_vox = np.round(mm_coords_to_vox(annotated_centerline_mm, inf_volume.spacing)).astype(int)
    generated_vox[:,0] -= 1
    generated_vox[:,1] -= 1
    annotated_vox[:,0] -= 1
    annotated_vox[:,1] -= 1
    for x, y, z in generated_vox:
        if 0 <= x < seg_vol.shape[0] and 0 <= y < seg_vol.shape[1] and 0 <= z < seg_vol.shape[2]:
            seg_vol[x, y, z] = 1
    for x, y, z in annotated_vox:
        if 0 <= x < seg_vol.shape[0] and 0 <= y < seg_vol.shape[1] and 0 <= z < seg_vol.shape[2]:
            seg_vol[x, y, z] = 2
    affine = np.diag(np.append(inf_volume.spacing, 1))
    nifti_img = nib.Nifti1Image(seg_vol, affine)
    nib.save(nifti_img, save_path)
    print(f"Saved centerline segmentation NIfTI to: {save_path}")

# --- Save a few patch examples (image patches + segmentation patches) ---
def triplanar_canvas(patch3d, patch3d_g=None, patch3d_b=None):
    """
    Convert a 3D patch to a 2D triplanar image.
    """
    pw = patch3d.shape[0]
    viz_painting = np.zeros((pw*2, pw*2, 3))
    patch3d_r = patch3d
    if patch3d_g is None:
        patch3d_g = patch3d
    if patch3d_b is None:
        patch3d_b = patch3d
    viz_painting[:pw, :pw, 0] = patch3d_r[:, :, pw//2]
    viz_painting[:pw, :pw, 1] = patch3d_g[:, :, pw//2]
    viz_painting[:pw, :pw, 2] = patch3d_b[:, :, pw//2]
    viz_painting[pw:2*pw, :pw, 0] = patch3d_r[:, pw//2, :]
    viz_painting[pw:2*pw, :pw, 1] = patch3d_g[:, pw//2, :]
    viz_painting[pw:2*pw, :pw, 2] = patch3d_b[:, pw//2, :]
    viz_painting[:pw, pw:2*pw, 0] = patch3d_r[pw//2, :, :]
    viz_painting[:pw, pw:2*pw, 1] = patch3d_g[pw//2, :, :]
    viz_painting[:pw, pw:2*pw, 2] = patch3d_b[pw//2, :, :]
    viz_painting = np.clip(viz_painting * 64, 0, 255)
    return viz_painting.astype(np.uint8)

# volume is [C, D, H, W] on GPU
def predict_full_volume(netG_A, volume):
    patch_size = (32, 32, 32)          # matches what the model saw in training
    overlap    = 0.5                   # 50 % overlap (stride = 16)
    n_classes  = 2
    # add batch dim expected by MONAI: [B,C,D,H,W]
    batched = volume[None]         

    with torch.no_grad():
        print(batched.shape)
        logits = sliding_window_inference(
            batched, roi_size=patch_size,
            sw_batch_size=4, overlap=overlap,
            predictor=lambda x: netG_A(x)[0]   # first output is seg probs
        )                                      # logits: [1,C,D,H,W]

    return logits[0]                           # drop batch dim

def show_overlay(img, pred, gt, z=None):
    z = pred.shape[2]//2 if z is None else z
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1); plt.title("MRI");       plt.axis("off")
    plt.imshow(img[0,:,:,z], cmap="gray")

    plt.subplot(1,3,2); plt.title("Prediction"); plt.axis("off")
    plt.imshow(img[0,:,:,z], cmap="gray")
    plt.imshow(pred[:,:,z], alpha=0.6, cmap="Reds")

    plt.subplot(1,3,3); plt.title("GT");        plt.axis("off")
    plt.imshow(img[0,:,:,z], cmap="gray")
    plt.imshow(gt[:,:,z],   alpha=0.6, cmap="Blues")

    plt.tight_layout();  plt.show()

def save_key_level_nifti(inf_volume,
                         pred_prob_patch, votes_patch,
                         gen_centerline_mm,
                         save_path, thr=0.5):
    """
    Create a 3-label volume for ONE key:
        1 = predicted small-bowel voxels along the trace
        2 = ground-truth small-bowel voxels
        3 = generated centre-line voxels

    pred_prob_patch : torch.Tensor  [C,X,Y,Z] from accumulate_patch_segmentation
    votes_patch     : torch.Tensor  [  X,Y,Z]           ”
    """
    sx, sy, sz = inf_volume.spacing
    affine = np.diag([sx, sy, sz, 1])

    # ── 1. stitch the prediction for this key alone ───────────────────────
    prob = (pred_prob_patch / votes_patch.clamp(min=1))[1]        # channel-1
    pred_mask = (prob > thr).cpu().numpy().astype(np.uint8)       # label 1

    # ── 2. ground-truth mask for the whole volume (already 0/1) ───────────
    gt_mask = ((inf_volume.seggt.squeeze(0).permute(0,1,2) == 1)
               .cpu().numpy().astype(np.uint8))                   # label 2

    # ── 3. voxelised generated centre-line ────────────────────────────────
    cl_mask = create_segmentation_mask(
        inf_volume, [np.asarray(gen_centerline_mm)], labels=[3], dilate=False
    )                                                             # label 3

    # ── 4. merge three layers into one label map ─────────────────────────
    combo = np.zeros_like(pred_mask, dtype=np.uint8)
    combo[pred_mask == 1] |= 1
    combo[gt_mask   == 1] |= 2
    combo[cl_mask   == 3] = 3

    nib.save(nib.Nifti1Image(combo, affine), save_path)
    print(f"Saved per-key 3-label NIfTI → {save_path}")

#########################
# Main Block
#########################
if __name__ == '__main__':
    # Parse options and set up model and dataset.
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = opt.n_traces
    opt.serial_batches = True
    opt.display_id = -1
    if opt.independent_dir:
        opt.input_nc += 3  # add 3 channels for previous direction

    gtdistthres = False
    stochastic_trace = True

    model = create_model(opt)
    model.setup(opt)
    if opt.eval or True:
        model.eval()

    opt.trainvols = str(opt.volume_number)
    opt.validationvol = opt.volume_number
    dummy_dataset = create_dataset(opt)
    inference_volume = dummy_dataset.dataset.volumes[opt.volume_number]

    # Define output directories.
    base_save_dir = f'{opt.results_dir}{opt.name}/{opt.tslice_start}_{datetime.now().strftime("%Y_%m_%d_%H_%M")}'
    save_dir_txt = f'{base_save_dir}/centerlines'
    save_dir_png = f'{base_save_dir}/png'
    save_dir_nifti = f'{base_save_dir}/nifti'
    save_dir_3d = f'{base_save_dir}/3d_matrices'
    save_dir_low_dice = f'{base_save_dir}/low_dice'
    os.makedirs(save_dir_txt, exist_ok=True)
    os.makedirs(save_dir_png, exist_ok=True)
    os.makedirs(save_dir_nifti, exist_ok=True)
    os.makedirs(save_dir_3d, exist_ok=True)
    os.makedirs(save_dir_low_dice, exist_ok=True)
    # Copy test_opt file to base_save_dir.
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    opt_file_name = os.path.join(expr_dir, 'test_opt.txt')
    copyfile(opt_file_name, f'{base_save_dir}/test_opt.txt')

    evaluate_full_mask = False
    if evaluate_full_mask:
        # Calculate dice coefficient for the full small bowel segmentation
        with torch.no_grad():
            input_tensor = inference_volume.data.cuda()

        # seg_probs: numpy array [C, D, H, W]   (after .cpu().numpy())
        seg_probs = predict_full_volume(model.netG_A, input_tensor)
        pred_labels = seg_probs.argmax(axis=0)           # class-id map
        pred_mask = (pred_labels == 1).cpu().numpy().astype(np.uint8)
        # ground truth:  seggt  shape [1, Y, X, Z]  → squeeze & permute to [X,Y,Z]
        gt_mask = (inference_volume.seggt                # note the attribute name
                                .squeeze(0)             # drop channel dim
                                .permute(0, 1, 2)       # Y,X,Z  → X,Y,Z
                                .cpu()
                                .numpy()
                                .astype(np.uint8))
        dice = dice_coefficient(pred_mask, gt_mask)
        print(f"Segmentation Dice (class-1) for volume {opt.volume_number}: {dice:.4f}")

        show_overlay(inference_volume.data.cpu().numpy(),
                    pred_mask, gt_mask, z=gt_mask.shape[2]//2)
        save_small_bowel_segmentation_nifti(inference_volume, seg_probs[1].cpu().numpy(),
                                            f"{save_dir_nifti}/{opt.volume_number}_pred_seg.nii.gz")

    evaluate_tube_mask = False
    if evaluate_tube_mask:
        # 1. Compute the full volume segmentation mask
        with torch.no_grad():
            input_tensor = inference_volume.data.cuda()

        # seg_probs: numpy array [C, D, H, W]   (after .cpu().numpy())
        seg_probs = predict_full_volume(model.netG_A, input_tensor)
        pred_labels = seg_probs.argmax(axis=0)           # class-id map
        pred_mask = (pred_labels == 1).cpu().numpy().astype(np.uint8)
        # ground truth:  seggt  shape [1, Y, X, Z]  → squeeze & permute to [X,Y,Z]
        gt_mask = ((inference_volume.seggt.squeeze(0).permute(0,1,2) == 1)
                   .cpu().numpy().astype(np.uint8))

        # 2. Build the tubular ROI mask  (label==1 inside tube, 0 elsewhere)
        tube_radius_mm = 25.0                          # paper uses 25–30 mm
        spacing = inference_volume.spacing             # (sx, sy, sz)
        dilate_vox = int(round(tube_radius_mm / min(spacing)))   # conservative

        # collect centre-lines in mm
        centerlines_mm = [vox_coords_to_mm(cl_vox, spacing)
                        for cl_vox in inference_volume.sint_segs_dense_vox.values()]

        roi_mask = create_segmentation_mask(
            inference_volume,
            centerlines_mm,
            labels=[1]*len(centerlines_mm),
            dilate=True,
            dilation_iterations=dilate_vox            # voxels, not mm
        )   # numpy uint8, shape = MRI volume

        # 3. Restrict both masks to that ROI before Dice
        pred_mask_roi = (pred_mask & roi_mask.astype(np.uint8))
        gt_mask_roi   = (gt_mask & roi_mask.astype(np.uint8))

        show_overlay(inference_volume.data.cpu().numpy(),
            pred_mask_roi, gt_mask_roi, z=gt_mask_roi.shape[2]//2)
        dice_roi = dice_coefficient(pred_mask_roi, gt_mask_roi)
        print(f"Tubular-ROI Dice (class-1) for vol {opt.volume_number}: {dice_roi:.4f}")
        save_small_bowel_segmentation_nifti(inference_volume, seg_probs[1].cpu().numpy(),
                                            f"{save_dir_nifti}/{opt.volume_number}_pred_seg.nii.gz")

    device = next(model.netG_A.parameters()).device
    C = opt.output_nc                  # background + all foreground classes
    vol_shape = inference_volume.data.shape[1:]
    seg_prob_total = torch.zeros((C, *vol_shape),
                                dtype=torch.float32, device=device)
    vote_total = torch.zeros(vol_shape,
                                dtype=torch.int32, device=device)
    generated_mm_all, annotated_mm_all = [], []
    metrics_rows = []


    # Loop over each key (centerline) in the volume.
    for key in sorted(inference_volume.sint_segs_dense_vox.keys(), key=int):
        np.random.seed(42)
        print(key)
        print(inference_volume.sint_segs_dense_vox[key].shape)

        median_trace, all_traces, all_trace_confidences, all_dice = compute_stochastic_trace(model, inference_volume, key, gtdist_thres=gtdistthres, low_dice_dir=save_dir_low_dice)
        
        # Save trace text files.
        np.savetxt(f'{save_dir_txt}/{opt.volume_number}_stochasticbf_median_trace_key{key}.txt', np.asarray(median_trace))
        for i, trace in enumerate(all_traces):
            np.savetxt(f'{save_dir_txt}/{opt.volume_number}_stochasticbf_trace_n{i}_key{key}.txt', np.asarray(trace))

        gt_trace = inference_volume.sint_segs_dense_vox[key]
        annotated_centerline_mm = vox_coords_to_mm(gt_trace, inference_volume.spacing)
        precision, recall, F1 = compute_centerline_metrics(median_trace, gt_trace, inference_volume.spacing, threshold=10.0)
        msd = mean_surface_distance(np.asarray(median_trace), gt_trace, inference_volume.spacing)

        metrics_rows.append({
            "Key":       key,
            "Dir":       "Forward",
            "Steps":     len(median_trace) - 1,          # = number of moves
            "Precision": precision,
            "Recall":    recall,
            "F1":        F1,
            "Dice":      float(np.mean(all_dice)),       # ensure plain Python type
            "MSD":       msd,
        })
        print(f"Metrics for key {key}: Precision={precision:.3f}  Recall={recall:.3f}  F1={F1:.3f}  Dice≈{np.mean(all_dice):.3f}")

        # ---- segmentation along this trace ----
        # probs_k, votes_k = accumulate_patch_segmentation(
        #         model.netG_A, inference_volume, median_trace,
        #         patch_size=opt.patch_size, stride_mm=opt.step_size)
        # seg_prob_total += probs_k.to(device)
        # vote_total     += votes_k.to(device)
        # generated_mm_all.append(np.asarray(median_trace))
        # annotated_mm_all.append(annotated_centerline_mm)

        # --- per-key 3-label NIfTI -------------------------------------------
        # key_nifti = (f"{save_dir_nifti}/"
        #             f"{opt.volume_number}_key{key}_pred_gt_cl.nii.gz")

        # save_key_level_nifti(inference_volume,
        #                     probs_k, votes_k,
        #                     median_trace,          # generated centre-line
        #                     key_nifti)

        # --- Save PNG overlays ---
        # PNG 1: Generated vs Annotated.
        centerlines_png_1 = [np.asarray(median_trace), annotated_centerline_mm]
        styles_1 = [
            {"color": "red", "linestyle": "-", "label": "Generated Centerline"},
            {"color": "green", "linestyle": "--", "label": "Annotated Centerline"}
        ]
        png_save_path_1 = f"{save_dir_png}/{opt.volume_number}_overlay_gen_vs_ann_key{key}.png"
        plot_centerline_overlay(inference_volume, opt, centerlines_png_1, styles_1, png_save_path_1,
                                title=f"Generated vs Annotated Overlay (Volume {opt.volume_number}, Key {key})")
        
        # PNG 2: Generated, Annotated and Bidirectional.
        bidir_available = False
        reverse_trace = None
        if len(median_trace) >= 2:
            end_point = np.array(median_trace[-1])
            second_last = np.array(median_trace[-2])
            last_direction = end_point - second_last
            if np.linalg.norm(last_direction) > 0:
                initial_reverse_dir = -last_direction / np.linalg.norm(last_direction)
                guidance_threshold_mm = 200 if not gtdistthres else 20
                reverse_trace, _, _, all_dice_rev = stochastic_track_trace(model, inference_volume, end_point, initial_reverse_dir, 
                                                            vox_coords_to_mm(gt_trace, inference_volume.spacing), guidance_threshold_mm)
                bidir_available = True
        if bidir_available:
            prec_r, rec_r, f1_r = compute_centerline_metrics(reverse_trace, gt_trace, inference_volume.spacing, threshold=10.0)
            msd_r = mean_surface_distance(np.asarray(reverse_trace), (gt_trace), inference_volume.spacing)
            metrics_rows.append({
                "Key":       key,
                "Dir":       "Reverse",
                "Steps":     len(reverse_trace) - 1,
                "Precision": prec_r,
                "Recall":    rec_r,
                "F1":        f1_r,
                "Dice":      float(np.mean(all_dice_rev)),
                "MSD":       msd_r,
            })
            fwd_vox = np.round(mm_coords_to_vox(median_trace, inference_volume.spacing)).astype(int)
            prec_rf, rec_rf, f1_rf = compute_centerline_metrics(reverse_trace, fwd_vox, inference_volume.spacing, threshold=10.0)
            metrics_rows.append({
                "Key":       key,
                "Dir":       "Reverse vs Forward",
                "Precision": prec_rf,
                "Recall":    rec_rf,
                "F1":        f1_rf,
                "Dice":      np.nan,
                "MSD":       np.nan,
            })
            centerlines_png_2 = [np.asarray(median_trace), annotated_centerline_mm, np.asarray(reverse_trace)]
            styles_2 = [
                {"color": "red", "linestyle": "-", "label": "Generated Centerline"},
                {"color": "green", "linestyle": "--", "label": "Annotated Centerline"},
                {"color": "blue", "linestyle": "--", "label": "Bidirectional Centerline"}
            ]
            png_save_path_2 = f"{save_dir_png}/{opt.volume_number}_overlay_gen_ann_bid_key{key}.png"
            plot_centerline_overlay(inference_volume, opt, centerlines_png_2, styles_2, png_save_path_2,
                                    title=f"Gen, Ann & Bidirectional Overlay (Volume {opt.volume_number}, Key {key})")

        # --- Save NIfTI segmentation masks and their dilated versions ---
        # For generated vs annotated.
        mask_gen_ann = create_segmentation_mask(inference_volume, [np.asarray(median_trace), annotated_centerline_mm], labels=[1,2], dilate=False)
        mask_gen_ann_dil = create_segmentation_mask(inference_volume, [np.asarray(median_trace), annotated_centerline_mm], labels=[1,2], dilate=True, dilation_iterations=1)
        nifti_path_gen_ann = f"{save_dir_nifti}/{opt.volume_number}_segmentation_gen_vs_ann_key{key}.nii.gz"
        nifti_path_gen_ann_dil = f"{save_dir_nifti}/{opt.volume_number}_segmentation_gen_vs_ann_dilated_key{key}.nii.gz"
        save_nifti_mask(inference_volume, mask_gen_ann, nifti_path_gen_ann)
        save_nifti_mask(inference_volume, mask_gen_ann_dil, nifti_path_gen_ann_dil)
        # Save 3D matrix as text file.
        txt_mask_path_gen_ann = f"{save_dir_3d}/{opt.volume_number}_segmentation_gen_vs_ann_key{key}.txt"
        np.savetxt(txt_mask_path_gen_ann, mask_gen_ann.reshape(-1), fmt='%d')
        
        # For generated, annotated, and bidirectional if available.
        if bidir_available:
            mask_gen_ann_bid = create_segmentation_mask(inference_volume, [np.asarray(median_trace), annotated_centerline_mm, np.asarray(reverse_trace)], labels=[1,2,3], dilate=False)
            mask_gen_ann_bid_dil = create_segmentation_mask(inference_volume, [np.asarray(median_trace), annotated_centerline_mm, np.asarray(reverse_trace)], labels=[1,2,3], dilate=True, dilation_iterations=1)
            nifti_path_gen_ann_bid = f"{save_dir_nifti}/{opt.volume_number}_segmentation_gen_ann_bid_key{key}.nii.gz"
            nifti_path_gen_ann_bid_dil = f"{save_dir_nifti}/{opt.volume_number}_segmentation_gen_ann_bid_dilated_key{key}.nii.gz"
            save_nifti_mask(inference_volume, mask_gen_ann_bid, nifti_path_gen_ann_bid)
            save_nifti_mask(inference_volume, mask_gen_ann_bid_dil, nifti_path_gen_ann_bid_dil)
            txt_mask_path_gen_ann_bid = f"{save_dir_3d}/{opt.volume_number}_segmentation_gen_ann_bid_key{key}.txt"
            np.savetxt(txt_mask_path_gen_ann_bid, mask_gen_ann_bid.reshape(-1), fmt='%d')
        
        # Optional: If in debug mode, stop after processing 2 keys.
        if 'debug' in opt.name and int(key) > 1:
            print('Debug mode, stopping after 2 keys.')
            break

    # finalize_volume_outputs(
    #     inference_volume,
    #     seg_prob_total.cpu(), vote_total.cpu(),
    #     generated_mm_all, annotated_mm_all,
    #     save_dir_nifti, str(opt.volume_number))

    # --- Save Original MRI Volume (in base directory) ---
    original_vol_save_path = f"{base_save_dir}/{opt.volume_number}_original_volume.nii.gz"
    save_original_volume_nifti(inference_volume, original_vol_save_path)

    metrics_df = pd.DataFrame(metrics_rows)
    csv_path   = f"{base_save_dir}/metrics_per_key.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n📄 Saved metrics table → {csv_path}\n")

    # (Optional text file for easy human read‑out)
    txt_path = f"{base_save_dir}/metrics_per_key.txt"
    metrics_df.to_string(open(txt_path, "w"), index=False, float_format="%.4f")
    print(f"📝 Also wrote human‑readable text file → {txt_path}\n")