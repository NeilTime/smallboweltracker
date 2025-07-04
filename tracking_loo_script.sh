#!/bin/bash
###############################################################################
# louis_loo_array.sh  – Leave-one-out CV for the Louis tracker
#
# 1.  Make executable:   chmod +x louis_loo_array.sh
# 2.  Submit array:      sbatch louis_loo_array.sh
###############################################################################

############################ Slurm directives #################################
#SBATCH --job-name=loo_fold_%a
#SBATCH --output=logs/loo_%a_%j.out
#SBATCH --error=logs/loo_%a_%j.err
#SBATCH --time=16:00:00
#SBATCH --partition=luna-gpu-long
#SBATCH --nodes=1
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-11         # 12 folds → one task per left-out volume
###############################################################################

set -euo pipefail
module purge
module load Anaconda3/2024.02-1
module load cuda/12.8
source activate myenv
mkdir -p logs

############################ Volume bookkeeping ###############################
# Ordered list of ALL volumes that appear in the normal training set
ALL_VOLS=(4 5 7 10 13 14 15 16 18 19 22 23 25 26)

LEFT_OUT=${ALL_VOLS[$SLURM_ARRAY_TASK_ID]}           # the held-out validation volume
RUNNAME="loo_v${LEFT_OUT}"                           # checkpoints/run name

# Comma-separated list of training volumes (ALL_VOLS minus LEFT_OUT)
TRAIN_VOL_ARR=()
for v in "${ALL_VOLS[@]}"; do
  [[ $v == "$LEFT_OUT" ]] || TRAIN_VOL_ARR+=("$v")
done
IFS=',' TRAIN_VOLS="${TRAIN_VOL_ARR[*]}"
unset IFS

echo "Fold $SLURM_ARRAY_TASK_ID  ➜  validate on $LEFT_OUT   train on $TRAIN_VOLS"

############################ Start Visdom (optional) ##########################
nohup visdom -port 8098 > visdom.log 2>&1 &
sleep 5

############################ Launch training ##################################
python train.py \
  --batch_size 32 \
  --backprop_crop 6 \
  --checkpoints_dir checkpoints \
  --dataroot /scratch/rth/lthulshof/abdominal_tracker/data_motility \
  --dataset_mode louis \
  --epoch 100 \
  --isosample_spacing 1.5 \
  --input_nc 1 \
  --loss_type ce \
  --lr_policy warm_cosine \
  --lr_preupdate \
  --lr 0.0001 \
  --lr_max 0.05 \
  --lr_decay_iters 101 \
  --mask_outer_slices \
  --masterpath MOT3D_multi_tslice_MII##a.hdf5 \
  --model abdominal_tracker_plus_segmentation \
  --name "$RUNNAME" \
  --netG vnet_3d \
  --ngf 32 \
  --seg_path masked_segmentations/MII##a_seg_t2_5c.nii \
  --lr_step_size 0.7 \
  --niter 0 \
  --niter_decay 101 \
  --gt_distances 10 \
  --displace_augmentation_mm 5 \
  --interp linear \
  --trainvols "$TRAIN_VOLS" \
  --validationvol "$LEFT_OUT" \
  --L2reg 0 \
  --dir_bce_factor 250 \
  --dir_bce_offset 2.25 \
  --non_centerline_ratio 0 \
  --optimizer sgd \
  --orig_gt_spacing 0.5 \
  --save_epoch_freq 50 \
  --seg_bce_factor 1 \
  --output_nc 2 \
  --patch_size 32 \
  --bidir_consistency_factor 0.1 \
  --bidir_consistency_decay 0.05 \
  --bidir_delay 9999

############################ Optional: inference on the held-out ##############

python guided_tracking_evaluator.py \
    --checkpoints_dir checkpoints \
    --confidence_thres 0.015 \
    --conformist_thres 15.0 \
    --dataroot /scratch/rth/lthulshof/abdominal_tracker/data_motility \
    --dataset_mode louis \
    --disable_oov_slack \
    --doubleback_slack_steps 5 \
    --epoch 100 \
    --hard_stop_oov \
    --isosample_spacing 1.5 \
    --masterpath MOT3D_multi_tslice_MII##a.hdf5 \
    --model abdominal_tracker_plus_segmentation \
    --seg_path masked_segmentations/MII##a_seg_t2_5c.nii \
    --moving_conf_average 3 \
    --n_traces 64 \
    --name "$RUNNAME" \
    --netG vnet_3d \
    --ngf 32 \
    --output_nc 2 \
    --patch_size 32 \
    --results_dir results \
    --start_randomness_mm 5.0 \
    --step_size 2.5 \
    --stochastic_trace \
    --test_time_augments 1 \
    --tslice_start "1" \
    --volume_number "$LEFT_OUT" \
    --min_maxprob_dist 1.4142

###############################################################################
echo "Fold $LEFT_OUT finished."