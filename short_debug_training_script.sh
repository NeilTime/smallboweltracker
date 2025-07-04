#!/bin/bash

#SBATCH --job-name=test_run             # Job name
#SBATCH --output=logs/%x_%j.out             # Standard output log (%x: job name, %j: job ID)
#SBATCH --error=logs/%x_%j.err              # Standard error log
#SBATCH --time=1:00:00                     # Max run time (HH:MM:SS)
#SBATCH --partition=luna-gpu-short           # Use long partition for actual training
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --gres=gpu:1g.10gb:1                # Request GPU
#SBATCH --nice=100
#SBATCH --cpus-per-task=2                   # Number of CPU cores per task
#SBATCH --mem=8G                           # More memory for actual training

echo "Loading modules..."
module purge
module load Anaconda3/2024.02-1
module load cuda/12.3

echo "Activating Conda environment..."
source activate myenv

echo "Python version and path:"
which python
python --version

echo "Checking PyTorch installation..."
python -c "import torch; print('Torch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda); print('GPU Count:', torch.cuda.device_count())"

echo "Starting Visdom server on port 8098..."
nohup visdom -port 8098 > visdom.log 2>&1 &
sleep 5  # Give Visdom a few seconds to start

# # Parse arguments
# runname=$1
# trainvols=$2

# if [ -z "$trainvols" ]; then
#     echo 'usage: tracking_inference_script.sh (training run name) (comma-separated list of training volume numbers)'
#     exit 1
# fi

# Create log directory if not exists
mkdir -p logs

echo "Running training script..."

runname=$1
trainvols=$2

if [ -z "$trainvols" ];
then
    echo 'usage: tracking_inference_script.sh (training run name) (comma-separated list of training volume numbers)'
    exit
fi

python train.py \
--batch_size 32 \
--backprop_crop 6 \
--beta1 0.0 \
--checkpoints_dir checkpoints \
--dataroot /home/rth/lthulshof/MScThesis/abdominal_tracker_original/data_motility \
--dataset_mode rt_colon \
--epoch 100 \
--isosample_spacing 1.5 \
--input_nc 1 \
--loss_type ce \
--lr_policy warm_cosine \
--lr_preupdate \
--lr 0.0001 \
--lr_max 0.05 \
--lr_decay_iters 100 \
--mask_outer_slices \
--masterpath /RT_BOMOPI_0## \
--model abdominal_tracker_plus_segmentation \
--name RT_colon_$runname \
--netG vnet_3d \
--ngf 16 \
--lr_step_size 0.7 \
--niter 0 \
--niter_decay 2 \
--gt_distances 10 \
--displace_augmentation_mm 2.5 \
--interp linear \
--trainvols $trainvols \
--validationvol 4 \
--L2reg 0 \
--ngf 32 \
--dir_bce_factor 250 \
--dir_bce_offset 2.25 \
--non_centerline_ratio 0 \
--optimizer sgd \
--orig_gt_spacing 0.5 \
--save_epoch_freq 1 \
--selfconsistency_factor 0 \
--seg_bce_factor 1 \
--output_nc 2 \
--patch_size 32 \
--validationvol 22
