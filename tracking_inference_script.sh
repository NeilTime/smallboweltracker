#!/bin/bash

runname=$1
vnum=$2
timepoint=$3

if [ -z "$timepoint" ];
then
    echo 'usage: tracking_inference_script.sh runname volume_number timepoint'
    exit
fi

python guided_tracking_evaluator.py \
--checkpoints_dir checkpoints \
--confidence_thres 0.015 \
--conformist_thres 15.0 \
--dataroot /home/louis/DATA/RT_MOTILITY/DATA_CENTERLINES \
--dataset_mode rt_colon \
--disable_oov_slack \
--doubleback_slack_steps 5 \
--epoch 100 \
--hard_stop_oov \
--isosample_spacing 1.5 \
--masterpath /RT_BOMOPI_0## \
--model abdominal_tracker_plus_segmentation \
--moving_conf_average 3 \
--n_traces 64 \
--name RT_colon_$runname \
--netG vnet_3d \
--ngf 32 \
--output_nc 2 \
--patch_size 32 \
--results_dir results \
--start_randomness_mm 5.0 \
--step_size 2.5 \
--stochastic_trace \
--test_time_augments 1 \
--tslice_start $timepoint \
--volume_number $vnum
