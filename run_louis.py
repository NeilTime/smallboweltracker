import subprocess
import sys
import time
import os
import webbrowser

def start_visdom():
    print("Starting Visdom server on port 8098...")
    visdom_log = open("visdom.log", "w")
    subprocess.Popen(["visdom", "-port", "8098"], stdout=visdom_log, stderr=subprocess.STDOUT)
    visdom_mode = input("Open Visdom? (y/n): ").strip().lower() == 'y'
    if visdom_mode:
        print("Opening Visdom in web browser...")
        time.sleep(5)  # Wait for Visdom to spin up
        webbrowser.open("http://localhost:8098")
    
def run_training(debug=False):
    print("Launching training...")

    cmd = [sys.executable, "train.py"]

    if debug:
        cmd += [
            "--batch_size", "32",
            "--backprop_crop", "6",
            "--beta1", "0.0",
            "--checkpoints_dir", "checkpoints",
            "--dataroot", r"C:\Users\P096347\MSc Thesis\abdominal_tracker_original\data_motility",
            "--dataset_mode", "louis",
            "--epoch", "100",
            "--isosample_spacing", "1.5",
            "--input_nc", "1",
            "--loss_type", "ce",
            "--lr_policy", "warm_cosine",
            "--lr_preupdate",
            "--lr", "0.0001",
            "--lr_max", "0.05",
            "--lr_decay_iters", "100",
            "--mask_outer_slices",
            "--masterpath", "MOT3D_multi_tslice_MII##a.hdf5",
            "--model", "abdominal_tracker_plus_segmentation",
            "--name", "yeye",
            "--netG", "vnet_3d",
            "--ngf", "32",
            "--seg_path", "masked_segmentations/MII##a_seg_t2_5c.nii",
            "--lr_step_size", "0.7",
            "--niter", "0",
            "--niter_decay", "2",
            "--gt_distances", "10",
            "--displace_augmentation_mm", "2.5",
            "--interp", "linear",
            "--trainvols", "4,5,7,10,13,14,15",
            "--validationvol", "22",
            "--L2reg", "0",
            "--dir_bce_factor", "150",
            "--dir_bce_offset", "1.5",
            "--non_centerline_ratio", "0",
            "--optimizer", "sgd",
            "--orig_gt_spacing", "0.5",
            "--save_epoch_freq", "1",
            "--selfconsistency_factor", "0",
            "--seg_bce_factor", "1",
            "--output_nc", "2",
            "--patch_size", "32",
            "--independent_dir",
        ]
    else:
        runname = input("Enter run name (e.g., yeye): ").strip()
        cmd += [
            "--batch_size", "32",
            "--backprop_crop", "6",
            "--checkpoints_dir", "checkpoints",
            "--dataset_mode", "louis+nifti",
            "--dataroot",
            r"C:\Users\P096347\MSc Thesis\abdominal_tracker_original\data_motility"
            r"+"
            r"C:\Users\P096347\MSc Thesis\usable_data",
            # "--dataroot", r"C:\Users\P096347\MSc Thesis\abdominal_tracker_original\data_motility",
            # "--dataset_mode", "louis",
            # "--dataroot", r"C:\Users\P096347\MSc Thesis\usable_data",
            # "--dataset_mode", "nifti",
            "--epoch", "100",
            "--isosample_spacing", "1.5",
            "--input_nc", "4", # 4 for independent_dir
            "--loss_type", "ce",
            "--lr_policy", "warm_cosine",
            "--lr_preupdate",
            "--lr", "0.0001",
            "--lr_max", "0.05",
            "--lr_decay_iters", "101",
            "--mask_outer_slices",
            "--masterpath", "MOT3D_multi_tslice_MII##a.hdf5",
            "--model", "abdominal_tracker_plus_segmentation",
            "--name", runname,
            "--netG", "vnet_3d",
            "--ngf", "32",
            "--seg_path", "masked_segmentations/MII##a_seg_t2_5c.nii",
            "--lr_step_size", "0.7",
            "--niter", "0",
            "--niter_decay", "101",
            "--gt_distances", "10",
            "--displace_augmentation_mm", "5",
            "--interp", "linear",
            "--trainvols", "5,7,10,13,14,15,16,18,19,23,25,26+3,5,6,7,8,9,10,11,12,13,14,18,19,20",
            "--validationvol", "22+2",
            # "--trainvols", "5,7,10,13,14,15,16,18,19,23,25,26",
            # "--validationvol", "22",
            # "--trainvols", "2,3,5,6,7,8,9,10,11,12,13,18,19,20",
            # "--validationvol", "14",
            "--L2reg", "0",
            # "--dir_bce_factor", "100",
            # "--dir_bce_offset", "0",
            "--dir_bce_factor", "250",
            "--dir_bce_offset", "2.25",
            "--non_centerline_ratio", "0",
            "--optimizer", "sgd",
            "--orig_gt_spacing", "0.5",
            "--save_epoch_freq", "50",
            # "--selfconsistency_factor", "2",
            "--seg_bce_factor", "1",
            "--output_nc", "2",  # 2 for background and small bowel, 5 can also be used for louis dataset
            "--patch_size", "32",
            # "--selfconsistency_delay", "0",
            # "--deep_supervision",
            "--bidir_consistency_factor", "0.1",
            "--bidir_consistency_decay", "0.05",
            "--bidir_delay", "9999",
            # "--independent_dir",
        ]

    print("Command to run:")
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    print("Training script finished with code:", result.returncode)

def run_inference():
    print("Launching inference...")

    # Ask user for inputs
    runname = input("Enter run name (e.g., yeye): ").strip()
    vnum = input("Enter volume number (e.g., 14): ").strip()
    timepoint = input("Enter timepoint (e.g., 3): ").strip()

    cmd = [
        sys.executable, "guided_tracking_evaluator.py",
        "--checkpoints_dir", "checkpoints",
        "--confidence_thres", "0.015",
        "--conformist_thres", "15.0",
        "--dataroot", r"C:\Users\P096347\MSc Thesis\abdominal_tracker_original\data_motility",
        "--dataset_mode", "louis",
        # "--dataroot", r"C:\Users\P096347\MSc Thesis\usable_data",
        # "--dataset_mode", "nifti",
        "--disable_oov_slack",
        "--doubleback_slack_steps", "5",
        "--epoch", "100",
        "--hard_stop_oov",
        "--isosample_spacing", "1.5",
        "--masterpath", "MOT3D_multi_tslice_MII##a.hdf5",
        "--model", "abdominal_tracker_plus_segmentation",
        "--seg_path", "masked_segmentations/MII##a_seg_t2_5c.nii",
        "--moving_conf_average", "3",
        "--n_traces", "64",
        "--name", runname,
        "--netG", "vnet_3d",
        "--ngf", "32",
        "--output_nc", "2",  # 2 for background and small bowel, 5 can also be used for louis dataset
        "--patch_size", "32",
        "--results_dir", "results",
        "--start_randomness_mm", "5.0",
        "--step_size", "2.5",
        "--stochastic_trace",
        "--test_time_augments", "1",  # Optionally remove this to match paper exactly
        "--tslice_start", timepoint,
        "--volume_number", vnum,
        "--independent_dir",
        "--min_maxprob_dist", "1.4142",  # 1.4142 / sqrt(2) for unit sphere
    ]

    print("Command to run:")
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    print("Inference script finished with code:", result.returncode)

def main():
    print("Choose mode: (1) Train or (2) Inference")
    choice = input("Enter 1 for training, 2 for inference: ").strip()

    if choice == "1":
        debug_mode = input("Enable debug mode? (y/n): ").strip().lower() == 'y'
        start_visdom()
        run_training(debug=debug_mode)
    elif choice == "2":
        run_inference()
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()