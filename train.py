"""General-purpose training script for abdominal tracker models
See options/base_options.py and options/train_options.py for more training options.
This script was adapted from the CycleGAN code: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

import time
import csv
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training samples = %d' % dataset_size)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.attach_dataset(dataset.dataset)
    model.set_validation_input(dataset.dataset.get_valdata(), dataset.dataset.valvol)  # required for vertex locs

    if '3d' in opt.dataset_mode:
        prevphase = opt.phase 
        opt.phase = 'val'
        valset = create_dataset(opt)
        model.valset = valset
        opt.phase = prevphase

    try:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, NVMLError_NotSupported
        nvmlInit()
        use_nvml = True
        nvml_handle = nvmlDeviceGetHandleByIndex(opt.gpu_ids[0])
    except ImportError:
        use_nvml = False
        
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    logpath = f'{opt.name}_perf.csv'
    with open(logpath, 'w', newline='') as logfile:
        writer = csv.writer(logfile)
        writer.writerow(['epoch', 'epoch_time_s', 'cumulative_time_s', 'peak_mem_MB', 
                         'gpu_util_pct' if use_nvml else ''])

        print('Training starts now')
        total_start = time.time()
        # reset peak‐mem counter
        torch.cuda.reset_peak_memory_stats()

        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
            epoch_start = time.time()
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
            dataset.dataset.set_epoch(epoch)
            model.epoch = epoch

            if opt.lr_preupdate:
                model.update_learning_rate()

            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                visualizer.reset()
                total_iters += opt.batch_size
                epoch_iter += opt.batch_size

                assert opt.batch_size > 1, 'need bs > 1 for batch_norm'
                if data['A'].shape[0] > 1:        # workaround for final batch possibly being size 1 and batchnorm crashing
                    model.set_input(data)         # unpack data from dataset and apply preprocessing
                    model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

                    # # evaluate GDT reachability on this mini-batch
                    # cov = model.gdt_shape_consistency(
                    #         seed_voxels=[tuple(p.tolist()) for p in data['seed_voxels']],
                    #         spacing   = tuple(data['spacing'][0].tolist())  # any sample is fine
                    #     )
                    # # store it so it shows up in printed/visualised losses
                    # model.loss_names.append('gdt_coverage')        # once, first iteration
                    # model.loss_gdt_coverage = cov

                    if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                        save_result = total_iters % opt.update_html_freq == 0
                        model.set_validation_input(dataset.dataset.get_valdata(), dataset.dataset.valvol)
                        model.compute_visuals()
                        visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

                iter_data_time = time.time()
                if opt.lr_continuous_update:
                    model.update_learning_rate_hires(epoch, i, len(dataset))

            # sync and measure
            torch.cuda.synchronize()
            epoch_time = time.time() - epoch_start
            cum_time   = time.time() - total_start

            # peak memory (MB) and reset for next epoch
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            torch.cuda.reset_peak_memory_stats()

            if use_nvml:
                try:
                    util = nvmlDeviceGetUtilizationRates(nvml_handle).gpu
                except NVMLError_NotSupported:          # ←  catch MIG-only failure
                    util = -1                           # or '' if you prefer blank
            else:
                util = ''

            # log it
            writer.writerow([epoch, f'{epoch_time:.2f}', f'{cum_time:.2f}', f'{peak_mem:.1f}', util])
            logfile.flush()

            print(f"Epoch {epoch} done in {epoch_time:.1f}s  "
                  f"(total {cum_time//60:.0f}m{cum_time%60:.0f}s), "
                  f"peak_mem={peak_mem:.1f}MB"
                  + (f", util={util}%" if use_nvml else ""))

            if epoch % opt.save_epoch_freq == 0:              # save our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)

            if not opt.lr_preupdate:
                model.update_learning_rate()                     # update learning rates at the end of every epoch.
