from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8098, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--load_pretrained', action='store_true', help='load pretrained_models.pth')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--batches_per_epoch', type=int, default=100, help='max number of batches per epoch')
        parser.add_argument('--niter', type=int, default=100, help='# of epochs at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of epochs to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adam', help='[adam|sgd]')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of optimizer')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for optimizer (minimum lr for warm_cosine)')
        parser.add_argument('--lr_max', type=float, default=0.0002, help='maximum learning rate for warm_cosine')
        parser.add_argument('--lr_step_size', type=float, default=0.1, help='step size of lr decay')
        parser.add_argument('--lr_preupdate', action='store_true', help='update LR before epoch instead of after')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--L2reg', type=float, default=0, help='factor for the L2 regularisation')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine | warm_cosine]')
        parser.add_argument('--lr_continuous_update', action='store_true', help='update lr every iter, not every epoch')
        parser.add_argument('--patience', type=int, default=5, help='patience for plateau lr policy')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations (T0 in warm_cosine)')
        parser.add_argument('--clip_gradient_norms', action='store_true', help='clip gradient norms to 1')
        parser.add_argument('--val_slice', type=int, default=100, help='slice for validation error visualisation')
        parser.add_argument('--n_val_batches', type=int, default=1, help='number of validation batches')
        parser.add_argument('--non_centerline_ratio', type=float, default=0, help='ratio (0-1) for non-cl patches')
        parser.add_argument('--bidir_consistency_factor', type=float, default=0.1, help='overall weight λ for the bidirectional-consistency loss')
        parser.add_argument('--bidir_consistency_decay',  type=float, default=0.05,help='decay α: weight = exp(-α · annotation_error)')
        parser.add_argument('--bidir_delay', type=int, default=8, help='delay for bidir consistency loss')
        parser.add_argument('--bidir_every', type=int, default=4, help='compute bidirectional loss every N batches')
        parser.add_argument('--bidir_step_mm', type=float, default=0.5, help='step size (in mm) for bidirectional sampling')
        parser.add_argument('--lambda_dir', type=float, default=0.4, help='weight for bidirectional consistency loss')
        parser.add_argument('--lambda_cycle', type=float, default=0, help='currently does not do anything')

        # bidirectional-consistency helpers
        parser.add_argument('--test_time_augments', type=int, default=1, help='number of test-time rotations/augmentations')
        parser.add_argument('--scale_merge_metric', type=str, default='mean', help='how to merge multi-scale shell outputs')
        parser.add_argument('--n_traces', type=int, default=24)
        parser.add_argument('--n_steps', type=int, default=1200)
        parser.add_argument('--step_size', type=float, default=2.5, help='mm advanced per tracking step')
        parser.add_argument('--confidence_thres', type=float, default=0.0)
        parser.add_argument('--conformist_thres', type=float, default=15.0)
        parser.add_argument('--start_randomness_mm', type=float, default=1.0)
        parser.add_argument('--moving_conf_average', type=int,   default=5)
        parser.add_argument('--rebuild_median', action='store_true')
        parser.add_argument('--hard_stop_oov', action='store_true')
        parser.add_argument('--disable_oov_slack', action='store_true')
        parser.add_argument('--doubleback_mindist', type=float, default=0.0)
        parser.add_argument('--doubleback_slack_steps', type=int, default=30)
        parser.add_argument('--min_maxprob_dist', type=float, default=1.4142)

        parser.add_argument('--raydist_zerofilter', action='store_true', help='filter selfconsistent raydists that go out of frame')

        self.isTrain = True
        return parser
