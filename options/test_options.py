from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--volume_number', type=int, default=18, help='Volume number to load.')
        parser.add_argument('--n_steps', type=int, default=500, help='# of steps to take.')
        parser.add_argument('--n_candidates', type=int, default=1, help='number of candidate tracks to produce.')
        parser.add_argument('--start_seed', type=str, default='10.0,10.0,10.0', help='Seed loc in voxel coords.')
        parser.add_argument('--step_size', type=float, default=0.5, help='step size in mm.')
        parser.add_argument('--results_dir', type=str, default='../../results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--combine_pixelwise_results', type=str, default='first', help='first, mean')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--test_time_augments', type=int, default=16, help='how many rotated versions to test')
        parser.add_argument('--confidence_thres', type=float, default=0.04, help='stopping criterium (logit confidence)')
        parser.add_argument('--stochastic_trace', action='store_true', help='use stochastic inference with multiple traces')
        parser.add_argument('--conformist_thres', type=float, default=10, help='kill stochastic traces more than this away from the median')
        parser.add_argument('--n_traces', type=int, default=16, help='number of stochastic traces')
        parser.add_argument('--start_randomness_mm', type=float, default=1, help='amount of randomness to add to initial seed point')
        parser.add_argument('--rebuild_median', action='store_true', help='rebuild median from survivors only.')
        parser.add_argument('--doubleback_mindist', type=float, default=3.0, help='dist threshold for double back detector.')
        parser.add_argument('--doubleback_slack_steps', type=int, default=18, help='slack steps for double back detector.')
        parser.add_argument('--hard_stop_oov', action='store_true', help='hard stop tracker 0.5vox out of view (assumes xyz spacing!)')
        parser.add_argument('--disable_oov_slack', action='store_true', help='disable 0.5vox oov slack')
        parser.add_argument('--semi_stochastic', action='store_true', help='only randomize start points.')
        parser.add_argument('--stepwise_cohort', action='store_true', help='use stepwise cohort system')
        parser.add_argument('--cohort_divergence_steps', type=int, default=5, help='number of steps in cohort initialisation.')
        parser.add_argument('--cohort_max_steps', type=int, default=30, help='number of steps per cohort.')
        parser.add_argument('--moving_conf_average', type=int, default=1, help='number of points in moving average on stopping criterium.')
        parser.add_argument('--min_maxprob_dist', type=float, default='1.4142',
                            help='min euclidean distance between selected points on unit sphere')
        parser.add_argument('--scale_merge_metric', type=str, default='mean',
                            help='multiscale shell metric (mean, max, [0,1,2,...]) ')
        parser.add_argument('--seg_inference_step_size', type=int, default=128, help='sliding window step size')

        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
