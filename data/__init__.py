"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
import numpy as np
from torch.utils.data import ConcatDataset, WeightedRandomSampler


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower():
            if issubclass(cls, BaseDataset):
                dataset = cls

    if dataset is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase."
            % (dataset_filename, target_dataset_name)
        )

    return dataset


def get_option_setter(dataset_name):
    """Return a function that injects dataset-specific default CLI options.

    Works for single names ("louis") or plus-separated lists ("louis+nifti").
    """
    # ───── single dataset ──────────────────────────────────────────────
    if '+' not in dataset_name:
        return find_dataset_using_name(dataset_name).modify_commandline_options

    # ───── composite ──────────────────────────────────────────────────
    names   = [n.strip() for n in dataset_name.split('+')]
    setters = [find_dataset_using_name(n).modify_commandline_options
               for n in names]

    def combined_setter(parser, is_train):
        # remember existing option strings
        seen = {opt for act in parser._actions for opt in act.option_strings}
        add_orig = parser.add_argument

        def add_safe(*args, **kw):
            # If already present, ignore duplicate
            if any(a in seen for a in args):
                return
            # Force list-friendly type for train/val split flags
            if any(a in ('--trainvols', '--validationvol') for a in args):
                kw['type'] = str
            action = add_orig(*args, **kw)
            seen.update(action.option_strings)
            return action

        # monkey-patch, run each dataset setter, restore
        parser.add_argument = add_safe
        for s in setters:
            parser = s(parser, is_train)
        parser.add_argument = add_orig
        return parser

    return combined_setter


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


def _split_plus(value, n):
    """Return list of length n: replicate if a single item was given."""
    parts = str(value).split('+')
    if len(parts) == 1:
        return parts * n
    if len(parts) != n:
        raise ValueError(f"Expected 1 or {n} '+'-separated items, got {len(parts)}")
    return parts


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self._concat  = ConcatDataset(datasets)
    def __len__(self):               return len(self._concat)
    def __getitem__(self, idx):      return self._concat[idx]

    # forward common helper calls to the *first* child (or broadcast)
    def set_epoch(self, epoch):
        for ds in self.datasets:
            if hasattr(ds, 'set_epoch'):
                ds.set_epoch(epoch)
    def get_valdata(self):
        return self.datasets[0].get_valdata()
    @property
    def valvol(self):
        return getattr(self.datasets[0], 'valvol', None)


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        self.opt = opt

        # allow <dataset_mode> strings like  "louis+nifti"
        mode_names = [m.strip() for m in opt.dataset_mode.split('+')]
        root_names = [r.strip() for r in opt.dataroot.split('+')]

        # allow one dataroot per dataset:  --dataroot pathA+pathB+…
        # if the user gives only one path, reuse it for every dataset
        if len(root_names) == 1:
            root_names *= len(mode_names)          # same root for all
        elif len(root_names) != len(mode_names):
            raise ValueError(
                f"--dataroot must contain either 1 path or exactly "
                f"{len(mode_names)} paths separated by '+'. "
                f"Got {len(root_names)}."
            )

        train_lists = _split_plus(getattr(opt, 'trainvols', '-1'), len(mode_names))
        val_ids     = _split_plus(getattr(opt, 'validationvol', '-1'), len(mode_names))

        # concatenation case
        if len(mode_names) > 1:
            datasets, lengths = [], []

            for i, mode in enumerate(mode_names):
                cls = find_dataset_using_name(mode)

                # save global flags so we can restore them afterwards
                save_root   = opt.dataroot
                save_tvols  = getattr(opt, 'trainvols',  None)
                save_vvol   = getattr(opt, 'validationvol', None)

                # dataset-specific overrides
                opt.dataroot       = root_names[i]
                opt.trainvols      = train_lists[i]
                opt.validationvol  = int(val_ids[i])        # dataset classes expect int 👈

                ds = cls(opt)                               # build the dataset

                # restore global flags so the next dataset sees the originals
                opt.dataroot       = save_root
                opt.trainvols      = save_tvols
                opt.validationvol  = save_vvol

                datasets.append(ds)
                lengths.append(len(ds))
                print(f"dataset [{type(ds).__name__}] from {root_names[i]} was created with train={train_lists[i]}  val={val_ids[i]}")

            # build joint dataset
            self.dataset = CombinedDataset(datasets)

            # keep batches domain-balanced
            weights = np.concatenate([np.full(n, 1.0 / n) for n in lengths]).astype(np.float32)
            sampler = WeightedRandomSampler(weights,
                                            num_samples=sum(lengths),
                                            replacement=True)
            shuffle = False      # sampler already shuffles

        # single dataset case
        else:
            cls = find_dataset_using_name(opt.dataset_mode)
            self.dataset = cls(opt)
            print(f"dataset [{type(self.dataset).__name__}] was created")
            sampler = None
            shuffle = not opt.serial_batches

        # build DataLoader (same kwargs as before, just plug in sampler/shuffle)
        num_workers = int(opt.num_threads)
        loader_kwargs = dict(
            dataset     = self.dataset,
            batch_size  = opt.batch_size,
            shuffle     = shuffle,
            sampler     = sampler,
            num_workers = num_workers,
        )
        if num_workers > 0:
            loader_kwargs.update(
                pin_memory         = True,
                persistent_workers = True,
                prefetch_factor    = 2,
            )
        self.dataloader = torch.utils.data.DataLoader(**loader_kwargs)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data