from __future__ import division, print_function, absolute_import
import pickle
import shutil
import os.path as osp
import warnings
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn

from .tools import mkdir_if_missing

__all__ = [
    'save_checkpoint', 'load_checkpoint', 'resume_from_checkpoint',
    'open_all_layers', 'open_specified_layers', 'count_num_param',
    'load_pretrained_weights'
]


def save_checkpoint(
    state, save_dir, is_best=False, remove_module_from_keys=False
):
    r"""Saves checkpoint.

    Args:
        state (dict): dictionary.
        save_dir (str): directory to save checkpoint.
        is_best (bool, optional): if True, this checkpoint will be copied and named
            ``model-best.pth.tar``. Default is False.
        remove_module_from_keys (bool, optional): whether to remove "module."
            from layer names. Default is False.

    Examples::
        >>> state = {
        >>>     'state_dict': model.state_dict(),
        >>>     'epoch': 10,
        >>>     'rank1': 0.5,
        >>>     'optimizer': optimizer.state_dict()
        >>> }
        >>> save_checkpoint(state, 'log/my_model')
    """
    mkdir_if_missing(save_dir)
    if remove_module_from_keys:
        # remove 'module.' in state_dict's keys
        state_dict = state['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        state['state_dict'] = new_state_dict
    # save
    epoch = state['epoch']
    fpath = osp.join(save_dir, 'model.pth.tar-' + str(epoch))
    torch.save(state, fpath)
    print('Checkpoint saved to "{}"'.format(fpath))
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model-best.pth.tar'))


def load_checkpoint(fpath):
    r"""Loads checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        >>> from torchreid.utils import load_checkpoint
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError('File path is None')
    fpath = osp.abspath(osp.expanduser(fpath))
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint


def resume_from_checkpoint(fpath, model, optimizer=None, scheduler=None):
    r"""Resumes training from a checkpoint.

    This will load (1) model weights and (2) ``state_dict``
    of optimizer if ``optimizer`` is not None.

    Args:
        fpath (str): path to checkpoint.
        model (nn.Module): model.
        optimizer (Optimizer, optional): an Optimizer.
        scheduler (LRScheduler, optional): an LRScheduler.

    Returns:
        int: start_epoch.

    Examples::
        >>> from torchreid.utils import resume_from_checkpoint
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> start_epoch = resume_from_checkpoint(
        >>>     fpath, model, optimizer, scheduler
        >>> )
    """
    print('Loading checkpoint from "{}"'.format(fpath))
    checkpoint = load_checkpoint(fpath)
    model.load_state_dict(checkpoint['state_dict'])
    print('Loaded model weights')
    if optimizer is not None and 'optimizer' in checkpoint.keys():
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Loaded optimizer')
    if scheduler is not None and 'scheduler' in checkpoint.keys():
        scheduler.load_state_dict(checkpoint['scheduler'])
        print('Loaded scheduler')
    start_epoch = checkpoint['epoch']
    print('Last epoch = {}'.format(start_epoch))
    if 'rank1' in checkpoint.keys():
        print('Last rank1 = {:.1%}'.format(checkpoint['rank1']))
    return start_epoch


def adjust_learning_rate(
    optimizer,
    base_lr,
    epoch,
    stepsize=20,
    gamma=0.1,
    linear_decay=False,
    final_lr=0,
    max_epoch=100
):
    r"""Adjusts learning rate.

    Deprecated.
    """
    if linear_decay:
        # linearly decay learning rate from base_lr to final_lr
        frac_done = epoch / max_epoch
        lr = frac_done*final_lr + (1.-frac_done) * base_lr
    else:
        # decay learning rate by gamma for every stepsize
        lr = base_lr * (gamma**(epoch // stepsize))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_bn_to_eval(m):
    r"""Sets BatchNorm layers to eval mode."""
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def open_all_layers(model):
    r"""Opens all layers in model for training.

    Examples::
        >>> from torchreid.utils import open_all_layers
        >>> open_all_layers(model)
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = True


def open_specified_layers(model, open_layers):
    r"""Opens specified layers in model for training while keeping
    other layers frozen.

    Args:
        model (nn.Module): neural net model.
        open_layers (str or list): layers open for training.

    Examples::
        >>> from torchreid.utils import open_specified_layers
        >>> # Only model.classifier will be updated.
        >>> open_layers = 'classifier'
        >>> open_specified_layers(model, open_layers)
        >>> # Only model.fc and model.classifier will be updated.
        >>> open_layers = ['fc', 'classifier']
        >>> open_specified_layers(model, open_layers)
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    if isinstance(open_layers, str):
        open_layers = [open_layers]

    for layer in open_layers:
        assert hasattr(
            model, layer
        ), '"{}" is not an attribute of the model, please provide the correct name'.format(
            layer
        )

    for name, module in model.named_children():
        if name in open_layers:
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def count_num_param(model):
    r"""Counts number of parameters in a model while ignoring ``self.classifier``.

    Args:
        model (nn.Module): network model.

    Examples::
        >>> from torchreid.utils import count_num_param
        >>> model_size = count_num_param(model)

    .. warning::
        
        This method is deprecated in favor of
        ``torchreid.utils.compute_model_complexity``.
    """
    warnings.warn(
        'This method is deprecated and will be removed in the future.'
    )

    num_param = sum(p.numel() for p in model.parameters())

    if isinstance(model, nn.DataParallel):
        model = model.module

    if hasattr(model,
               'classifier') and isinstance(model.classifier, nn.Module):
        # we ignore the classifier because it is unused at test time
        num_param -= sum(p.numel() for p in model.classifier.parameters())

    return num_param


def load_pretrained_weights(model, weight_path):
    r"""Loads pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        >>> from torchreid.utils import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path)
        )
    else:
        print(
            'Successfully loaded pretrained weights from "{}"'.
            format(weight_path)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )

def compute_model_complexity(
    model, input_size, verbose=False, only_conv_linear=True
):
    """Returns number of parameters and FLOPs.

    .. note::
        (1) this function only provides an estimate of the theoretical time complexity
        rather than the actual running time which depends on implementations and hardware,
        and (2) the FLOPs is only counted for layers that are used at test time. This means
        that redundant layers such as person ID classification layer will be ignored as it
        is discarded when doing feature extraction. Note that the inference graph depends on
        how you construct the computations in ``forward()``.

    Args:
        model (nn.Module): network model.
        input_size (tuple): input size, e.g. (1, 3, 256, 128).
        verbose (bool, optional): shows detailed complexity of
            each module. Default is False.
        only_conv_linear (bool, optional): only considers convolution
            and linear layers when counting flops. Default is True.
            If set to False, flops of all layers will be counted.

    Examples::
        >>> from torchreid import models, utils
        >>> model = models.build_model(name='resnet50', num_classes=1000)
        >>> num_params, flops = utils.compute_model_complexity(model, (1, 3, 256, 128), verbose=True)
    """
    registered_handles = []
    layer_list = []
    layer = namedtuple('layer', ['class_name', 'params', 'flops'])

    def _add_hooks(m):

        def _has_submodule(m):
            return len(list(m.children())) > 0

        def _hook(m, x, y):
            params = sum(p.numel() for p in m.parameters())
            class_name = str(m.__class__.__name__)
            flops_counter = _get_flops_counter(only_conv_linear)
            if class_name in flops_counter:
                flops = flops_counter[class_name](m, x, y)
            else:
                flops = 0
            layer_list.append(
                layer(class_name=class_name, params=params, flops=flops)
            )

        # only consider the very basic nn layer
        if _has_submodule(m):
            return

        handle = m.register_forward_hook(_hook)
        registered_handles.append(handle)

    default_train_mode = model.training

    model.eval().apply(_add_hooks)
    input = torch.rand(input_size)
    if next(model.parameters()).is_cuda:
        input = input.cuda()
    model(input) # forward

    for handle in registered_handles:
        handle.remove()

    model.train(default_train_mode)

    if verbose:
        per_module_params = defaultdict(list)
        per_module_flops = defaultdict(list)

    total_params, total_flops = 0, 0

    for layer in layer_list:
        total_params += layer.params
        total_flops += layer.flops
        if verbose:
            per_module_params[layer.class_name].append(layer.params)
            per_module_flops[layer.class_name].append(layer.flops)

    if verbose:
        num_udscore = 55
        print('  {}'.format('-' * num_udscore))
        print('  Model complexity with input size {}'.format(input_size))
        print('  {}'.format('-' * num_udscore))
        for class_name in per_module_params:
            params = int(np.sum(per_module_params[class_name]))
            flops = int(np.sum(per_module_flops[class_name]))
            print(
                '  {} (params={:,}, flops={:,})'.format(
                    class_name, params, flops
                )
            )
        print('  {}'.format('-' * num_udscore))
        print(
            '  Total (params={:,}, flops={:,})'.format(
                total_params, total_flops
            )
        )
        print('  {}'.format('-' * num_udscore))

    return total_params, total_flops