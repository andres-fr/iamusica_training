#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This module hosts re-usable functionality aimed at deep learning training.
Specifically:
* NN initialization
* Optimizers
* Loss functions
"""


import torch


# ##############################################################################
# # NN INITIALIZATION
# ##############################################################################
def init_periodic_eye_(tnsr):
    """
    Extension to non-square matrices, the diagonal of ones wraps around shorter
    dimension. Useful to initialize RNNs with identity operators.
    """
    h, w = tnsr.shape
    if w >= h:
        for beg in range(0, w, h):
            torch.nn.init.eye_(tnsr[:, beg:beg+h])
    else:
        for beg in range(0, h, w):
            torch.nn.init.eye_(tnsr[beg:beg+w, :])


def rnn_initializer(rnn_module, init_fn, bias_val=0.0):
    """
    Wrapper to initialize RNNs
    """
    for wname, w in zip(rnn_module._flat_weights_names,
                        rnn_module._flat_weights):
        if wname.startswith("weight"):
            init_fn(w)
        elif wname.startswith("bias"):
            w.data.fill_(bias_val)
        else:
            raise RuntimeError(f"Unexpected RNN weight name? {wname}")


def init_weights(module, init_fn=torch.nn.init.kaiming_normal,
                 bias_val=0.0, verbose=False):
    """
    Custom, layer-aware initializer for PyTorch modules.

    :param init_fn: initialization function, such that ``init_fn(weight)``
      modifies in-place the weight values. If ``None``, found weights won't be
      altered
    :param float bias_val: Any module with biases will initialize them to this
      constant value

    Usage example, inside of any ``torch.nn.Module.__init__`` method:

    if init_fn is not None:
            self.apply(lambda module: init_weights(module, init_fn, 0.0))

    Apply is applied recursively to any submodule inside, so this works.
    """
    if isinstance(module, (torch.nn.Linear,
                           torch.nn.Conv1d,
                           torch.nn.Conv2d)):
        if init_fn is not None:
            init_fn(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(bias_val)
    elif isinstance(module, (torch.nn.GRU, torch.nn.LSTM)):
        rnn_initializer(module, init_fn, bias_val)
    else:
        if verbose:
            print("init_weights: ignored module:", module.__class__.__name__)


# ##############################################################################
# # SGDR OPTIMIZER
# ##############################################################################
class CosineAnnealer(object):
    """
    State machine that models the learning rate behaviour as described in
    https://arxiv.org/pdf/1608.03983.pdf
    Basically, it iterates over the first quadrant of the cosinus, and
    incorporates some convenience parameters to scale min, max and freq.
    Usage example::

      annealer = CosineAnnealer(1000, 100, 50, 0.9, 2)
      for i in range(2000):
      print(">>>", i, "   ", annealer.step())
      if i==200:
          annealer.restart()

    It also implements the iterable interface, so you can treat it as an
    infinite iterator without explicitly calling ``.step()``, as follows::

    for i, (lr, wrapped) in zip(range(1000000), annealer):
        print(lr)
    """

    PI_HALF = torch.pi / 2

    def __init__(self, maximum, minimum, restart_period,
                 scaleratio_after_cycle=1.0, periodratio_after_cycle=1.0):
        """
        Input:
        * maximum, minimum: floats
        * restart_period: positive integer
        * scaleratio: once the current step reaches the end of the period,
          it is restarted. At this point, the instance's maximum and minimum
          can be modified, by multiplying them by this ratio
        * periodratio: same as scaleratio, but affects the restart_period
        """
        # initial values
        self._initial_maximum = maximum
        self._initial_minimum = minimum
        self._initial_period = restart_period
        self._scaleratio = scaleratio_after_cycle
        self._periodratio = periodratio_after_cycle
        # declare moving values and set initial values to them
        self.restart()

    def restart(self):
        """
        Resets internal step counter, min, max and period values. Calling
        ``next()`` after restart will return the initial maximum, and will
        decrease by the initial period.
        """
        self._step = -1.0
        self.current_maximum = self._initial_maximum
        self.current_minimum = self._initial_minimum
        self.current_period = self._initial_period

    def _forward_step(self):
        """
        Increments the step counter, and if it wraps over the current period,
        updates the period and min/max values by multiplying them by the scale
        and period ratios given at construction, and returns True (returns
        False otherwise).
        """
        s = self._step + 1
        if s < self.current_period:
            self._step = s
            return False
        else:
            self._step = 0
            self.current_maximum *= self._scaleratio
            self.current_minimum *= self._scaleratio
            self.current_period *= self._periodratio
            return True

    def _cos(self, scalar):
        """
        """
        # Old-school scalar operation. Now we have torch.tensor
        return torch.cos(torch.FloatTensor([scalar]))[0].item()

    def step(self):
        """
        Calculates the current value for the cosinus function, then increments
        the internal step counter, optionally wrapping it around when the
        period is completed, and finally returns the tuple (val, end_period):
        * val: the value for the current min, max and period after
          incrementing.
        * end_period: True if the step has been reset to 0 after incrementing,
          (so the next step will retrieve the start of a new period), False
          otherwise
        """
        cos_term = self._cos(self.PI_HALF * self._step /
                             (self.current_period-1))
        val = self.current_minimum + cos_term * (self.current_maximum -
                                                 self.current_minimum)
        wrapped = self._forward_step()
        return val, wrapped

    def __iter__(self):
        """
        """
        return self

    def __next__(self):
        """
        """
        val, wrapped = self.step()
        return val, wrapped


class SGDR(torch.optim.SGD):
    """
    Merger of vanilla ``torch.optim.SGD`` and a custom ``CosineAnnealer``.
    It is similar to using ``torch.optim.lr_scheduler.CosineAnnealingLR``,
    but integrated and offers a bit more flexibility. Use like a regular SGD,
    it will update the learning rate upon every call to ``step()``.
    """

    def __init__(self, params, lr_max=1e-3, lr_min=1e-5, lr_period=10000,
                 lr_decay=1.0, lr_slowdown=1.0,
                 cycle_end_hook_fn=None, **sgd_kwargs):
        """
        """
        super().__init__(params, lr_max, **sgd_kwargs)
        self.lr_annealer = CosineAnnealer(lr_max, lr_min, lr_period, lr_decay,
                                          lr_slowdown)
        self.cycle_end_hook_fn = cycle_end_hook_fn

    def _update_lr(self, value):
        """
        """
        for g in self.param_groups:
            g["lr"] = value

    def get_lr(self):
        """
        """
        lr_list = [g["lr"] for g in self.param_groups]
        result = lr_list[0]
        assert all(lr == result for lr in lr_list), \
            "Different learning rates? should never happen"
        return result

    def step(self):
        """
        Update LR before calling super step.
        """
        next_lr, wrapped = self.lr_annealer.step()
        #
        if wrapped and self.cycle_end_hook_fn is not None:
            self.cycle_end_hook_fn()
        #
        self._update_lr(next_lr)
        super().step()
        return wrapped


# ##############################################################################
# # MASKED LOSS
# ##############################################################################
class MaskedBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    """
    This module extends ``torch.nn.BCEWithlogitsloss`` with the possibility
    to multiply each scalar loss by a mask number between 0 and 1, before
    aggregating via average.
    """

    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs, reduction="none")

    def forward(self, pred, target, mask=None):
        """
        """
        eltwise_loss = super().forward(pred, target)
        if mask is not None:
            assert mask.min() >= 0, "Mask must be in [0, 1]!"
            assert mask.max() <= 1, "Mask must be in [0, 1]!"
            eltwise_loss = eltwise_loss * mask
        result = eltwise_loss.mean()
        #
        return result
