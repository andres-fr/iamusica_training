#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
PyTorch implementation of AdamW with learning-rate warmup and cosine annealing
with warm restarts.
"""


import torch


# ##############################################################################
# # SCHEDULERS
# ##############################################################################
class ConstantSchedule:
    """
    An iterable that produces a constant value every time it is called. Usage
    example::
      s = ConstantSchedule(value=3.14)
      a = s()
      b = s()
    """

    def __init__(self, value=1.0):
        """
        """
        self.value = value
        self.reset()

    def __iter__(self):
        """
        """
        return self

    def __next__(self):
        """
        """
        result = self()
        return result

    def reset(self):
        """
        """
        self.schedule = self.schedule_generator()

    def __call__(self):
        """
        """
        result = next(self.schedule)
        return result

    def schedule_generator(self):
        """
        This method can be overriden to generate different schedules.
        """
        while True:
            yield self.value


class CosineSchedule(ConstantSchedule):
    """
    An iterable that produces a scheduled value every time it is called. The
    schedule consists of an optional ramp-up from zero to ``maximum``, followed
    by a cosine decay back to zero. The total duration is ``num_steps``, and
    the ratio of the ramp-up is determined by the ``warmup`` parameter.

    example::
      s = CosineSchedule(maximum=10, num_steps=1000, warmup=0.5)
      all_s = list(s)
    """

    PI_HALF = torch.pi / 2

    def __init__(self, maximum=1.0, num_steps=1000, warmup=0.1):
        """
        """
        # initial values
        self.maximum = maximum
        self.num_steps = num_steps
        self.warmup = warmup
        #
        self.reset()

    def schedule_generator(self):
        """
        """
        assert 0 <= self.warmup <= 1, "Warmup must be between 0 and 1!"
        warmup_steps = round(self.num_steps * self.warmup)
        cosine_steps = self.num_steps - warmup_steps
        #
        step = 0
        while step < warmup_steps:
            value = self.maximum * (step / warmup_steps)
            #
            step += 1
            yield value
        #
        step = 0
        while step < cosine_steps:
            value = self.maximum * torch.cos(
                torch.tensor(self.PI_HALF *
                             (step / (cosine_steps - 1)))).item()
            #
            step += 1
            yield value


class CosineWrSchedule(CosineSchedule):
    """
    An iterable that produces a scheduled pair ``(value, is_cycle_end)`` every
    time it is called. The schedule follows cyclical cosine decays, from
    ``maximum`` to zero. The first cycle has ``period`` number of steps, and
    the subsequent cycles ``decay`` in value and ``slowdown`` in period.

    The second value, ``is_cycle_end``, is a boolean that is true only if
    the associated value is the end of a cycle.

    Optionally, a ramp-up from zero to ``maximum`` can be applied before the
    first cycle, with a duration given in ratio to ``period``. This ramp-up
    doesn't replace the regular cycles, it is prepended to the schedule.

    Usage example:
    s = CosineWrSchedule(1, 100, decay=0.5, slowdown=2, warmup=0.5)
    values, is_end = zip(*(s() for _ in range(1000)))
    """

    def __init__(self, maximum=1.0, period=1000, decay=1.0, slowdown=1.0,
                 warmup=0.1):
        """
        """
        # initial values
        self._initial_maximum = maximum
        self._initial_period = period
        self._decay = decay
        self._slowdown = slowdown
        self.warmup = warmup
        #
        self.reset()

    def cycle_generator(self, maxval, period):
        """
        """
        step = 0
        while step < (period - 1):
            value = maxval * torch.cos(
                torch.tensor(self.PI_HALF * (step / (period - 1)))).item()
            #
            step += 1
            yield (value, False)
        # finally, for step = (period - 1)
        value = maxval * torch.cos(
            torch.tensor(self.PI_HALF * (step / (period - 1)))).item()
        yield (value, True)

    def schedule_generator(self):
        """
        """
        assert 0 <= self.warmup <= 1, "Warmup must be between 0 and 1!"
        warmup_steps = round(self._initial_period * self.warmup)
        # warmup
        step = 0
        while step < warmup_steps:
            value = self._initial_maximum * (step / warmup_steps)
            #
            step += 1
            yield (value, False)
        # cosine cycles
        maxval = self._initial_maximum
        period = self._initial_period
        while True:
            yield from self.cycle_generator(maxval, period)
            maxval *= self._decay
            period *= self._slowdown


# ##############################################################################
# # SCHEDULED OPTIMIZERS
# ##############################################################################
class ScheduledOptimizer:
    """
    This is a mix-in that provides a ``set_lr`` and ``get_lr`` method to a
    PyTorch optimizer. It assumes that the optimizer has the same learning rate
    for all its parameters.
    """

    def set_lr(self, lr_val):
        """
        """
        for g in self.param_groups:
            g["lr"] = lr_val

    def get_lr(self):
        """
        """
        lr_list = [g["lr"] for g in self.param_groups]
        result = lr_list[0]
        assert all(lr == result for lr in lr_list), \
            "Different learning rates per group not supported"
        return result


class SGDR(torch.optim.SGD, ScheduledOptimizer):
    """
    SGD with a CosineWrSchedule. Apart from the extra parameters at
    initialization, it can be used as a regular PyTorch optimizer (assuming
    it has the same LR for all its parameters).
    """
    def __init__(self, params, lr_max=1e-3, lr_period=1000, lr_decay=1.0,
                 lr_slowdown=1.0, cycle_warmup=0.1, cycle_end_hook_fn=None,
                 **kwargs):
        """
        See CosineWrSchedule for information about the parameters.
        """
        self.lr_sched = CosineWrSchedule(
            lr_max, lr_period, lr_decay, lr_slowdown, cycle_warmup)
        self.cycle_end_hook_fn = cycle_end_hook_fn
        super(self.__class__, self).__init__(params, **kwargs)

    def step(self, *args, **kwargs):
        """
        """
        # if we provided a scheduler, apply/update it it
        lr, is_cycle_end = self.lr_sched()
        self.set_lr(lr)
        if is_cycle_end and self.cycle_end_hook_fn is not None:
            self.cycle_end_hook_fn()
        # regular opt step
        super(self.__class__, self).step(*args, **kwargs)


class AdamWR(torch.optim.AdamW, ScheduledOptimizer):
    """
    AdamW with a CosineWrSchedule. Apart from the extra parameters at
    initialization, it can be used as a regular PyTorch optimizer (assuming
    it has the same LR for all its parameters).
    """
    def __init__(self, params, lr_max=1e-3, lr_period=1000, lr_decay=1.0,
                 lr_slowdown=1.0, cycle_warmup=0.1, cycle_end_hook_fn=None,
                 **kwargs):
        """
        See CosineWrSchedule for information about the parameters.
        """
        self.lr_sched = CosineWrSchedule(
            lr_max, lr_period, lr_decay, lr_slowdown, cycle_warmup)
        self.cycle_end_hook_fn = cycle_end_hook_fn
        super(self.__class__, self).__init__(params, **kwargs)

    def step(self, *args, **kwargs):
        """
        """
        # if we provided a scheduler, apply/update it it
        lr, is_cycle_end = self.lr_sched()
        self.set_lr(lr)
        if is_cycle_end and self.cycle_end_hook_fn is not None:
            self.cycle_end_hook_fn()
        # regular opt step
        super(self.__class__, self).step(*args, **kwargs)
