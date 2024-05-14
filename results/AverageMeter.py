# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 02:22:38 2024

@author: Bram
"""

class RunningAverageMeter(object):
    """ Computes and stores the average and current value - taken from the CNF example in the torch diffeq repo """

    def __init__(self, momentum: float = 0.99) -> None:
        self.momentum = momentum
        self.reset()

    def reset(self, val = None) -> None:
        self.val = val
        self.avg = 0 if val is None else val

    def update(self, val: float) -> None:
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


class EpochAverageMeter(object):
    """ Computes the average of the loss over the epoch used for the validation run """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.avg = 0
        self.seen_value = 0
        self.seen_samples = 0

    def update(self, val: float, size: int) -> None:
        self.seen_value += val * size
        self.seen_samples += size
        self.avg = self.seen_value / self.seen_samples
