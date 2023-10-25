# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np

class EarlyStop:
    """
    Early stopper, based on the mean return over the last "wait_epochs" epochs.
    
    Parameters
    ----------
    wait_epochs : int
        Number of epochs to wait before stopping after the mean return has not improved.
    delta : float
        Minimum improvement in mean return to consider an improvement.
    strict : bool
        If True, the wait_epochs is reset when the mean return improves.
    """
    def __init__(self, wait_epochs=1, min_delta=0.1, strict=True):
        self.wait_epochs = wait_epochs
        self.delta = min_delta
        self.strict = strict
        self.best_mean_return = -np.inf
        self.best_mean_return_epoch = 0
        self.waited_epochs = 0
    
    def should_stop(self, epoch, mean_return):
        if mean_return > self.best_mean_return + self.delta:
            self.best_mean_return = mean_return
            self.best_mean_return_epoch = epoch
            if self.strict:
                self.waited_epochs = 0
            else:
                self.waited_epochs -= 1
        else:
            self.waited_epochs += 1
        
        if self.waited_epochs >= self.wait_epochs:
            return True
        
        return False
