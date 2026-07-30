# Copyright (c) GrokCV. All rights reserved.
from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class AdaZoomIterBootstrapHook(Hook):

    def __init__(self, start_iter: int = 0) -> None:
        self.start_iter = int(start_iter)
        self._applied = False

    def before_train(self, runner) -> None:
        if self._applied or self.start_iter <= 0:
            return
        train_loop = runner.train_loop
        if hasattr(train_loop, '_iter'):
            train_loop._iter = int(self.start_iter)
        self._applied = True
        runner.logger.info(
            f'Bootstrap IterBasedTrainLoop from iter={self.start_iter}.')
