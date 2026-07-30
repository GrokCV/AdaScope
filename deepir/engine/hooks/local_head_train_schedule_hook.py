from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner


@HOOKS.register_module()
class LocalHeadTrainScheduleHook(Hook):
    """Schedule when the local branch participates in training loss."""

    def __init__(self, start_epoch: int = 0, interval_iters: int = 1):
        self.start_epoch = int(start_epoch)
        self.interval_iters = int(interval_iters)
        self._base_enabled = None

        if self.start_epoch < 0:
            raise ValueError('start_epoch must be >= 0.')
        if self.interval_iters < 1:
            raise ValueError('interval_iters must be >= 1.')

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, 'module') else model

    def _ensure_base_state(self, runner: Runner) -> bool:
        if self._base_enabled is not None:
            return self._base_enabled

        model = self._unwrap_model(runner.model)
        self._base_enabled = bool(getattr(model, 'train_local_head', False))
        return self._base_enabled

    def _should_enable_local_loss(self, runner: Runner) -> bool:
        if not self._ensure_base_state(runner):
            return False
        if runner.epoch < self.start_epoch:
            return False
        if self.interval_iters == 1:
            return True
        return ((runner.iter + 1) % self.interval_iters) == 0

    def _set_local_loss_enabled(self, runner: Runner, enabled: bool) -> None:
        model = self._unwrap_model(runner.model)
        if not hasattr(model, 'train_local_head'):
            return
        use_local_head = bool(getattr(model, 'use_local_head', False))
        model.train_local_head = bool(enabled and use_local_head and self._ensure_base_state(runner))

    def before_train_epoch(self, runner: Runner) -> None:
        base_enabled = self._ensure_base_state(runner)
        runner.logger.info(
            '[LocalHeadTrainScheduleHook] '
            f'epoch={runner.epoch} base_enabled={base_enabled} '
            f'start_epoch={self.start_epoch} interval_iters={self.interval_iters}')

    def before_train_iter(self,
                          runner: Runner,
                          batch_idx: int,
                          data_batch=None) -> None:
        self._set_local_loss_enabled(runner, self._should_enable_local_loss(runner))

    def after_train(self, runner: Runner) -> None:
        self._set_local_loss_enabled(runner, self._ensure_base_state(runner))
