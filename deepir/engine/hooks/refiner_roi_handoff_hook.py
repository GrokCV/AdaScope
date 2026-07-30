import math

from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner


@HOOKS.register_module()
class RefinerRoiHandoffHook(Hook):
    """Schedule refiner ROI handoff alpha for local branch inference."""

    def __init__(self,
                 warmup_epochs: int = 2,
                 transition_epochs: int = 4,
                 curve: str = 'linear',
                 test_alpha: float = 1.0):
        self.warmup_epochs = int(warmup_epochs)
        self.transition_epochs = int(transition_epochs)
        self.curve = str(curve)
        self.test_alpha = self._clamp_alpha(test_alpha)

        if self.warmup_epochs < 0:
            raise ValueError('warmup_epochs must be >= 0.')
        if self.transition_epochs < 0:
            raise ValueError('transition_epochs must be >= 0.')
        if self.curve not in ('linear', 'cosine', 'stair'):
            raise ValueError("curve must be one of: 'linear', 'cosine', 'stair'.")

    @staticmethod
    def _clamp_alpha(alpha: float) -> float:
        alpha = float(alpha)
        return max(0.0, min(1.0, alpha))

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, 'module') else model

    def _compute_train_alpha(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return 0.0

        if self.transition_epochs <= 0:
            return 1.0

        progress = float(epoch - self.warmup_epochs + 1) / float(self.transition_epochs)
        progress = max(0.0, min(1.0, progress))

        if self.curve == 'linear':
            return progress
        if self.curve == 'cosine':
            return 0.5 * (1.0 - math.cos(math.pi * progress))
        return math.ceil(progress * self.transition_epochs) / float(self.transition_epochs)

    def _set_alpha(self, runner: Runner, alpha: float, mode: str) -> None:
        model = self._unwrap_model(runner.model)
        if not hasattr(model, 'set_refiner_handoff_alpha'):
            return

        alpha = self._clamp_alpha(alpha)
        model.set_refiner_handoff_alpha(alpha)
        runner.logger.info(
            f'[RefinerRoiHandoffHook] mode={mode} epoch={runner.epoch} alpha={alpha:.4f}')

    def before_train_epoch(self, runner: Runner) -> None:
        alpha = self._compute_train_alpha(runner.epoch)
        self._set_alpha(runner, alpha, mode='train')

    def before_val_epoch(self, runner: Runner) -> None:
        alpha = self._compute_train_alpha(runner.epoch)
        self._set_alpha(runner, alpha, mode='val')

    def before_test_epoch(self, runner: Runner) -> None:
        self._set_alpha(runner, self.test_alpha, mode='test')
