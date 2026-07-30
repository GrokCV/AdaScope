from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner


@HOOKS.register_module()
class RefinerTrainValStageSwitchHook(Hook):
    """Enable refiner after a given epoch during train/val only."""

    def __init__(self,
                 enable_epoch: int = 12,
                 enabled_alpha: float = 1.0,
                 disabled_alpha: float = 0.0):
        self.enable_epoch = int(enable_epoch)
        self.enabled_alpha = float(enabled_alpha)
        self.disabled_alpha = float(disabled_alpha)

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, 'module') else model

    @staticmethod
    def _set_requires_grad(module, enabled: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad_(enabled)

    def _apply(self, runner: Runner, mode: str) -> None:
        model = self._unwrap_model(runner.model)
        enable_refiner = runner.epoch >= self.enable_epoch

        if hasattr(model, 'use_refiner'):
            model.use_refiner = bool(enable_refiner)

        if hasattr(model, 'refiner'):
            self._set_requires_grad(getattr(model, 'refiner', None), enable_refiner)

        if hasattr(model, 'set_refiner_handoff_alpha'):
            alpha = self.enabled_alpha if enable_refiner else self.disabled_alpha
            model.set_refiner_handoff_alpha(alpha)

        runner.logger.info(
            f'[RefinerTrainValStageSwitchHook] mode={mode} '
            f'epoch={runner.epoch} enable_refiner={enable_refiner}')

    def before_train_epoch(self, runner: Runner) -> None:
        self._apply(runner, mode='train')

    def before_val_epoch(self, runner: Runner) -> None:
        self._apply(runner, mode='val')
