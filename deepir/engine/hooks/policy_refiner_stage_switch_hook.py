from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner


@HOOKS.register_module()
class PolicyRefinerStageSwitchHook(Hook):
    """Disable refiner before warmup ends, then freeze env and train policy."""

    def __init__(self, enable_epoch: int = 12, enable_val_epoch: int = None):
        self.enable_epoch = int(enable_epoch)
        self.enable_val_epoch = (
            int(enable_val_epoch) if enable_val_epoch is not None else int(enable_epoch))

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, 'module') else model

    def _apply(self, runner: Runner, mode: str) -> None:
        model = self._unwrap_model(runner.model)
        enabled_epoch = self.enable_val_epoch if mode == 'val' else self.enable_epoch
        enabled = runner.epoch >= enabled_epoch

        if hasattr(model, 'set_policy_refiner_stage'):
            model.set_policy_refiner_stage(enabled=enabled, mode=mode)
        elif hasattr(model, 'use_refiner'):
            model.use_refiner = bool(enabled)

        runner.logger.info(
            f'[PolicyRefinerStageSwitchHook] mode={mode} '
            f'epoch={runner.epoch} enable_refiner={enabled}')

    def before_train_epoch(self, runner: Runner) -> None:
        self._apply(runner, mode='train')

    def before_val_epoch(self, runner: Runner) -> None:
        self._apply(runner, mode='val')

    def after_val_epoch(self, runner: Runner, metrics=None) -> None:
        model = self._unwrap_model(runner.model)
        if metrics is None or getattr(model, 'policy_stage_enabled', False):
            return
        if not hasattr(model, 'maybe_update_policy_reward_baseline'):
            return

        score = None
        for key in ('merged_voc/mAP', 'pascal_voc/mAP'):
            if key in metrics:
                score = metrics[key]
                break

        if score is None:
            return

        updated = model.maybe_update_policy_reward_baseline(
            score=float(score),
            epoch=int(runner.epoch))
        if updated:
            runner.logger.info(
                f'[PolicyRefinerStageSwitchHook] cached raw baseline '
                f'epoch={runner.epoch} score={float(score):.4f}')
