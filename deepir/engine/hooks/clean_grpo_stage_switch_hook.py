from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner


@HOOKS.register_module()
class CleanGRPOStageSwitchHook(Hook):
    """Warm up detector first, then switch to clean GRPO and track best merge policy."""

    def __init__(
        self,
        enable_epoch: int = 12,
        enable_val_epoch: int = None,
        reference_metric_key: str = 'merged_voc/mAP',
    ) -> None:
        self.enable_epoch = int(enable_epoch)
        self.enable_val_epoch = (
            int(enable_val_epoch) if enable_val_epoch is not None else int(enable_epoch))
        self.reference_metric_key = str(reference_metric_key)

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
            f'[CleanGRPOStageSwitchHook] mode={mode} '
            f'epoch={runner.epoch} enable_refiner={enabled}')

    def before_train_epoch(self, runner: Runner) -> None:
        self._apply(runner, mode='train')

    def before_val_epoch(self, runner: Runner) -> None:
        self._apply(runner, mode='val')

    def _extract_score(self, metrics):
        if metrics is None:
            return None
        for key in (self.reference_metric_key, 'merged_voc/mAP', 'pascal_voc/mAP'):
            if key in metrics:
                return float(metrics[key])
        return None

    def after_val_epoch(self, runner: Runner, metrics=None) -> None:
        model = self._unwrap_model(runner.model)
        score = self._extract_score(metrics)
        if score is None:
            return

        if not getattr(model, 'policy_stage_enabled', False):
            if not hasattr(model, 'maybe_update_policy_reward_baseline'):
                return
            updated = model.maybe_update_policy_reward_baseline(
                score=score,
                epoch=int(runner.epoch))
            if updated:
                runner.logger.info(
                    f'[CleanGRPOStageSwitchHook] cached warmup baseline '
                    f'epoch={runner.epoch} score={score:.4f}')
            return

        if not hasattr(model, 'maybe_update_grpo_reference'):
            return
        updated = model.maybe_update_grpo_reference(
            score=score,
            epoch=int(runner.epoch))
        if updated:
            runner.logger.info(
                f'[CleanGRPOStageSwitchHook] updated reference policy '
                f'epoch={runner.epoch} score={score:.4f}')
