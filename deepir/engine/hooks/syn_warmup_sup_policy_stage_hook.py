from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner


@HOOKS.register_module()
class SynWarmupSupPolicyStageHook(Hook):
    """Three-stage schedule for warmup, supervised refiner, and policy training."""

    def __init__(
        self,
        warmup_epochs: int = 12,
        refiner_supervised_epochs: int = 4,
        reference_metric_key: str = 'merged_voc/mAP',
    ) -> None:
        self.warmup_epochs = int(warmup_epochs)
        self.refiner_supervised_epochs = int(refiner_supervised_epochs)
        self.reference_metric_key = str(reference_metric_key)

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, 'module') else model

    def _train_stage(self, epoch: int) -> str:
        if epoch < self.warmup_epochs:
            return 'warmup'
        if epoch < self.warmup_epochs + self.refiner_supervised_epochs:
            return 'refiner_supervised'
        return 'policy'

    def _val_stage(self, epoch: int) -> str:
        if epoch <= self.warmup_epochs:
            return 'warmup'
        if epoch <= self.warmup_epochs + self.refiner_supervised_epochs:
            return 'refiner_supervised'
        return 'policy'

    def _apply(self, runner: Runner, mode: str) -> str:
        model = self._unwrap_model(runner.model)
        stage = self._train_stage(runner.epoch) if mode == 'train' else self._val_stage(runner.epoch)
        if hasattr(model, 'set_training_stage'):
            model.set_training_stage(stage=stage, mode=mode)
        elif hasattr(model, 'use_refiner'):
            model.use_refiner = stage != 'warmup'
        runner.logger.info(
            f'[SynWarmupSupPolicyStageHook] mode={mode} epoch={runner.epoch} stage={stage}')
        return stage

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
        score = self._extract_score(metrics)
        if score is None:
            return
        model = self._unwrap_model(runner.model)
        stage = self._val_stage(int(runner.epoch))
        epoch = int(runner.epoch)
        if stage == 'warmup':
            if hasattr(model, 'maybe_update_policy_reward_baseline'):
                updated = model.maybe_update_policy_reward_baseline(score=score, epoch=epoch)
                if updated:
                    runner.logger.info(
                        f'[SynWarmupSupPolicyStageHook] cached warmup detector epoch={epoch} score={score:.4f}')
            return
        if stage == 'refiner_supervised':
            if hasattr(model, 'maybe_update_refiner_supervised_baseline'):
                updated = model.maybe_update_refiner_supervised_baseline(score=score, epoch=epoch)
                if updated:
                    runner.logger.info(
                        f'[SynWarmupSupPolicyStageHook] cached supervised refiner epoch={epoch} score={score:.4f}')
            return
        if hasattr(model, 'maybe_update_policy_reference'):
            updated = model.maybe_update_policy_reference(score=score, epoch=epoch)
            if updated:
                runner.logger.info(
                    f'[SynWarmupSupPolicyStageHook] updated policy reference epoch={epoch} score={score:.4f}')
