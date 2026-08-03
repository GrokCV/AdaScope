from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner


@HOOKS.register_module()
class SynWarmupSupGRPOStageHook(Hook):
    """Three-stage schedule for syn raw-cluster detector.

    Stages:
    1. warmup: train detector only
    2. refiner_supervised: freeze detector, supervise refiner to GT envelopes
    3. grpo: freeze detector, optimize refiner with GRPO
    """

    def __init__(
        self,
        warmup_epochs: int = 12,
        refiner_supervised_epochs: int = 4,
        reference_metric_key: str = 'merged_voc/mAP',
    ) -> None:
        self.warmup_epochs = int(warmup_epochs)
        self.refiner_supervised_epochs = int(refiner_supervised_epochs)
        self.reference_metric_key = str(reference_metric_key)
        self._loaded_checkpoint_epoch = None

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, 'module') else model

    def _train_stage(self, epoch: int) -> str:
        if epoch < self.warmup_epochs:
            return 'warmup'
        if epoch < self.warmup_epochs + self.refiner_supervised_epochs:
            return 'refiner_supervised'
        return 'grpo'

    def _val_stage(self, epoch: int) -> str:
        if epoch <= self.warmup_epochs:
            return 'warmup'
        if epoch <= self.warmup_epochs + self.refiner_supervised_epochs:
            return 'refiner_supervised'
        return 'grpo'

    def _effective_epoch(self, runner: Runner, mode: str) -> int:
        if mode == 'test' and self._loaded_checkpoint_epoch is not None:
            return int(self._loaded_checkpoint_epoch)
        return int(runner.epoch)

    def _apply(self, runner: Runner, mode: str) -> str:
        model = self._unwrap_model(runner.model)
        epoch = self._effective_epoch(runner, mode)
        stage = self._train_stage(epoch) if mode == 'train' else self._val_stage(epoch)

        if hasattr(model, 'set_training_stage'):
            model.set_training_stage(stage=stage, mode=mode)
        elif hasattr(model, 'set_policy_refiner_stage'):
            model.set_policy_refiner_stage(enabled=(stage == 'grpo'), mode=mode)
        elif hasattr(model, 'use_refiner'):
            model.use_refiner = stage != 'warmup'

        runner.logger.info(
            f'[SynWarmupSupGRPOStageHook] mode={mode} '
            f'epoch={epoch} stage={stage}')
        return stage

    def before_train_epoch(self, runner: Runner) -> None:
        self._apply(runner, mode='train')

    def before_val_epoch(self, runner: Runner) -> None:
        self._apply(runner, mode='val')

    def before_test_epoch(self, runner: Runner) -> None:
        self._apply(runner, mode='test')

    def after_load_checkpoint(self, runner: Runner, checkpoint=None) -> None:
        if checkpoint is None:
            return
        meta = checkpoint.get('meta', {})
        epoch = meta.get('epoch', None)
        self._loaded_checkpoint_epoch = int(epoch) if epoch is not None else None

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
            if not hasattr(model, 'maybe_update_policy_reward_baseline'):
                return
            updated = model.maybe_update_policy_reward_baseline(score=score, epoch=epoch)
            if updated:
                runner.logger.info(
                    f'[SynWarmupSupGRPOStageHook] cached warmup detector '
                    f'epoch={epoch} score={score:.4f}')
            return

        if stage == 'refiner_supervised':
            if not hasattr(model, 'maybe_update_refiner_supervised_baseline'):
                return
            updated = model.maybe_update_refiner_supervised_baseline(score=score, epoch=epoch)
            if updated:
                runner.logger.info(
                    f'[SynWarmupSupGRPOStageHook] cached supervised refiner '
                    f'epoch={epoch} score={score:.4f}')
            return

        if not hasattr(model, 'maybe_update_grpo_reference'):
            return
        updated = model.maybe_update_grpo_reference(score=score, epoch=epoch)
        if updated:
            runner.logger.info(
                f'[SynWarmupSupGRPOStageHook] updated GRPO reference '
                f'epoch={epoch} score={score:.4f}')
