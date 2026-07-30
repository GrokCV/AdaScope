from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner


@HOOKS.register_module()
class AdaZoomStageHook(Hook):
    """Stage scheduler for AdaZoom paper-style reproduction.

    Stages:
    1. detector: train the detector on full images
    2. policy: train AdaZoom policy with REINFORCE + pseudo labels
    3. joint: collaborative training of detector and AdaZoom
    """

    def __init__(
        self,
        detector_epochs: int = 0,
        policy_epochs: int = 0,
        detector_iters: int = 90000,
        policy_iters: int = 5000,
        collaborative_policy_iters: int = 500,
        collaborative_detector_iters: int = 1000,
        detector_lr: float = 1e-3,
        policy_lr: float = 2e-5,
        collaborative_detector_lr: float = 1e-4,
        collaborative_policy_lr: float = 2e-6,
    ) -> None:
        self.detector_epochs = int(detector_epochs)
        self.policy_epochs = int(policy_epochs)
        self.detector_iters = int(detector_iters)
        self.policy_iters = int(policy_iters)
        self.collaborative_policy_iters = int(collaborative_policy_iters)
        self.collaborative_detector_iters = int(collaborative_detector_iters)
        self.detector_lr = float(detector_lr)
        self.policy_lr = float(policy_lr)
        self.collaborative_detector_lr = float(collaborative_detector_lr)
        self.collaborative_policy_lr = float(collaborative_policy_lr)

    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, 'module') else model

    def _stage(self, epoch: int, iteration: int) -> tuple:
        if self.detector_iters > 0 or self.policy_iters > 0:
            if iteration < self.detector_iters:
                return 'detector_full', self.detector_lr, False
            if iteration < self.detector_iters + self.policy_iters:
                return 'policy', self.policy_lr, False
            cycle = self.collaborative_policy_iters + self.collaborative_detector_iters
            if cycle <= 0:
                return 'policy', self.collaborative_policy_lr, True
            offset = iteration - self.detector_iters - self.policy_iters
            phase = offset % cycle
            if phase < self.collaborative_policy_iters:
                return 'policy', self.collaborative_policy_lr, True
            return 'detector_patch', self.collaborative_detector_lr, True
        if epoch < self.detector_epochs:
            return 'detector_full', self.detector_lr, False
        if epoch < self.detector_epochs + self.policy_epochs:
            return 'policy', self.policy_lr, False
        return 'policy', self.collaborative_policy_lr, True

    @staticmethod
    def _set_lr(runner: Runner, lr: float) -> None:
        optim_wrapper = getattr(runner, 'optim_wrapper', None)
        if optim_wrapper is None:
            return
        optimizer = getattr(optim_wrapper, 'optimizer', None)
        if optimizer is None:
            return
        param_groups = getattr(optimizer, 'param_groups', None)
        if param_groups is None:
            return
        for group in param_groups:
            group['lr'] = lr

    def before_train_epoch(self, runner: Runner) -> None:
        stage, lr, collaborative = self._stage(int(runner.epoch), int(getattr(runner, 'iter', 0)))
        model = self._unwrap(runner.model)
        if hasattr(model, 'set_policy_stage'):
            model.set_policy_stage(stage)
        if hasattr(model, 'set_collaborative_reward_enabled'):
            model.set_collaborative_reward_enabled(collaborative)
        self._set_lr(runner, lr)
        runner.logger.info(
            f'[AdaZoomStageHook] epoch={runner.epoch} iter={getattr(runner, "iter", 0)} stage={stage} lr={lr:.2e} collaborative={collaborative}')

    def before_train_iter(self, runner: Runner, batch_idx: int, data_batch=None) -> None:
        stage, lr, collaborative = self._stage(int(runner.epoch), int(getattr(runner, 'iter', 0)))
        model = self._unwrap(runner.model)
        if hasattr(model, 'set_policy_stage'):
            model.set_policy_stage(stage)
        if hasattr(model, 'set_collaborative_reward_enabled'):
            model.set_collaborative_reward_enabled(collaborative)
        self._set_lr(runner, lr)
