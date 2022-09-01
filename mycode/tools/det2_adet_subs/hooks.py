import os
import shutil
import torch
import logging

from detectron2.engine.train_loop import HookBase
from time import sleep
from fvcore.common.param_scheduler import ParamScheduler
from detectron2.solver import LRMultiplier
from collections import Counter

from mycode.tools import styler


__all__ = [
    "ColabModelCopier",
]


def _smart_identify_colab_save_dir(checkpoint_dir, colab_dir):
    if checkpoint_dir.split('/')[0] == 'output':    # Ignore 'output' prefix
        checkpoint_dir = '/'.join(checkpoint_dir.split('/')[1:])
    
    # Eliminate replicated folder name
    _path = colab_dir + "/" + checkpoint_dir
    path = _path
    idx = 1
    while os.path.exists(path) and len(os.listdir(path))!=0:
        # If path dont exist, use the path.
        # If path exists but empty, use the path (if program failed to run, an empty folder may be created).
        # If path exists and not empty, use a new path.
        idx += 1
        path = _path + f"_{idx}"
    return path


def _smart_identify_colab_resume_dir(checkpoint_dir, colab_dir):
    if checkpoint_dir.split('/')[0] == 'output':    # Ignore 'output' prefix
        checkpoint_dir = '/'.join(checkpoint_dir.split('/')[1:])
    
    # Eliminate replicated folder name
    _path = colab_dir + "/" + checkpoint_dir
    next_path = _path
    idx = 1
    while os.path.exists(next_path):
        idx += 1
        last_path = next_path
        next_path = _path + f"_{idx}"
    if last_path is None:
        raise FileNotFoundError(f'"{next_path}"')
    return last_path


class ColabModelCopier(HookBase):
    def __init__(self, periodic_checkpointer, resume=False, save_dir="../drive/MyDrive/train_result"):
        self.chkpnt_save_dir = periodic_checkpointer.checkpointer.save_dir
        self.period = periodic_checkpointer.period
        self.file_prefix = periodic_checkpointer.file_prefix
        
        if resume is False:
            self.save_dir = _smart_identify_colab_save_dir(
                checkpoint_dir=self.chkpnt_save_dir,
                colab_dir=save_dir
            )
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = _smart_identify_colab_resume_dir(
                checkpoint_dir=self.chkpnt_save_dir,
                colab_dir=save_dir
            )

        self.last_saves = []
        print("[ColabModelCopier] (resume={}) Using colab dir: {}".format(resume, styler.stylize(self.save_dir, 'blue')))

    def after_step(self, **kwargs):
        iteration = self.trainer.iter

        if (iteration + 1) % self.period == 0:

            # for pth in self.last_saves:
            #     os.remove(pth)

            model_name = "{}_{:07d}.pth".format(self.file_prefix, iteration)
            model_pth = self.chkpnt_save_dir + "/" + model_name
            info_pth = self.chkpnt_save_dir + "/last_checkpoint"

            new_model_pth = self.save_dir+"/"+model_name
            new_info_pth = self.save_dir+"/last_checkpoint"

            print("[ColabModelCopier] Copying {} to {};\n                          {} to {}".format(
                styler.stylize(model_pth, 'blue'), styler.stylize(new_model_pth, 'blue'),
                styler.stylize(info_pth, 'blue'), styler.stylize(new_info_pth, 'blue')
            ))
            # print(f"Going to copy...")
            # print(f"\tmodel_pth: {model_pth}")
            # print(f"\tinfo_pth: {info_pth}")
            # print(f"\tnew_model_pth: {new_model_pth}")
            # print(f"\tnew_info_pth: {new_info_pth}")

            shutil.copy(model_pth, new_model_pth)
            shutil.copy(info_pth, new_info_pth)
            
            self.last_saves = [new_model_pth, new_info_pth]

        if iteration >= self.trainer.max_iter - 1:

            # for pth in self.last_saves:
            #     os.remove(pth)
            
            model_name = f"{self.file_prefix}_final.pth"
            model_pth = self.chkpnt_save_dir + "/" + model_name
            info_pth = self.chkpnt_save_dir + "/last_checkpoint"

            new_model_pth = self.save_dir+"/"+model_name
            new_info_pth = self.save_dir+"/last_checkpoint"
            
            print("[ColabModelCopier] Copying {} to {};\n                          {} to {}".format(
                styler.stylize(model_pth, 'red'), styler.stylize(new_model_pth, 'red'),
                styler.stylize(info_pth, 'red'), styler.stylize(new_info_pth, 'red')
            ))
            # print(f"Going to copy...")
            # print(f"{model_pth}")
            # print(f"\tinfo_pth: {info_pth}")
            # print(f"\tnew_model_pth: {new_model_pth}")
            # print(f"\tnew_info_pth: {new_info_pth}")

            shutil.copy(model_pth, new_model_pth)
            shutil.copy(info_pth, new_info_pth)


class ColabSuspender(HookBase):

    def __init__(self, print_period=60):
        self.print_period = print_period

    def after_train(self, **kwargs):
        count = 0
        while True:
            count += 1
            print(f"[ColabSuspendHood] period={self.print_period}, count={count}")
            sleep(self.print_period)


class LRScheduler(HookBase):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self, optimizer=None, scheduler=None, verbose=True):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim.LRScheduler or fvcore.common.param_scheduler.ParamScheduler):
                if a :class:`ParamScheduler` object, it defines the multiplier over the base LR
                in the optimizer.

        If any argument is not given, will try to obtain it from the trainer.
        """
        self._optimizer = optimizer
        self._scheduler = scheduler
        self.verbose = verbose

    def before_train(self):
        self._optimizer = self._optimizer or self.trainer.optimizer
        if isinstance(self.scheduler, ParamScheduler):
            self._scheduler = LRMultiplier(
                self._optimizer,
                self.scheduler,
                self.trainer.max_iter,
                last_iter=self.trainer.iter - 1,
            )
        self._best_param_group_id = LRScheduler.get_best_param_group_id(self._optimizer)

    @staticmethod
    def get_best_param_group_id(optimizer):
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    return i
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    return i

    def after_step(self):
        if self.verbose is True:
            lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
            self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)
        self.scheduler.step()

    @property
    def scheduler(self):
        return self._scheduler or self.trainer.scheduler

    def state_dict(self):
        if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
            return self.scheduler.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
            logger = logging.getLogger(__name__)
            logger.info("Loading scheduler from state_dict ...")
            self.scheduler.load_state_dict(state_dict)
