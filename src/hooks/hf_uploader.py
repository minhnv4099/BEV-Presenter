#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from __future__ import annotations

import os.path
import os.path as osp
from typing import Optional, List, Union, TYPE_CHECKING

import mmengine
from huggingface_hub import HfApi, save_torch_state_dict
from mmengine.runner.checkpoint import find_latest_checkpoint
from .hook import Hook
from src.registry import HOOKS
from src.utils.logging import getLogger

if TYPE_CHECKING:
    from src.runner.runner import Runner

DATA_BATCH = Optional[Union[dict, tuple, list]]

logger = getLogger(__name__)


@HOOKS.register_module()
class CheckpointUploader(Hook):

    priority = "VERY_LOW"

    def __init__(self,
                 repo_id: str,
                 token: Optional[str] = None,
                 interval: int = 1,
                 by_epoch: bool = True,
                 max_keep_ckpts: int = -1,
                 save_last: bool = True,
                 save_best: Union[str, List[str], None] = None,
                 filename_tmpl: Optional[str] = None,
                 save_begin: int = 0,
                 **kwargs):
        self.hfapi = HfApi(token=token)
        self.repo_id = repo_id
        self.token = token
        self.interval = interval
        self.by_epoch = by_epoch
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.save_begin = save_begin
        self.args = kwargs

        if filename_tmpl is None:
            if self.by_epoch:
                self.filename_tmpl = 'epoch_{step}.pth'
            else:
                self.filename_tmpl = 'iter_{step}.pth'
        else:
            self.filename_tmpl = filename_tmpl

    @property
    def repo_url(self):
        return f"https://huggingface.co/{self.repo_id}"

    def before_train(self, runner: 'Runner') -> None:
        """Finish all operations, related to checkpoint.

        This function will get the appropriate file client, and the directory
        to save these checkpoints of the model.

        Args:
            runner (Runner): The runner of the training process.
        """
        if not self.hfapi.repo_exists(self.repo_id, token=self.token, repo_type='model'):
            self.hfapi.create_repo(
                self.repo_id,
                token=self.token,
                private=False,
                repo_type='model'
            )
        frequency = f"after every {self.interval} {{type}}"
        if self.by_epoch:
            frequency = frequency.format(type='epochs')
        else:
            frequency = frequency.format(type='iterations')

        msg = f'Checkpoints will be pushed to repo {self.repo_url!r} {frequency}'
        runner.logger.info(msg)

    def after_train_iter(self,
                         runner: 'Runner',
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs=Optional[dict]) -> None:
        """Save the checkpoint and synchronize buffers after each iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        if self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` iterations which start at ``self.save_begin``
        # 2. reach the last iteration of training
        should_upload = (
                self.every_n_train_iters(runner, self.interval, self.save_begin) or
                (self.save_last and self.is_last_train_iter(runner))
        )
        if should_upload:
            runner.logger.info(f'Pushing checkpoint at {runner.iter + 1} iterations')
            self._push_checkpoint(runner)

    def after_train_epoch(self, runner: Runner) -> None:
        """Save the checkpoint and synchronize buffers after each epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        if not self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` epochs which start at ``self.save_begin``
        # 2. reach the last epoch of training
        should_upload = (
                self.every_n_epochs(runner, self.interval, self.save_begin) or
                (self.save_last and self.is_last_train_epoch(runner))
        )
        if should_upload:
            runner.logger.info(f'Uploading checkpoint at {runner.epoch + 1} epochs')
            # self._push_checkpoint(runner)

    def after_train(self, runner: Runner) -> None:
        runner.logger.info("Pushing visualizing data to repo...")
        self._push_tensorboard(runner)

    def _push_checkpoint(self, runner):
        resume_from = find_latest_checkpoint(runner.work_dir)
        self.hfapi.upload_file(
            path_or_fileobj=resume_from,
            path_in_repo=os.path.basename(resume_from),
            repo_id=self.repo_id,
            token=self.token
        )

    def _push_tensorboard(self, runner: Runner):
        self.hfapi.upload_folder(
            repo_id=self.repo_id,
            token=self.token,
            folder_path=runner.log_dir,
            path_in_repo=runner.experiment_name,
            repo_type='model',
        )

    def _push_safetensors(self, runner: Runner) -> None:
        """Save the current checkpoint and delete outdated checkpoint.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.by_epoch:
            step = runner.epoch + 1
            meta = dict(epoch=step, iter=runner.iter)
        else:
            step = runner.iter + 1
            meta = dict(epoch=runner.epoch, iter=step)

        ckpt_filename = self.filename_tmpl.format(step=step)
        ckpt_filename = ckpt_filename.replace('pth', 'safetensors')
        ckpt_filename += '{suffix}'
        ckpt_dir = osp.join(runner.log_dir, 'checkpoint')
        mmengine.mkdir_or_exist(ckpt_dir)

        save_torch_state_dict(
            runner.model.state_dict(),
            ckpt_dir,
            filename_pattern=ckpt_filename,
            safe_serialization=True
        )
        self.hfapi.upload_folder(repo_id=self.repo_id, folder_path=ckpt_dir)
