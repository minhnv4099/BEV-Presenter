#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from __future__ import annotations

import json
import os.path as osp
import time
import io
from contextlib import suppress
from typing import Optional, Union, TYPE_CHECKING, Dict
from tempfile import TemporaryDirectory
from huggingface_hub import HfApi, save_torch_state_dict, HfFileSystem

from .hook import Hook
from src.runner.utils import find_latest_checkpoint, find_best_checkpoint
from src.registry import HOOKS
from src.utils.fileio import load, dump
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
                 save_last: bool = True,
                 filename_tmpl: Optional[str] = None,
                 save_begin: int = 0,
                 **kwargs):
        self.hfapi = HfApi(token=token)
        self.hffs = HfFileSystem(token=token)
        self.repo_id = repo_id

        self.previous_last_ckpt = ''
        self.previous_best_ckpt = dict()

        if '/' in repo_id:
            self.repo_id = f"{repo_id}-{{repo_name}}"
        else:
            self.repo_id = f"{repo_id}/{{repo_name}}"

        self.token = token
        self.interval = interval
        self.by_epoch = by_epoch
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

    def path_format(self, path: str):
        return f"{self.repo_id}/{path}"

    def before_run(self, runner: Runner) -> None:
        self.repo_id = self.repo_id.format(repo_name=runner.experiment_name)

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
                repo_type='model',
                exist_ok=True,
            )
            time.sleep(2.0)

        frequency = f"after every {self.interval} {{type}}"
        if self.by_epoch:
            frequency = frequency.format(type='epochs')
        else:
            frequency = frequency.format(type='steps')

        runner.logger.info(f'Checkpoints will be pushed to repo {self.repo_url!r} {frequency}.')

    def after_train(self, runner: Runner) -> None:
        runner.logger.info("Pushing visualizing data and safetensors to repo...")
        self._push_tensorboard(runner)
        self._push_safetensors(runner)

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
            runner.logger.info(f'Pushing checkpoint at {runner.epoch + 1} epochs...')
            self._push_checkpoint(runner)

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
            runner.logger.info(f'Pushing checkpoint at {runner.iter + 1} steps...')
            self._push_checkpoint(runner)

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        self._push_best_ckpt(runner)

    def _push_last_ckpt(self, runner: Runner):
        prev_last_ckpt_path = osp.join(osp.join(runner.experiment_dir, 'prev_last_checkpoint'))
        if runner.need_resume:
            if osp.isfile(prev_last_ckpt_path):
                self.previous_last_ckpt = load(prev_last_ckpt_path, file_format='json')

        last_ckpt = find_latest_checkpoint(runner.experiment_dir)

        if last_ckpt is not None and osp.isfile(last_ckpt):
            logger.info("Pushing last checkpoint...")
            if last_ckpt != self.previous_last_ckpt:
                self._write_content(
                    content=last_ckpt,
                    path_in_repo=osp.basename(last_ckpt))

                self._delete_remote_file(osp.basename(self.previous_last_ckpt))
                self.previous_last_ckpt = last_ckpt

            # write last checkpoint meta
            self._write_content(
                content=osp.basename(last_ckpt),
                path_in_repo="last_checkpoint",
                content_type='str')

        dump(self.previous_last_ckpt, prev_last_ckpt_path, file_format='json')

    def _push_best_ckpt(self, runner: Runner):
        prev_best_ckpt_path = osp.join(osp.join(runner.experiment_dir, 'prev_best_checkpoint'))
        if runner.need_resume:
            if osp.isfile(prev_best_ckpt_path):
                self.previous_best_ckpt = load(prev_best_ckpt_path, file_format='json')

        best_ckpt = find_best_checkpoint(runner.experiment_dir)

        if best_ckpt is not None:
            logger.info("Pushing best checkpoint...")
            if isinstance(best_ckpt, dict):
                for ckpt_type, ckpt_path in best_ckpt.items():
                    if ckpt_path is None or not osp.isfile(ckpt_path):
                        continue

                    if best_ckpt[ckpt_type] != self.previous_best_ckpt.get(ckpt_type):
                        self._write_content(
                            content=best_ckpt[ckpt_type],
                            path_in_repo=osp.basename(ckpt_path))

                        self._delete_remote_file(
                            osp.basename(self.previous_best_ckpt.get(ckpt_type, '')))
                        self.previous_best_ckpt[ckpt_type] = ckpt_path

                    best_ckpt[ckpt_type] = osp.basename(best_ckpt[ckpt_type])

            elif isinstance(best_ckpt, str):
                self._write_content(
                    content=best_ckpt,
                    path_in_repo=osp.basename(best_ckpt))

                best_ckpt = osp.basename(best_ckpt)

            # write best checkpoint meta
            self._write_content(
                content=json.dumps(best_ckpt, indent=2),
                path_in_repo='best_checkpoint',
                content_type='str')

        dump(self.previous_best_ckpt, prev_best_ckpt_path, file_format='json')

    def _push_checkpoint(self, runner: Runner):
        """Push last and best checkpoint to hub."""
        self._push_last_ckpt(runner)
        self._push_best_ckpt(runner)

    def _push_tensorboard(self, runner: Runner):
        self.hfapi.upload_folder(
            repo_id=self.repo_id,
            token=self.token,
            folder_path=runner.log_dir,
            path_in_repo=runner.timestamp,
            repo_type='model',
            ignore_patterns=["vis_data/"],
            delete_patterns='vis_data/*',
            run_as_future=False
        )

        self.hfapi.upload_folder(
            repo_id=self.repo_id,
            token=self.token,
            folder_path=osp.join(runner.log_dir, 'vis_data'),
            path_in_repo=runner.timestamp,
            repo_type='model',
            run_as_future=False
        )

        self._write_content(
            content=runner.save_cfg_file,
            path_in_repo=osp.basename(runner.save_cfg_file),
            content_type='path')

    def _push_safetensors(self, runner: Runner) -> None:
        """Save the current checkpoint and delete outdated checkpoint.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.by_epoch:
            step = runner.epoch
        else:
            step = runner.iter

        ckpt_filename = self.filename_tmpl.format(step=step)
        ckpt_filename = ckpt_filename.replace('pth', 'safetensors')
        fname, ext = osp.splitext(ckpt_filename)

        ckpt_filename = fname + '{suffix}' + ext
        with TemporaryDirectory(dir=runner.experiment_dir) as tmpdir:
            save_torch_state_dict(
                runner.model.state_dict(),
                tmpdir,
                filename_pattern=ckpt_filename,
                safe_serialization=True
            )
            self.hfapi.upload_folder(
                repo_id=self.repo_id,
                folder_path=tmpdir,
                delete_patterns=['*.safetensors']
            )

    def _write_content(self, content: str, path_in_repo: str, content_type: Optional[str] = None):
        if content_type == 'str':
            content = io.BytesIO(content.encode())

        self.hfapi.upload_file(
            path_or_fileobj=content,
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            token=self.token,
        )

    def _delete_remote_file(self, filename: str):
        with suppress(Exception, BaseException):
            self.hfapi.delete_file(
                path_in_repo=filename,
                repo_id=self.repo_id,
                repo_type='model',
                token=self.token
            )
