#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from __future__ import annotations

import json
import shutil
import os.path as osp
from contextlib import suppress
from typing import Optional, TYPE_CHECKING, Literal
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from .hook import Hook
from src.registry import HOOKS
from src.utils.logging import getLogger

if TYPE_CHECKING:
    from src.runner.runner import Runner

logger = getLogger(__name__)


@HOOKS.register_module()
class CheckpointResumer(Hook):
    """Resume checkpoint from online"""

    priority = 'VERY_LOW'

    def __init__(self,
                 repo_id: str,
                 token: Optional[str] = None,
                 resume_type: Literal['last', 'best'] = 'last'):
        self.hfapi = HfApi(token=token)
        self.hffs = HfFileSystem(token=token)
        self.repo_id = repo_id
        self.resume_type = resume_type

        if '/' in repo_id:
            self.repo_id = f"{repo_id}-{{repo_name}}"
        else:
            self.repo_id = f"{repo_id}/{{repo_name}}"

        self.token = token

    @property
    def repo_url(self):
        return f"https://huggingface.co/{self.repo_id}"

    def path_format(self, text: str):
        return f"{self.repo_id}/{text}"

    def before_run(self, runner: Runner) -> None:
        self.repo_id = self.repo_id.format(repo_name=runner.experiment_name)

    def before_train(self, runner: Runner) -> None:
        if runner.need_resume and not runner.has_loaded:
            runner.logger.info(f"Loading latest checkpoint from remote {self.repo_url!r}.")
            ckpt_fpath = self._get_checkpoint_path('last')
            if ckpt_fpath is not None:
                runner.resume(filename=ckpt_fpath)
                shutil.rmtree(osp.split(ckpt_fpath)[0], ignore_errors=True)

    def before_val(self, runner: Runner) -> None:
        if runner.need_resume and not runner.has_loaded:
            runner.logger.info(f"Loading best checkpoint from remote {self.repo_url!r}.")
            ckpt_fpath = self._get_checkpoint_path('best', 'mAP')
            if ckpt_fpath is not None:
                runner.load_checkpoint(filename=ckpt_fpath)
                shutil.rmtree(osp.split(ckpt_fpath)[0], ignore_errors=True)

    def _get_checkpoint_path(self, ckpt_type: Literal['last', 'best'], metric: str = 'loss'):
        ckpt_path = None

        with suppress(FileNotFoundError):
            if ckpt_type == 'last':
                with self.hffs.open(self.path_format('last_checkpoint'), mode='r') as f:
                    ckpt_fname = f.read().strip()

                ckpt_path = hf_hub_download(repo_id=self.repo_id, filename=ckpt_fname)
            elif ckpt_type == 'best':
                with self.hffs.open(self.path_format('best_checkpoint'), mode='rb') as f:
                    ckpt = json.load(f)
                    logger.info(f'Available metrics to get best checkpoint: {[ckpt.keys()]}')

                if metric not in ckpt:
                    logger.error(f"Metric {metric} is not key to get checkpoint.")
                else:
                    ckpt_path = hf_hub_download(
                        repo_id=self.repo_id, filename=ckpt[metric])

        return ckpt_path
