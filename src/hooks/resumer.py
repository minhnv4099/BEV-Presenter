#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from __future__ import annotations

import shutil
import os.path as osp
from typing import Optional, TYPE_CHECKING
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from .hook import Hook
from src.registry import HOOKS

if TYPE_CHECKING:
    from src.runner.runner import Runner


@HOOKS.register_module()
class CheckpointResumer(Hook):
    """Resume checkpoint from online"""

    def __init__(self,
                 repo_id: str,
                 token: Optional[str] = None):
        self.hfapi = HfApi(token=token)
        self.hffs = HfFileSystem(token=token)
        self.repo_id = repo_id

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

    def before_train(self, runner: Runner) -> None:
        self.repo_id = self.repo_id.format(repo_name=runner.experiment_name)
        if runner.need_resume and not runner.has_loaded:
            runner.logger.info(f"Loading latest checkpoint from {self.repo_url!r}.")
            ckpt_fpath = self._load_checkpoint()
            if ckpt_fpath is not None:
                runner.resume(filename=ckpt_fpath)
                shutil.rmtree(osp.split(ckpt_fpath)[0], ignore_errors=True)

    def _load_checkpoint(self):
        try:
            with self.hffs.open(self.path_format('last_checkpoint'), mode='r') as f:
                ckpt_fname = f.read().strip()

            return hf_hub_download(repo_id=self.repo_id, filename=ckpt_fname)
        except:
            return None
