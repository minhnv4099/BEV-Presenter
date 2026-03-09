#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from __future__ import annotations

import os.path as osp
from collections import deque
from typing import Optional, List, Dict, TYPE_CHECKING, Union
from mmengine.dist import master_only
from mmengine.hooks import CheckpointHook
from mmengine.dist import is_main_process
from mmengine.fileio import FileClient, get_file_backend
from src.registry import HOOKS
from src.utils.logging import getLogger
from src.utils.fileio import dump

if TYPE_CHECKING:
    from src.runner import Runner


logger = getLogger(__name__)


@HOOKS.register_module()
class CheckpointHookV2(CheckpointHook):

    _default_greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mDice', 'mIoU',
        'mAcc', 'aAcc', 'NDS', 'mAP'
    ]
    _default_less_keys = ['loss']

    def _init_rule(self, rules: list[str], key_indicators: list[str]) -> None:
        """Initialize rule, key_indicator, comparison_func, and best score. If
        key_indicator is a list of string and rule is a string, all metric in
        the key_indicator will share the same rule.

        Here is the rule to determine which rule is used for key indicator when
        the rule is not specific (note that the key indicator matching is case-
        insensitive):

        1. If the key indicator is in ``self.greater_keys``, the rule
            will be specified as 'greater'.
        2. Or if the key indicator is in ``self.less_keys``, the rule
            will be specified as 'less'.
        3. Or if any one item in ``self.greater_keys`` is a substring of
            key_indicator, the rule will be specified as 'greater'.
        4. Or if any one item in ``self.less_keys`` is a substring of
            key_indicator, the rule will be specified as 'less'.

        Args:
            rules (List[Optional[str]]): Comparison rule for best score.
            key_indicators (List[str]): Key indicator to determine
                the comparison rule.
        """
        if rules is None:
            rules = [None] * len(key_indicators)

        if len(rules) == 1:
            rules = rules * len(key_indicators)

        self.rules = []
        for rule, key_indicator in zip(rules, key_indicators):

            if rule not in self.rule_map and rule is not None:
                raise KeyError('rule must be greater, less or None, '
                               f'but got {rule}.')

            if rule is None and key_indicator != 'auto':
                # `_lc` here means we use the lower case of keys for
                # case-insensitive matching
                key_indicator_lc = key_indicator.lower()
                greater_keys = {key.lower() for key in self.greater_keys}
                less_keys = {key.lower() for key in self.less_keys}

                if key_indicator_lc in greater_keys:
                    rule = 'greater'
                elif key_indicator_lc in less_keys:
                    rule = 'less'
                elif any(key in key_indicator_lc for key in greater_keys):
                    rule = 'greater'
                elif any(key in key_indicator_lc for key in less_keys):
                    rule = 'less'
                else:
                    raise ValueError('Cannot infer the rule for key '
                                     f'{key_indicator}, thus a specific rule '
                                     'must be specified.')
            if rule is not None:
                self.is_better_than[key_indicator] = self.rule_map[rule]
            self.rules.append(rule)

        self.key_indicators = key_indicators

    def before_run(self, runner) -> None:
        if self.out_dir is None:
            self.out_dir = runner.work_dir

        # If self.file_client_args is None, self.file_client will not
        # used in CheckpointHook. To avoid breaking backward compatibility,
        # it will not be removed util the release of MMEngine1.0
        self.file_client = FileClient.infer_client(self.file_client_args, self.out_dir)

        if self.file_client_args is None:
            self.file_backend = get_file_backend(
                self.out_dir, backend_args=self.backend_args)
        else:
            self.file_backend = self.file_client

    def before_train(self, runner: Runner) -> None:
        """Finish all operations, related to checkpoint.

        This function will get the appropriate file client, and the directory
        to save these checkpoints of the model.

        Args:
            runner (Runner): The runner of the training process.
        """
        # if `self.out_dir` is not equal to `runner.work_dir`, it means that
        # `self.out_dir` is set so the final `self.out_dir` is the
        # concatenation of `self.out_dir` and the last level directory of
        # `runner.work_dir`
        if self.out_dir != runner.work_dir:
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.file_backend.join_path(
                self.out_dir, basename)  # type: ignore  # noqa: E501

        frequency = self._get_frequency()
        runner.logger.info(f'Checkpoints will be saved to {self.out_dir!r} {frequency}.')

        if self.save_best is not None:
            self._ini_best_ckpt(runner)
            runner.logger.info(f'The best checkpoint will be saved to {self.out_dir!r} '
                               f'based on {self.key_indicators} with rules {self.rules} {frequency}.')

        if self.max_keep_ckpts > 0:
            keep_ckpt_ids = []
            if 'keep_ckpt_ids' in runner.message_hub.runtime_info:
                keep_ckpt_ids = runner.message_hub.get_info('keep_ckpt_ids')

                while len(keep_ckpt_ids) > self.max_keep_ckpts:
                    step = keep_ckpt_ids.pop(0)
                    if is_main_process():
                        path = self.file_backend.join_path(
                            self.out_dir, self.filename_tmpl.format(step))
                        if self.file_backend.isfile(path):
                            self.file_backend.remove(path)
                        elif self.file_backend.isdir(path):
                            # checkpoints saved by deepspeed are directories
                            self.file_backend.rmtree(path)

            self.keep_ckpt_ids: deque = deque(keep_ckpt_ids, self.max_keep_ckpts)
            runner.logger.info(f"Keep maximum {self.max_keep_ckpts} checkpoints in local.")

        if self.published_keys:
            runner.logger.info(f"Publish checkpoint with keys: {self.published_keys} after training.")

    def _get_frequency(self):
        frequency = f"after every {self.interval} {{type}}"
        if self.by_epoch:
            frequency = frequency.format(type='epochs')
        else:
            frequency = frequency.format(type='steps')

        return frequency

    def _ini_best_ckpt(self, runner: Runner):
        new_best_ckpt = not osp.isfile(osp.join(runner.experiment_dir, 'best_checkpoint'))

        if len(self.key_indicators) == 1:
            if 'best_ckpt' not in runner.message_hub.runtime_info or new_best_ckpt:
                self.best_ckpt_path = None
            else:
                self.best_ckpt_path = runner.message_hub.get_info('best_ckpt')
        else:
            for key_indicator in self.key_indicators:
                best_ckpt_name = f'best_ckpt_{key_indicator}'
                if best_ckpt_name not in runner.message_hub.runtime_info or new_best_ckpt:
                    self.best_ckpt_path_dict[key_indicator] = None
                else:
                    self.best_ckpt_path_dict[key_indicator] = runner.message_hub.get_info(best_ckpt_name)

    def after_train_epoch(self, runner: Runner, outputs: Optional[dict] = None) -> None:
        """Save the checkpoint and synchronize buffers after each epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        if not self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` epochs which start at ``self.save_begin``
        # 2. reach the last epoch of training
        if self.every_n_epochs(runner, self.interval, self.save_begin) or (
                self.save_last and self.is_last_train_epoch(runner)):
            runner.logger.info(
                f'Saving checkpoint at {runner.epoch + 1} epochs')
            self._save_checkpoint(runner)

            if outputs is not None:
                self._save_best_checkpoint(runner, outputs)

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[Union[dict, tuple, list]] = None,
                         outputs: Optional[dict] = None) -> None:
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
        if (self.every_n_train_iters(runner, self.interval, self.save_begin)
                or (self.save_last and self.is_last_train_iter(runner))):
            runner.logger.info(
                f'Saving checkpoint at {runner.iter + 1} iterations')
            self._save_checkpoint(runner)

            if outputs is not None:
                self._save_best_checkpoint(runner, outputs)

    def _save_best_checkpoint(self, runner, metrics) -> None:
        """Save the current checkpoint and delete outdated checkpoint.

        Args:
            runner (Runner): The runner of the training process.
            metrics (dict): Evaluation results of all metrics.
        """
        if not self.save_best:
            return

        if self.by_epoch:
            ckpt_filename = self.filename_tmpl.format(runner.epoch)
            cur_type, cur_time = 'epoch', runner.epoch
        else:
            ckpt_filename = self.filename_tmpl.format(runner.iter)
            cur_type, cur_time = 'iter', runner.iter

        meta = dict(epoch=runner.epoch, iter=runner.iter)

        # handle auto in self.key_indicators and self.rules before the loop
        if 'auto' in self.key_indicators:
            self._init_rule(self.rules, [list(metrics.keys())[0]])

        new_best_ckpt = not osp.isfile(osp.join(runner.experiment_dir, 'best_checkpoint'))
        best_ckpt_updated = False
        # save best logic
        # get score from messagehub
        for key_indicator, rule in zip(self.key_indicators, self.rules):
            if key_indicator not in metrics:
                continue

            key_score = metrics[key_indicator]

            if len(self.key_indicators) == 1:
                best_score_key = 'best_score'
                runtime_best_ckpt_key = 'best_ckpt'
                best_ckpt_path = self.best_ckpt_path
            else:
                best_score_key = f'best_score_{key_indicator}'
                runtime_best_ckpt_key = f'best_ckpt_{key_indicator}'
                best_ckpt_path = self.best_ckpt_path_dict[key_indicator]

            if best_score_key not in runner.message_hub.runtime_info or new_best_ckpt:
                best_score = self.init_value_map[rule]
            else:
                best_score = runner.message_hub.get_info(best_score_key)

            if key_score is None or not self.is_better_than[key_indicator](
                    key_score, best_score):
                continue

            best_ckpt_updated = True

            best_score = key_score
            runner.message_hub.update_info(best_score_key, best_score)

            if best_ckpt_path and is_main_process():
                is_removed = False
                if self.file_backend.isfile(best_ckpt_path):
                    self.file_backend.remove(best_ckpt_path)
                    is_removed = True
                elif self.file_backend.isdir(best_ckpt_path):
                    # checkpoints saved by deepspeed are directories
                    self.file_backend.rmtree(best_ckpt_path)
                    is_removed = True

                if is_removed:
                    runner.logger.info(
                        f'The previous best checkpoint {best_ckpt_path} '
                        'is removed')

            best_ckpt_name = f'best_{key_indicator}_{ckpt_filename}'
            # Replace illegal characters for filename with `_`
            best_ckpt_name = best_ckpt_name.replace('/', '_')
            if len(self.key_indicators) == 1:
                self.best_ckpt_path = self.file_backend.join_path(  # type: ignore # noqa: E501
                    self.out_dir, best_ckpt_name)
                runner.message_hub.update_info(runtime_best_ckpt_key,
                                               self.best_ckpt_path)
            else:
                self.best_ckpt_path_dict[key_indicator] = self.file_backend.join_path(  # type: ignore # noqa: E501
                        self.out_dir, best_ckpt_name)
                runner.message_hub.update_info(
                    runtime_best_ckpt_key,
                    self.best_ckpt_path_dict[key_indicator])
            runner.save_checkpoint(
                self.out_dir,
                filename=best_ckpt_name,
                file_client_args=self.file_client_args,
                save_optimizer=False,
                save_param_scheduler=False,
                meta=meta,
                by_epoch=False,
                backend_args=self.backend_args)

            runner.logger.info(
                f'The best checkpoint with {best_score:0.4f} {key_indicator} '
                f'at {cur_time} {cur_type} is saved to {best_ckpt_name}.')

        # save checkpoint again to update the best_score and best_ckpt stored
        # in message_hub because the checkpoint saved in `after_train_epoch`
        # or `after_train_iter` stage only keep the previous best checkpoint
        # not the current best checkpoint which causes the current best
        # checkpoint can not be removed when resuming training.
        if best_ckpt_updated and self.last_ckpt is not None:
            self._save_checkpoint_with_step(runner, cur_time, meta)

        best_ckpt_file = osp.join(runner.experiment_dir, 'best_checkpoint')
        best_ckpt = getattr(self, 'best_ckpt_path', None) or getattr(self, 'best_ckpt_path_dict', None)
        dump(best_ckpt, best_ckpt_file, indent=2)

    def before_val(self, runner: Runner) -> None:
        frequency = self._get_frequency()
        if self.save_best is not None:
            self._ini_best_ckpt(runner)
            runner.logger.info(f'The best checkpoint will be saved to {self.out_dir!r} '
                               f'based on {self.key_indicators} with rules {self.rules} {frequency}.')

    def after_val_epoch(self, runner: Runner, metrics: dict):
        """Save the checkpoint and synchronize buffers after each evaluation
        epoch.

        Args:
            runner (Runner): The runner of the training process.
            metrics (dict): Evaluation results of all metrics
        """
        if len(metrics) == 0:
            runner.logger.warning(
                'Since `metrics` is an empty dict, the behavior to save '
                'the best checkpoint will be skipped in this evaluation.')
            return

        self._save_best_checkpoint(runner, metrics)

    def after_train(self, runner) -> None:
        """Publish the checkpoint after training.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.published_keys is None:
            return

        if self.save_last and self.last_ckpt is not None:
            self._publish_model(runner, self.last_ckpt)

        if getattr(self, 'best_ckpt_path', None) is not None:
            self._publish_model(runner, str(self.best_ckpt_path))
        if getattr(self, 'best_ckpt_path_dict', None) is not None:
            for best_ckpt in self.best_ckpt_path_dict.values():
                self._publish_model(runner, best_ckpt)

    @master_only
    def _publish_model(self, runner, ckpt_path: str) -> None:
        """Remove unnecessary keys from ckpt_path and save the new checkpoint.

        Args:
            runner (Runner): The runner of the training process.
            ckpt_path (str): The checkpoint path that ought to be published.
        """
        from mmengine.runner import save_checkpoint
        from mmengine.runner.checkpoint import _load_checkpoint
        checkpoint = _load_checkpoint(ckpt_path)
        assert self.published_keys is not None
        removed_keys = []
        for key in list(checkpoint.keys()):
            if key not in self.published_keys:
                removed_keys.append(key)
                checkpoint.pop(key)
        if removed_keys:
            logger.info(
                f'Key {removed_keys} will be removed because they are not '
                'found in published_keys. If you want to keep them, '
                f'please set `{removed_keys}` in published_keys',
                logger='current')
        checkpoint_data = pickle.dumps(checkpoint)
        sha = hashlib.sha256(checkpoint_data).hexdigest()
        final_path = osp.splitext(ckpt_path)[0] + f'-{sha[:8]}.pth'
        save_checkpoint(checkpoint, final_path)
        logger.info(
            f'The checkpoint ({ckpt_path}) is published to '
            f'{final_path}.',
            logger='current')