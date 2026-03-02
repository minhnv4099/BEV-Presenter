#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from os import path as osp
import copy
from typing import Optional, Union, Sequence, List, Callable
from collections import OrderedDict, defaultdict
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from src.structures.data_container import DataContainer as DC
from src.registry import DATASETS
from src.structures.bbox_3d import LiDARInstance3DBoxes, CameraInstance3DBoxes
from src.structures.bbox_3d.utils import get_box_type
from src.utils.logging import getLogger
from src.utils.fileio import load
from .nuscnes_eval import CustomNuScenesEval
from .det3d_dataset import Det3DDataset
from .compose import Compose

logger = getLogger(__name__)
TrainSampleType = dict[str, Union[torch.Tensor, list[dict]]]
TestSampleType = dict[str, list[Union[torch.Tensor, list[dict]]]]


@DATASETS.register_module()
class CustomNuScenesDataset(Dataset):
    r"""NuScenes Dataset.

    This dataset only adds camera intrinsics and extrinsics to the results.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to 'detection_cvpr_2019'.
        use_valid_flag (bool): Whether to use `use_valid_flag` key in the info
            file as mask to filter gt_boxes and gt_names. Defaults to False.
    """
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    AttrMapping = {
        'cycle.with_rider': 0,
        'cycle.without_rider': 1,
        'pedestrian.moving': 2,
        'pedestrian.standing': 3,
        'pedestrian.sitting_lying_down': 4,
        'vehicle.moving': 5,
        'vehicle.parked': 6,
        'vehicle.stopped': 7,
    }
    AttrMapping_rev = [
        'cycle.with_rider',
        'cycle.without_rider',
        'pedestrian.moving',
        'pedestrian.standing',
        'pedestrian.sitting_lying_down',
        'vehicle.moving',
        'vehicle.parked',
        'vehicle.stopped',
    ]
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')

    def __init__(
        self,
        ann_file: str,
        data_root: str,
        queue_length: int = 4,
        bev_size: tuple[int, int] = (200, 200),
        overlap_test: bool = False,
        debug_pipeline: bool = False,
        modality: dict = dict(use_lidar=False, use_camera=True),
        pipeline: List[Union[dict, Callable]] = (),
        test_mode: bool = False,
        with_velocity: bool = True,
        load_type: str = 'fov_image_based',
        box_type_3d: str = "CAMERA",
        classes: list[str] = None,
        frames: Sequence[int] = (),
        filter_empty_gt: bool = True,
        *args, **kwargs
    ):
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.data_infos = self.load_data_infos()

        self.pipeline = Compose(pipeline, debug=debug_pipeline)
        self.modality = modality
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.test_mode = test_mode
        self.class_names = classes
        self.frames = frames
        self.filter_empty_gt = filter_empty_gt

        self.name2idx = {
            name: idx
            for idx, name in enumerate(classes)
        }
        self.idx2name = {
            idx: name
            for idx, name in self.name2idx.items()
        }

        assert load_type in ('frame_based', 'mv_image_based', 'fov_image_based')
        self.load_type = load_type
        self.with_velocity = with_velocity
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

    def load_data_infos(self):
        return load(self.ann_file)['infos']

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx: int):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)

        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                new_idx = np.random.randint(0, len(self))
                while new_idx == idx:
                    new_idx = np.random.randint(0, len(self))

                idx = new_idx
                continue

            return data

    def pre_pipeline(self, results):
        """Initialization before data preparation.
        Args:
            results (dict): Dict before data preprocessing.
                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results['img_prefix'] = ''
        results['seg_prefix'] = ''
        results['proposal_file'] = ''
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def prepare_test_data(self, index) -> TestSampleType:
        """Prepare data for testing with augmentation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data of the corresponding index.
            Dict with keys:

                - `img`: list of `n_aug` tensor shape of `(n_queue, n_cam, c, h, w)`.
                - `img_metas`: list of `n_aug` dict of `n_queue`. See `med:'get_data_info'` for fields in each element.
        """
        # another way to gather data queue
        data_queue: OrderedDict[int, list[dict]] = OrderedDict()
        input_dict = self.get_data_info(index)
        cur_scene_token = input_dict['scene_token']
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        data_queue[0] = example

        for frame_idx in self.frames:
            chosen_idx = index + frame_idx
            if frame_idx == 0 or chosen_idx < 0 or chosen_idx >= len(self.data_infos):
                continue
            input_dict = self.get_data_info(chosen_idx)
            if input_dict['scene_token'] == cur_scene_token:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                data_queue[frame_idx] = example

        data_queue = OrderedDict(sorted(data_queue.items(), reverse=True))
        ret = defaultdict(list)
        n_aug = len(data_queue[0])

        for i in range(n_aug):
            single_aug_data_queue = []
            for t in data_queue.keys():
                single_aug_data_queue.append(data_queue[t][i])

            single_aug_sample = self.union2one(single_aug_data_queue)
            for key, value in single_aug_sample.items():
                ret[key].append(value)

        return dict(ret)

    def prepare_train_data(self, index: int) -> dict[str, Union[torch.Tensor, list]]:
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
            Dict with keys:

                - `img`: tensor shape of `(n_queue, n_cam, c, h, w)`.
                - `img_metas`: dict of `n_queue`. See :med:`get_data_info` for fields in each element.
                - `gt_bboxes_3d`: group truth bbox.
                - `gt_labels_3d`: ground truth labels.
        """
        queue = []
        index_list = list(range(index - self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)

        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None

            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)

            if self.filter_empty_gt and (example is None or len(example['gt_labels_3d']) == 0):
                return None
            queue.append(example)

        return self.union2one(queue)

    def get_data_info(self, index: int):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
                - img_filename (list): List of num_cam.
                - lidar2img (list): List of num_cam.
                - cam_intrinsic (list): List of num_cam.
                - lidar2cam (list): List of num_cam.
        """
        data_info = self.data_infos[index]
        input_dict = dict(
            pts_filename=data_info['lidar_path'],
            sample_idx=data_info['token'],
            prev_idx=data_info['prev'],
            next_idx=data_info['next'],
            can_bus=data_info['can_bus'],
            frame_idx=data_info['frame_idx'],
            sweeps=data_info['sweeps'],
            scene_token=data_info['scene_token'],
            ego2global_translation=data_info['ego2global_translation'],
            ego2global_rotation=data_info['ego2global_rotation'],
            timestamp=data_info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in data_info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']

        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        if not self.test_mode:
            input_dict['ann_info'] = self.parse_ann_info(data_info)

        return input_dict

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): 3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        # ann_info = super().parse_ann_info(info)
        # if ann_info is not None:
        #     ann_info = self._filter_with_mask(ann_info)
        ann_info = dict()
        n_instance = len(info['gt_boxes'])

        if n_instance != 0:
            gt_bboxes_3d = info['gt_boxes']
            gt_velocities = info['gt_velocity']
            if self.with_velocity:
                nan_mask = np.isnan(gt_velocities[:, 0])
                gt_velocities[nan_mask] = [0.0, 0.0]
                gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocities], axis=-1)

            # TODO: convert string -> index
            gt_labels_3d = [self.name2idx.get(name, -1) for name in info['gt_names']]
            gt_labels_3d = np.array(gt_labels_3d, dtype=np.int64)
            attr_labels = np.array(info['gt_names'], dtype=np.str_)

            # filter out no need labels and bbox
            mask_labels = gt_labels_3d != -1

            ann_info['gt_bboxes_3d'] = gt_bboxes_3d[mask_labels]
            ann_info['gt_labels_3d'] = gt_labels_3d[mask_labels]
            ann_info['attr_labels'] = attr_labels[mask_labels]
        # empty instance
        else:
            if self.with_velocity:
                ann_info['gt_bboxes_3d'] = np.zeros((0, 9), dtype=np.float32)
            else:
                ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)
            ann_info['attr_labels'] = np.array(0, dtype=np.str_)

        if self.load_type in ['fov_image_based', 'mv_image_based']:
            ann_info['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
            ann_info['gt_bboxes_labels'] = np.array(0, dtype=np.int64)
            # ann_info['attr_labels'] = np.array(0, dtype=np.int64)
            ann_info['centers_2d'] = np.zeros((0, 2), dtype=np.float32)
            ann_info['depths'] = np.zeros((0, ), dtype=np.float32)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # TODO: Unify the coordinates
        if self.load_type in ['fov_image_based', 'mv_image_based']:
            gt_bboxes_3d = CameraInstance3DBoxes(
                ann_info['gt_bboxes_3d'],
                box_dim=ann_info['gt_bboxes_3d'].shape[-1],
                origin=(0.5, 0.5, 0.5))
        else:
            gt_bboxes_3d = LiDARInstance3DBoxes(
                ann_info['gt_bboxes_3d'],
                box_dim=ann_info['gt_bboxes_3d'].shape[-1],
                origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        ann_info['gt_bboxes_3d'] = gt_bboxes_3d

        return ann_info

    def _filter_with_mask(self, ann_info: dict) -> dict:
        """Remove annotations that do not need to be cared.

        Args:
            ann_info (dict): Dict of annotation infos.

        Returns:
            dict: Annotations after filtering.
        """
        return ann_info
        filtered_annotations = {}
        if self.use_valid_flag:
            filter_mask = ann_info['bbox_3d_isvalid']
        else:
            filter_mask = ann_info['num_lidar_pts'] > 0
        for key in ann_info.keys():
            if key != 'instances':
                filtered_annotations[key] = (ann_info[key][filter_mask])
            else:
                filtered_annotations[key] = ann_info[key]
        return filtered_annotations

    def union2one(self, queue: list[dict]):
        """Combine list of samples into dict.
        Only operate keys `img` and `img_metas`, keep the rest untouched.

        Args:
            queue (list[dict]): List of dict, each presents for one sample.
        Returns:
            dict: A dictionary with combined keys:

                - `img`: tensor shape of `(n_queue, n_cam, c, h, w)`.
                - `img_metas`: dict of `n_queue`. See `med:'get_data_info'` for fields in each element.
                - And the rest.
        """
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data if isinstance(each['img_metas'], DC) else each['img_metas']
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        imgs_list = [torch.from_numpy(np.array(each['img'])) for each in queue]
        # still keep remaining keys 'gt_bboxes_3d', 'gt_labels_3d' in the last element queue
        # combine img and img_meta
        queue[-1]['img'] = torch.stack(imgs_list, dim=0).permute(0, 1, 4, 2, 3)
        queue[-1]['img_metas'] = metas_map

        return queue[-1]
