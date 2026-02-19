# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Minh Nguyen
# ---------------------------------------------
import os.path
from os import path as osp
import sys
sys.path.append('.')

import argparse
import nuscenes_converter
from src.utils.logging import getLogger

logger = getLogger(__name__)


def nuscenes_data_prep(root_path: str,
                       can_bus_root_path: str,
                       info_prefix: str,
                       version: str,
                       dataset_name: str,
                       out_dir: str,
                       max_sweeps: int = 10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        can_bus_root_path (str): Path of can bus.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    if version == 'v1.0-test':
        info_test_path = osp.join(out_dir, f'{info_prefix}_infos_temporal_test.pkl')
        nuscenes_converter.export_2d_annotation(root_path, info_test_path, version=version)
    else:
        info_train_path = osp.join(out_dir, f'{info_prefix}_infos_temporal_train.pkl')
        info_val_path = osp.join(out_dir, f'{info_prefix}_infos_temporal_val.pkl')

        nuscenes_converter.export_2d_annotation(root_path, info_train_path, version=version)
        nuscenes_converter.export_2d_annotation(root_path, info_val_path, version=version)
        # create_groundtruth_database(dataset_name, root_path, info_prefix, f'{out_dir}/{info_prefix}_infos_train.pkl')


def get_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        'dataset',
        metavar='data-name',
        default='nuscenes',
        help='name of the dataset'
    )
    parser.add_argument(
        '--root-path',
        type=str,
        default='data/nuscenes/',
        help='specify the root path of dataset')
    parser.add_argument(
        '--canbus',
        type=str,
        default='./data/nuscenes/can_bus',
        help='specify the root path of nuScenes canbus')
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0-mini',
        required=False,
        help='specify the dataset version. Default to v1.0-mini')
    parser.add_argument(
        '--max-sweeps',
        type=int,
        default=10,
        required=False,
        help='specify sweeps of lidar per example')
    parser.add_argument(
        '--out-dir',
        type=str,
        default=None,
        required=False,
        help='dir to save files')
    parser.add_argument('--extra-tag', type=str, default='nuscenes', help='prefix of paths what would be saved')
    parser.add_argument('--workers', type=int, default=4, help='number of threads to be used')

    return parser.parse_args()


def main():
    args = get_args()
    if not args.root_path.rstrip(osp.sep).endswith(args.version):
        root_path = str(os.path.join(args.root_path, args.version))

    if args.out_dir is None:
        args.out_dir = root_path

    if args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)

        test_version = f'{args.version}-test'
        nuscenes_data_prep(
            root_path=root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)


if __name__ == "__main__":
    main()
