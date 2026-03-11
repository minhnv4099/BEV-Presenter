#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
import os
import sys

import huggingface_hub

sys.path.append(os.getcwd())
from os import path as osp
from huggingface_hub import HfApi
from argparse import ArgumentParser, Namespace
from src.utils.logging import getLogger
import tarfile

logger = getLogger(__name__)

DEFAULT_DATA_DIR = "data"


def safe_extract(tar: str, out_dir: str, members=None):
    with tarfile.open(tar, 'r:*') as tar:
        for member in tar.getmembers():
            # member_path = os.path.join(out_dir, member.name)
            # if not os.path.abspath(member_path).startswith(os.path.abspath(path)):
            #     raise Exception("Attempted Path Traversal in Tar File")
            # tar.extractall(out_dir)
            tar.extract(member=member, path=out_dir)


def get_args():
    parser = ArgumentParser(
        description="Tool to download nuscene data (.tar file) from hugging face repo hub "
                    "and extract items.")
    parser.add_argument(
        '--repo',
        default="5421Project/nuscene",
        help="Repo where to download data from.")
    parser.add_argument(
        '--token',
        required=False,
        default=None,
        help='Token to access the repo. The token can be used through 3 ways:'
             '1. (Highly recommend) Past token to file `~/.cache/huggingface/token` (in Linux)\n'
             '2. Set to environment variable `HF_TOKEN`\n'
             '3. Pass directly by command.')
    parser.add_argument(
        '--out-dir',
        default=None,
        help="Directory to save downloaded file.")
    parser.add_argument(
        '--cache-dir',
        default=None,
        help="Directory to cache downloaded files/folders.")
    parser.add_argument(
        '--version',
        default='v1.0-mini',
        choices=['v1.0-mini', 'v1.0-base'],
        help="Version of data. Default to 'v1.0-mini'.")
    parser.add_argument(
        '--flag',
        default="nuscenes",
        help="Flag of the data (subdir in out dir).")

    return parser.parse_args()


def main(args: Namespace):
    args.out_dir = args.out_dir or DEFAULT_DATA_DIR
    flag_out_dir = osp.join(args.out_dir, args.flag)
    os.makedirs(flag_out_dir, exist_ok=True)

    args.token = args.token or huggingface_hub.get_token()

    if args.token is None:
        token_link = 'https://docs.google.com/document/d/129TDtn83w0sZky860JnfVmeleP7Bv74rjDI1J3dz83M/edit?usp=sharing'
        logger.error(
            f'Used token in {token_link!r} to access '
            f'repo https://huggingface.co/datasets/{args.repo}. '
            'The token can be used through 3 ways: \n'
            '1. (Highly recommend) Past token to file `~/.cache/huggingface/token` (in Linux)\n'
            '2. Set to environment variable `HF_TOKEN`\n'
            '3. Pass directly by command.')

        assert False, "Do not provide token access to the repo."

    msg = (
        "Preparing downloading data:"
        f"\n\tfrom repo: {args.repo},"
        f"\n\tversion: {args.version},"
        f"\n\tsaved in: {flag_out_dir}."
    )
    logger.info(msg)

    hfapi = HfApi()

    data_path = hfapi.hf_hub_download(
        repo_id=args.repo,
        repo_type='dataset',
        filename=f"{args.version}.tar",
        local_dir=flag_out_dir,
        cache_dir=args.cache_dir,
        token=args.token
    )
    safe_extract(tar=data_path, out_dir=osp.join(flag_out_dir, args.version))
    logger.info(f"Nuscenes data is saved in {osp.join(flag_out_dir, args.version)!r}.")

    data_path = hfapi.hf_hub_download(
        repo_id=args.repo,
        repo_type='dataset',
        filename="can_bus.tar",
        local_dir=flag_out_dir,
        cache_dir=args.cache_dir,
        token=args.token
    )
    safe_extract(tar=data_path, out_dir=flag_out_dir)
    logger.info(f"Can bus data is saved in {osp.join(flag_out_dir, 'can_bus')!r}.")


if __name__ == "__main__":
    arguments = get_args()
    main(arguments)
