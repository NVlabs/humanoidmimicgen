import argparse
import os
from pathlib import Path

from termcolor import colored
from tqdm import tqdm

import robocasa
import urllib.request

from robocasa.utils.dataset_registry import (
    MULTI_STAGE_TASK_DATASETS,
    SINGLE_STAGE_TASK_DATASETS,
    get_ds_path,
)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, download_dir, fname=None, check_overwrite=True):
    """
    First checks that @url is reachable, then downloads the file
    at that url into the directory specified by @download_dir.
    Prints a progress bar during the download using tqdm.

    Modified from https://github.com/tqdm/tqdm#hooks-and-callbacks, and
    https://stackoverflow.com/a/53877507.

    Args:
        url (str): url string
        download_dir (str): path to directory where file should be downloaded
        check_overwrite (bool): if True, will sanity check the download fpath to make sure a file of that name
            doesn't already exist there
    """

    # check if url is reachable. We need the sleep to make sure server doesn't reject subsequent requests
    # assert url_is_alive(url), "@download_url got unreachable url: {}".format(url)
    # time.sleep(0.5)

    if fname is None:
        # infer filename from url link
        fname = url.split("/")[-1]
    file_to_write = os.path.join(download_dir, fname)

    # If we're checking overwrite and the path already exists,
    # we ask the user to verify that they want to overwrite the file
    if check_overwrite and os.path.exists(file_to_write):
        user_response = input(f"Warning: file {file_to_write} already exists. Overwrite? y/n ")
        assert user_response.lower() in {
            "yes",
            "y",
        }, f"Did not receive confirmation. Aborting download."

    print(colored(f"Downloading to {file_to_write}", "yellow"))

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=fname) as t:
        urllib.request.urlretrieve(url, filename=file_to_write, reporthook=t.update_to)


def download_datasets(tasks, ds_types, overwrite=False, dryrun=False):
    if tasks is None:
        tasks = list(SINGLE_STAGE_TASK_DATASETS.keys()) + list(MULTI_STAGE_TASK_DATASETS.keys())

    for task_name in tasks:
        for ds_type in ds_types:
            print(colored(f"Task: {task_name}\nDataset type: {ds_type}", "yellow"))

            ds_path, ds_info = get_ds_path(task_name, ds_type, return_info=True)
            if ds_path is None:
                print(
                    colored(
                        f"No dataset for this task and dataset type exists!\nSkipping.\n",
                        "yellow",
                    )
                )
                continue
            ds_dir = "/".join(ds_path.split("/")[0:-1])
            fname = ds_path.split("/")[-1]

            Path(ds_dir).mkdir(parents=True, exist_ok=True)

            if overwrite is False and os.path.exists(ds_path):
                print(colored(f"Dataset already exists under {ds_path}\nSkipping.\n", "yellow"))
                continue

            if not dryrun:
                download_url(
                    url=ds_info["url"],
                    download_dir=ds_dir,
                    fname=fname,
                    check_overwrite=(overwrite is False),
                )
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Tasks to download datasets for. Defaults to all tasks",
    )

    parser.add_argument(
        "--ds_types",
        type=str,
        nargs="+",
        default=["human_raw", "human_im"],
        choices=["human_raw", "human_im", "mg_im"],
        help="Dataset types. Choose one or multiple options among human_raw, human_im, mg_im",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="automatically overwrite any existing files",
    )

    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="simulate without downloading datasets",
    )

    args = parser.parse_args()

    ans = input("This script may download several Gb of data. Proceed? (y/n) ")
    if ans == "y":
        print("Proceeding...")
    else:
        print("Aborting.")
        exit()

    download_datasets(
        tasks=args.tasks,
        ds_types=args.ds_types,
        overwrite=args.overwrite,
        dryrun=args.dryrun,
    )
