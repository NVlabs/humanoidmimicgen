"""
Legacy Script: This downloader is no longer actively maintained.
Consider using assetkit instead.
"""

import subprocess
import sys
from termcolor import colored

LEGACY_NOTICE = colored(
    "[WARNING] This is a legacy script.\nPlease switch to using `assetkit` commands directly.",
    "yellow",
)

INSTALL_INSTRUCTIONS = colored(
    "[ERROR] Failed to execute `assetkit pull`.\n"
    "Make sure AssetKit is installed.\n\n"
    "You can install it from:\n"
    "https://gitlab-master.nvidia.com/ncherniadev/assetkit",
    "red",
)


def download():
    print(LEGACY_NOTICE)

    try:
        subprocess.run(["assetkit", "pull"], check=True)
    except FileNotFoundError:
        print(INSTALL_INSTRUCTIONS)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(INSTALL_INSTRUCTIONS)
        sys.exit(e.returncode)


if __name__ == "__main__":
    download()
