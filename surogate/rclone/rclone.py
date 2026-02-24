import os
import subprocess
import time
from functools import wraps
from pathlib import Path
from shutil import which
from subprocess import Popen


def __check_installed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not is_installed():
            raise Exception(
                "rclone is not installed on this system. Please install it here: https://rclone.org/"
            )

        return func(*args, **kwargs)

    return wrapper


def is_installed() -> bool:
    """
    :return: True if rclone is correctly installed on the system.
    """
    return which("rclone") is not None


def is_folder_mounted(mount_path: str) -> bool:
    """
    :param mount_path: The path to check if it is mounted.
    :return: True if the folder is mounted.
    """
    try:
        with open("/proc/mounts", "r") as mounts_file:
            for mount in mounts_file:
                # match whole mount point (split by space)
                parts = mount.split()
                if len(parts) >= 2 and parts[1] == mount_path:
                    return True
    except FileNotFoundError:
        pass
    return False

def stop_mount(process: Popen, dest_folder: str) -> None:
    """
    Terminates a subprocess.Popen process.
    :param dest_folder: destination folder to unmount.
    :param process: The process to terminate.
    """
    process.terminate()
    try:
        process.wait(5)
    except subprocess.TimeoutExpired:
        process.kill()

    unmount_folder(dest_folder)

def unmount_folder(mount_path: str) -> bool:
    """
    :param mount_path: The path to unmount.
    :return: The process of the unmount command.
    """
    process = subprocess.Popen(
        args=["fusermount", "-u", mount_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        process.wait(5)
    except subprocess.TimeoutExpired:
        process.kill()
    return process.returncode


def rclone_mount(
        repo_id: str,
        branch: str,
        dest_folder: str,
        timeout: int = 5,
        check_interval: int = 1,
) -> subprocess.Popen:
    if not is_installed():
        raise Exception("rclone is not installed. Install: https://rclone.org/")

    dest_path = Path(dest_folder)
    dest_path.mkdir(parents=True, exist_ok=True)
    if is_folder_mounted(dest_folder):
        if unmount_folder(dest_folder) != 0:
            raise Exception(f"Folder {dest_folder} is already mounted and unmounting failed.")

    command = [
        "rclone",
        "mount",
        "--read-only",
        "--no-modtime",
        "--vfs-cache-mode=full",
        "--vfs-read-chunk-size=16M",
        "--vfs-read-chunk-streams=16",
        f"lakefs:{repo_id}/{branch}",
        str(dest_path),
    ]

    process = subprocess.Popen(
        args=command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True
    )

    start = time.time()
    while True:
        rc = process.poll()
        if rc is not None:
            stderr = process.stderr.read() if process.stderr else ""
            raise Exception(f"rclone exited early with code {rc}. stderr: {stderr}")

        try:
            if is_folder_mounted(dest_folder):
                with os.scandir(dest_path) as it:
                    if any(True for _ in it):
                        return process
        except FileNotFoundError:
            pass

        if time.time() - start >= timeout:
            process.kill()
            try:
                process.wait(3)
            except subprocess.TimeoutExpired:
                process.kill()
            unmount_folder(dest_folder)
            raise Exception(f"Mount did not succeed within {timeout}s. rclone process killed.")

        time.sleep(check_interval)
