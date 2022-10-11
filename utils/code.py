
from datetime import datetime
import os
import shutil
import pathlib
from typing import Optional, List


def duplicate_code(source_dir: pathlib.Path, dest_dir: pathlib.Path,
                   excluded_local_subdirs: Optional[List[str]] = None,
                   exclude_git=True, exclude_notebooks=True, exclude_pycharm=True):
    """
    Copies all files from a folder that is supposed to hold python code (and other assets).
    Destination folder will be erased.
    """
    # Remove dest folder if exists
    t_start = datetime.now()
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.mkdir(dest_dir)
    excluded_subdirs = list() if excluded_local_subdirs is None else excluded_local_subdirs
    # Copy, we analyze the first level only
    for p in source_dir.glob("./*"):
        if p.is_dir():
            # __pycache__ and ipynb checkpoints won't be removed from sub-directories...
            if p.name in excluded_subdirs or (exclude_git and p.name == '.git')\
                    or (exclude_pycharm and (p.name in ['__pycache__', '.pytest_cache', '.idea']))\
                    or (exclude_notebooks and p.name == '.ipynb_checkpoints')\
                    or p == dest_dir:
                continue
            shutil.copytree(p, dest_dir.joinpath(p.name))
        elif p.is_file():
            if (exclude_notebooks and p.suffix == '.ipynb') or (exclude_git and p.name.startswith('.git')):
                continue
            shutil.copy(p, dest_dir.joinpath(p.name))
    print("Copied files and folders in {:.1f}s".format((datetime.now()-t_start).total_seconds()))

