import os
import pathlib


def get_root() -> str:
    """
    :return: abs path to the repo dir
    """
    p = pathlib.Path(__file__).parent.parent.absolute()
    return str(p) + os.path.sep
