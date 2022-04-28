import errno
import json
import os
from typing import Dict


def _create_directory_if_not_exist(file_path: str) -> None:
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def read_dict(file_path: str) -> Dict:
    """ Read dictionary from path"""
    with open(file_path) as f:
        return json.load(f)
