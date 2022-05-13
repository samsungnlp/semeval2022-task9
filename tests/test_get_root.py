import unittest
import pathlib

from src.get_root import get_root


class TestGetRoot(unittest.TestCase):
    def test_get_root(self):
        root = get_root()
        self.assertTrue(pathlib.Path(root).is_dir())
